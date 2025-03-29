import time
from datetime import datetime
import tensorflow as tf
from absl import app
from absl import flags
import keras
import math
import json
import hashlib
import numpy as np
import tf_shell
import tf_shell_ml
import os
import signal
import sys
import keras_tuner as kt
import shutil
from experiment_utils import (
    features_party_job,
    labels_party_job,
    TensorBoard,
    LRWarmUp,
)
from noise_multiplier_finder import search_noise_multiplier

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
flags.DEFINE_float("beta_1", 0.8, "Beta 1 for Adam optimizer")
flags.DEFINE_float("epsilon", 1.0, "Differential privacy parameter")
flags.DEFINE_integer("epochs", 10, "Number of epochs")
flags.DEFINE_enum(
    "party",
    "b",
    ["f", "l", "b"],
    "Which party is this, `f` `l`, or `b`, for feature, label, or both.",
)
flags.DEFINE_bool("gpu", False, "Offload jacobain computation to GPU on features party")
flags.DEFINE_string(
    "cluster_spec",
    f"""{{
  "{features_party_job}": ["localhost:2222"],
  "{labels_party_job}": ["localhost:2223"],
}}""",
    "Cluster spec",
)
flags.DEFINE_integer("backprop_cleartext_sz", 23, "Cleartext size for backpropagation")
flags.DEFINE_integer("backprop_scaling_factor", 4, "Scaling factor for backpropagation")
flags.DEFINE_integer("backprop_noise_offset", 16, "Noise offset for backpropagation")
flags.DEFINE_integer("noise_cleartext_sz", 26, "Cleartext size for noise")
flags.DEFINE_integer("noise_noise_offset", 0, "Noise offset for noise")
flags.DEFINE_bool("eager_mode", False, "Eager mode")
flags.DEFINE_bool("plaintext", False, "Run without encryption or masking (but with noise).")
flags.DEFINE_bool("check_overflow", False, "Check for overflow in the protocol.")
flags.DEFINE_bool("tune", False, "Tune hyperparameters (or use default values).")
FLAGS = flags.FLAGS


class HyperModel(kt.HyperModel):
    def __init__(self, labels_party_dev, features_party_dev, jacobian_devs, cache_path, num_examples):
        super().__init__()
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.jacobian_devs = jacobian_devs
        self.cache_path = cache_path
        self.num_examples = num_examples

    def hp_hash(self, hp_dict):
        """Returns a stable short hash for a dictionary of hyperparameter values."""
        # Convert dict to canonical JSON string and hash it
        hp_dict["epsilon"] = FLAGS.epsilon
        hp_dict["eager_mode"] = FLAGS.eager_mode
        hp_dict["plaintext"] = FLAGS.plaintext
        data = json.dumps(hp_dict, sort_keys=True)
        return hashlib.md5(data.encode("utf-8")).hexdigest()[:8]

    def get_cache_filename(self, hp_dict):
        """Returns a path to use for caching these hyperparameters."""
        os.makedirs(self.cache_path, exist_ok=True)
        return os.path.join(self.cache_path, f"{self.hp_hash(hp_dict)}.lock")

    def fit(self, hp, model, *args, **kwargs):
        hp_values = hp.values
        cache_file = self.get_cache_filename(hp_values)

        # If the cache file already exists, it means we crashed previously.
        # Read the fail_count from the file.
        fail_count = 0
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                # Read the existing failure count (default 0 if empty)
                content = f.read().strip()
                fail_count = int(content) if content else 0

                if fail_count >= 1:
                    raise RuntimeError(f"Skipping trial with HP={hp_values} after prior failures.")
                else:
                    fail_count += 1

        # Overwrite the number of failures
        with open(cache_file, "w") as f:
            f.write(str(fail_count))

        return super().fit(hp, model, *args, **kwargs)

    def build(self, hp):
        # Define functions which generate encryption context for the
        # backpropagation and noise parts of the protocol. When not executing
        # eagerly, autocontext can be used to automatically determine the
        # encryption parameters. When executing eagerly, parameters must be
        # specified manually (or simply copied from a previous run which uses
        # autocontext).
        log2_cleartext_sz=hp.Int("backprop_cleartext_sz", min_value=16, max_value=24, step=1, default=FLAGS.backprop_cleartext_sz)
        scaling_factor=hp.Choice("backprop_scaling_factor", values=[2, 4, 8, 16, 32], default=FLAGS.backprop_scaling_factor)
        noise_offset_log2=hp.Choice("backprop_noise_offset", values=[0, 8, 16, 32, 48], default=FLAGS.backprop_noise_offset)
        log2_cleartext_sz=hp.Int("noise_cleartext_sz", min_value=36, max_value=36, step=1, default=FLAGS.noise_cleartext_sz)
        noise_offset_log2=hp.Choice("noise_noise_offset", values=[0, 40], default=FLAGS.noise_noise_offset)
        # 0 and 40 correspond to ring degree of 2**12 and 2**13

        def backprop_context_fn(read_cache):
            if FLAGS.eager_mode:
                return tf_shell.create_context64(
                    log_n=12,
                    main_moduli=[36030591770263553, 18014745055510529],
                    plaintext_modulus=8404993,
                    scaling_factor=flags.FLAGS.backprop_scaling_factor,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=log2_cleartext_sz,
                    scaling_factor=scaling_factor,
                    noise_offset_log2=noise_offset_log2,
                    read_from_cache=read_cache,
                    cache_path=self.cache_path,
                )

        def noise_context_fn(read_cache):
            if FLAGS.eager_mode:
                return tf_shell.create_context64(
                    log_n=12,
                    main_moduli=[11016591278081, 20931523428353],
                    plaintext_modulus=67239937,
                    scaling_factor=1,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=log2_cleartext_sz,
                    noise_offset_log2=noise_offset_log2,
                    read_from_cache=read_cache,
                    cache_path=self.cache_path,
                )

        def noise_multiplier_fn(batch_size):
            # Set delta to 1/num_examples, rounded to nearest power of 10.
            target_delta = 10**int(math.floor(math.log10(1 / self.num_examples)))
            print(f"Target delta {target_delta}")
            return search_noise_multiplier(
                target_epsilon=FLAGS.epsilon,
                target_delta=target_delta,
                epochs=FLAGS.epochs,
                training_num_samples=self.num_examples,
                batch_size=batch_size,
            )

        # Create the model. When using DPSGD, you must use Shell* layers.
        model = tf_shell_ml.DpSgdSequential(
            layers=[
                tf_shell_ml.ShellDense(
                    100,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,
                ),
                tf_shell_ml.ShellDense(
                    2,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=backprop_context_fn,
            noise_context_fn=noise_context_fn,
            noise_multiplier_fn=noise_multiplier_fn,
            labels_party_dev=self.labels_party_dev,
            features_party_dev=self.features_party_dev,
            cache_path=self.cache_path,
            jacobian_devices=self.jacobian_devs,
            disable_encryption=FLAGS.plaintext,
            disable_masking=FLAGS.plaintext,
            check_overflow_INSECURE=FLAGS.check_overflow or FLAGS.tune,
        )

        model.build(input_shape=(None, 784))
        model.summary()

        # Learning rate warm up is good practice for large batch sizes.
        # see https://arxiv.org/pdf/1706.02677
        lr = hp.Choice("learning_rate", values=[0.1, 0.01, 0.001], default=FLAGS.learning_rate)
        lr_schedule = LRWarmUp(
            initial_learning_rate=lr,
            decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
                lr,
                decay_steps=1,
                decay_rate=1,  # No decay.
            ),
            warmup_steps=4,
        )

        beta_1 = hp.Choice("beta_1", values=[0.7, 0.8, 0.9], default=FLAGS.beta_1)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        return model


def main(_):
    # Allow killing server with control-c
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if FLAGS.party == "b":
        # Override the features and labels party devices.
        labels_party_dev = "/job:localhost/replica:0/task:0/device:CPU:0"
        features_party_dev = "/job:localhost/replica:0/task:0/device:CPU:0"
        jacobian_dev = None
        if FLAGS.gpu:
            num_gpus = len(tf.config.list_physical_devices("GPU"))
            jacobian_dev = [
                f"/job:localhost/replica:0/task:0/device:GPU:{i}"
                for i in range(num_gpus)
            ]

    else:
        # Set up the distributed training environment.
        features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
        labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"
        jacobian_dev = None
        if FLAGS.gpu:
            num_gpus = len(tf.config.list_physical_devices("GPU"))
            jacobian_dev = [
                f"/job:{features_party_job}/replica:0/task:0/device:GPU:{i}"
                for i in range(num_gpus)
            ]

        if FLAGS.party == "f":
            this_job = features_party_job
        else:
            this_job = labels_party_job

        print(FLAGS.cluster_spec)

        cluster = tf.train.ClusterSpec(eval(FLAGS.cluster_spec))

        server = tf.distribute.Server(
            cluster,
            job_name=this_job,
            task_index=0,
        )

        tf.config.experimental_connect_to_cluster(cluster)

        # The labels party just runs the server while the training is driven by
        # the features party.
        if this_job == labels_party_job:
            print(f"{this_job} server started.", flush=True)
            server.join()  # Wait for the features party to finish.
            exit(0)

    # Set up training data.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
    x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))

    # Create masks for digits 3 and 8
    train_mask = (y_train == 3) | (y_train == 8)
    test_mask = (y_test == 3) | (y_test == 8)

    # Filter the datasets
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    # Relabel 3 as 0 and 8 as 1 if you want binary classification
    y_train = (y_train == 8).astype(np.int32)
    y_test = (y_test == 8).astype(np.int32)

    # Convert to one-hot encoding.
    y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)

    num_examples = len(x_train)
    print("Number of training examples:", num_examples)

    # Limit the number of features to reduce the memory footprint for testing.
    # x_train, x_test = x_train[:, :350], x_test[:, :350]

    tf.config.run_functions_eagerly(FLAGS.eager_mode)

    # Shuffle both x_train and y_train together so the order is the same across
    # parties. If x and y are shuffled separately, tf.Dataset does not suffle in
    # the same order even when the seed is the same.
    shuffle_seed = int(time.time())
    with tf.device(labels_party_dev):
        labels_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            # .shuffle(2**14, seed=shuffle_seed)  # Works with HE but not plaintext?
            .map(lambda x, y: y)
        )
        labels_dataset = labels_dataset.batch(2**12)

    with tf.device(features_party_dev):
        features_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            # .shuffle(2**14, seed=shuffle_seed)  # Works with HE but not plaintext?
            .map(lambda x, y: x)
        )
        features_dataset = features_dataset.batch(2**12)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        hypermodel = HyperModel(
            labels_party_dev=labels_party_dev,
            features_party_dev=features_party_dev,
            jacobian_devs=jacobian_dev,
            cache_path="cache-mnist-dpsgd-binary",
            num_examples=num_examples,
        )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/dpsgd-binary-{stamp}"
    tb = TensorBoard(
        log_dir=logdir,
        # ExperimentTensorBoard kwargs.
        party=FLAGS.party,
        gpu_enabled=FLAGS.gpu,
        num_gpus=len(tf.config.list_physical_devices("GPU")),
        cluster_spec=FLAGS.cluster_spec,
        target_delta=1e-4,
        training_num_samples=num_examples,
        epochs=FLAGS.epochs,
        # TensorBoard kwargs.
        write_steps_per_second=True,
        update_freq="batch",
        profile_batch=2,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    if FLAGS.tune:
        # Tune the hyperparameters.
        tuner = kt.RandomSearch(
            hypermodel,
            max_trials=60,
            objective=[
                kt.Objective('val_categorical_accuracy', direction='max'),
                kt.Objective('time', direction='min')
            ],
            directory="kerastuner",
            project_name="mnist-dpsgd-binary",
            max_consecutive_failed_trials=30,
        )
        tuner.search_space_summary()
        tuner.search(
            features_dataset,
            labels_dataset,
            epochs=FLAGS.epochs,
            validation_data=val_dataset,
            callbacks=[tb, stop_early],
        )
        tuner.results_summary()

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # Print everything in best_hps
        print("Best hyperparameters:")
        for key in best_hps.values:
            print(f"{key}: {best_hps.get(key)}")

    else:
        # Train the model.
        tuner = kt.GridSearch(
            hypermodel,
            max_trials=1,
            objective=[
                kt.Objective('val_categorical_accuracy', direction='max'),
                kt.Objective('time', direction='min')
            ],
            directory="kerastuner",
            project_name="default_hps",
            max_consecutive_failed_trials=1,
            overwrite=True,  # Always overwrite previous runs.
        )
        trial = tuner.oracle.create_trial("single_run_trial")

        # Remove the cache path to ignore errors from previous runs.
        dirpath = os.path.abspath("") + "/cache-mnist-dpsgd-binary"
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)

        tuner.run_trial(
            trial,
            features_dataset,
            labels_dataset,
            epochs=FLAGS.epochs,
            validation_data=val_dataset,
            callbacks=[tb, stop_early],
        )


if __name__ == "__main__":
    app.run(main)
