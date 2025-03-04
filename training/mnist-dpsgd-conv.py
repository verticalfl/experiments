import time
from datetime import datetime
import tensorflow as tf
from absl import app
from absl import flags
import keras
import numpy as np
import tf_shell
import tf_shell_ml
import os
import signal
import sys
import keras_tuner as kt
from experiment_utils import (
    features_party_job,
    labels_party_job,
    ExperimentTensorBoard,
    LRWarmUp,
)

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
flags.DEFINE_float("noise_multiplier", 1.00, "Noise multiplier for DP-SGD")
flags.DEFINE_integer("epochs", 10, "Number of epochs")
flags.DEFINE_enum(
    "party", "b", ["f", "l", "b"], "Which party is this, `f` `l`, or `b`, for feature, label, or both."
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
flags.DEFINE_integer("backprop_scaling_factor", 2, "Scaling factor for backpropagation")
flags.DEFINE_integer("backprop_noise_offset", 48, "Noise offset for backpropagation")
flags.DEFINE_integer("noise_cleartext_sz", 38, "Cleartext size for noise")
flags.DEFINE_integer("noise_noise_offset", 31, "Noise offset for noise")
flags.DEFINE_bool("eager_mode", False, "Eager mode")
flags.DEFINE_bool("plaintext", False, "Run without encryption, noise, or masking.")
flags.DEFINE_bool("check_overflow", False, "Check for overflow in the protocol.")
flags.DEFINE_bool("tune", False, "Tune hyperparameters (or use default values).")
FLAGS = flags.FLAGS

# Clip the input images to make testing faster.
clip_by = 0

class HyperModel(kt.HyperModel):
    def __init__(self, labels_party_dev, features_party_dev, jacobian_devs, cache_path):
        super().__init__()
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.jacobian_devs = jacobian_devs
        self.cache_path = cache_path

    def build(self, hp):
        # Define functions which generate encryption context for the
        # backpropagation and noise parts of the protocol. When not executing
        # eagerly, autocontext can be used to automatically determine the
        # encryption parameters. When executing eagerly, parameters must be
        # specified manually (or simply copied from a previous run which uses
        # autocontext).
        def backprop_context_fn(read_cache):
            if FLAGS.eager_mode:
                return tf_shell.create_context64(
                    log_n=13,
                    main_moduli=[1152920548708581377, 1152918758512312321, 566803680264193, 568043046912001],
                    plaintext_modulus=8404993,
                    scaling_factor=flags.FLAGS.backprop_scaling_factor,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=hp.Int(
                        "backprop_cleartext_sz", min_value=16, max_value=26, step=1, default=FLAGS.backprop_cleartext_sz
                    ),
                    scaling_factor=hp.Choice(
                        "backprop_scaling_factor", values=[2, 4, 8, 16, 32], default=FLAGS.backprop_scaling_factor
                    ),
                    noise_offset_log2=hp.Choice(
                        "backprop_noise_offset", values=[0, 8, 16, 32, 48], default=FLAGS.backprop_noise_offset
                    ),
                    read_from_cache=read_cache,
                    cache_path=self.cache_path,
                )

        def noise_context_fn (read_cache):
            if FLAGS.eager_mode:
                # TODO: FIX
                return tf_shell.create_context64(
                    log_n=13,
                    main_moduli=[369295477609627649, 45036033854832641],
                    plaintext_modulus=274878136321,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=hp.Int(
                        "noise_cleartext_sz", min_value=36, max_value=36, step=1, default=FLAGS.noise_cleartext_sz
                    ),
                    noise_offset_log2=hp.Choice(
                        "noise_noise_offset", values=[0, 40], default=FLAGS.noise_noise_offset
                        # 0 and 40 correspond to ring degree of 2**12 and 2**13
                    ),
                    read_from_cache=read_cache,
                    cache_path=self.cache_path,
                )

        # Create the model. When using DPSGD, you must use Shell* layers. Note
        # this takes roughly an hour per batch!
        model = tf_shell_ml.DpSgdSequential(
            layers=[
                # Model from tensorflow-privacy tutorial. The first 2 layers may
                # be skipped and the model still has ~95% accuracy (plaintext,
                # no noise).
                tf_shell_ml.Conv2D(
                    filters=16,
                    kernel_size=4,
                    strides=2,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,  # Note: tf-shell specific
                ),
                tf_shell_ml.MaxPool2D(
                    pool_size=(2, 2),
                    strides=1,
                ),
                tf_shell_ml.Flatten(),
                tf_shell_ml.ShellDense(
                    32,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,  # Note: tf-shell specific
                ),
                tf_shell_ml.ShellDense(
                    10,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=backprop_context_fn,
            noise_context_fn=noise_context_fn,
            labels_party_dev=self.labels_party_dev,
            features_party_dev=self.features_party_dev,
            noise_multiplier=FLAGS.noise_multiplier,
            cache_path=self.cache_path,
            jacobian_devices=self.jacobian_devs,
            disable_encryption=FLAGS.plaintext,
            disable_masking=FLAGS.plaintext,
            disable_noise=FLAGS.plaintext,
            check_overflow_INSECURE=FLAGS.check_overflow or FLAGS.tune,
        )

        model.build([None, 28 - (2 * clip_by), 28 - (2 * clip_by), 1])
        model.summary()

        # Learning rate warm up is good practice for large batch sizes.
        # see https://arxiv.org/pdf/1706.02677
        lr = hp.Choice("learning_rate", values=[0.1, 0.01, 0.001], default=FLAGS.learning_rate)
        lr_schedule = LRWarmUp(
            initial_learning_rate=lr,
            decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
                lr,
                decay_steps=1,
                decay_rate=1,
            ),
            warmup_steps=32,
        )

        beta_1 = hp.Choice("beta_1", values=[0.7, 0.8, 0.9], default=0.8)
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
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0"
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0"
        jacobian_dev = None
        if FLAGS.gpu:
            num_gpus = len(tf.config.list_physical_devices('GPU'))
            jacobian_dev = [f"/job:localhost/replica:0/task:0/device:GPU:{i}" for i in range(num_gpus)]

    else:
        # Set up the distributed training environment.
        features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
        labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"
        jacobian_dev = None
        if FLAGS.gpu:
            num_gpus = len(tf.config.list_physical_devices('GPU'))
            jacobian_dev = [f"/job:{features_party_job}/replica:0/task:0/device:GPU:{i}" for i in range(num_gpus)]

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

        # The labels party just runs the server while the training is driven by the
        # features party.
        if this_job == labels_party_job:
            print(f"{this_job} server started.", flush=True)
            server.join()  # Wait for the features party to finish.
            exit(0)

    # Set up training data.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
    x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(
        x_test, (-1, 28, 28, 1)
    )
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    # Clip the input images to make testing faster.
    x_train = x_train[:, clip_by : (28 - clip_by), clip_by : (28 - clip_by), :]
    x_test = x_test[:, clip_by : (28 - clip_by), clip_by : (28 - clip_by), :]

    num_examples = len(x_train)
    print("Number of training examples:", num_examples)

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
            cache_path="cache-mnist-dpsgd-conv",
        )

        tuner = kt.Hyperband(
            hypermodel,
            objective=kt.Objective("val_categorical_accuracy", direction="max"),
            max_epochs=10,
            factor=3,
            directory="kerastuner",
            project_name="mnist-dpsgd-conv",
        )

        keras_hps = kt.HyperParameters()
        default_hp_model = tuner.hypermodel.build(keras_hps)


    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/dpsgd-conv-{stamp}"
    tb = ExperimentTensorBoard(
        log_dir=logdir,
        # ExperimentTensorBoard kwargs.
        noise_multiplier=FLAGS.noise_multiplier,
        party=FLAGS.party,
        gpu_enabled=FLAGS.gpu,
        num_gpus=len(tf.config.list_physical_devices('GPU')),
        layers=default_hp_model.layers,
        cluster_spec=FLAGS.cluster_spec,
        target_delta=1e-5,
        training_num_samples=num_examples,
        epochs=FLAGS.epochs,
        # TensorBoard kwargs.
        write_steps_per_second=True,
        update_freq="batch",
        profile_batch=2,
    )
    # Tensorboard callbacks only write hyperparammeters to the log if their
    # class is keras.callbacks.TensorBoard. This is a hack to make the
    # ExperimentTensorBoard class look like a keras.callbacks.TensorBoard
    # instance.
    tb.__class__ = keras.callbacks.TensorBoard

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    if FLAGS.tune:
        # Tune the hyperparameters.
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
        history = default_hp_model.fit(
            features_dataset,
            labels_dataset,
            epochs=FLAGS.epochs,
            validation_data=val_dataset,
            callbacks=[tb, stop_early],
        )


if __name__ == "__main__":
    app.run(main)
