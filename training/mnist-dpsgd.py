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
from experiment_utils import features_party_job, labels_party_job, ExperimentTensorBoard

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
flags.DEFINE_integer("backprop_scaling_factor", 4, "Scaling factor for backpropagation")
flags.DEFINE_integer("backprop_noise_offset", 14, "Noise offset for backpropagation")
flags.DEFINE_integer("noise_cleartext_sz", 25, "Cleartext size for noise")
flags.DEFINE_integer("noise_noise_offset", 0, "Noise offset for noise")
FLAGS = flags.FLAGS


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

        # Set up the distributed training environment.
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
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    # Limit the number of features to reduce the memory footprint for testing.
    # x_train, x_test = x_train[:, :350], x_test[:, :350]

    num_examples = len(x_train)
    print("Number of filtered samples:", num_examples)

    # Shuffle both x_train and y_train together so the order is the same across
    # parties. If x and y are shuffled separately, tf.Dataset does not suffle in
    # the same order even when the seed is the same.
    shuffle_seed = int(time.time())
    with tf.device(labels_party_dev):
        labels_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(2**14, seed=shuffle_seed)
            .map(lambda x, y: y)
        )
        labels_dataset = labels_dataset.batch(2**10)

    with tf.device(features_party_dev):
        features_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(2**14, seed=shuffle_seed)
            .map(lambda x, y: x)
        )
        features_dataset = features_dataset.batch(2**10)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        cache_path = "./cache-mnist-dpsgd/"

        # Create the model. When using DPSGD, you must use Shell* layers.
        model = tf_shell_ml.DpSgdSequential(
            layers=[
                tf_shell_ml.ShellDense(
                    100,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,
                ),
                tf_shell_ml.ShellDense(
                    10,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=lambda read_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=flags.FLAGS.backprop_cleartext_sz,
                scaling_factor=flags.FLAGS.backprop_scaling_factor,
                noise_offset_log2=flags.FLAGS.backprop_noise_offset,
                read_from_cache=read_cache,
                cache_path=cache_path,
            ),
            noise_context_fn=lambda read_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=flags.FLAGS.noise_cleartext_sz,
                scaling_factor=1,
                noise_offset_log2=flags.FLAGS.noise_noise_offset,
                read_from_cache=read_cache,
                cache_path=cache_path,
            ),
            labels_party_dev=labels_party_dev,
            features_party_dev=features_party_dev,
            noise_multiplier=FLAGS.noise_multiplier,
            cache_path=cache_path,
            # jacobian_pfor=True,
            # jacobian_pfor_iterations=128,
            jacobian_devices=jacobian_dev,
            # check_overflow_INSECURE=True,
            # disable_encryption=True,
            # disable_masking=True,
            # disable_noise=True,
        )

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/dpsgd-{stamp}"
    tb = ExperimentTensorBoard(
        log_dir=logdir,
        # ExperimentTensorBoard kwargs.
        noise_multiplier=FLAGS.noise_multiplier,
        learning_rate=FLAGS.learning_rate,
        party=FLAGS.party,
        gpu_enabled=FLAGS.gpu,
        num_gpus=len(tf.config.list_physical_devices('GPU')),
        layers=model.layers,
        cluster_spec=FLAGS.cluster_spec,
        target_delta=1e-5,
        training_num_samples=60000,
        epochs=FLAGS.epochs,
        backprop_cleartext_sz=FLAGS.backprop_cleartext_sz,
        backprop_scaling_factor=FLAGS.backprop_scaling_factor,
        backprop_noise_offset=FLAGS.backprop_noise_offset,
        noise_cleartext_sz=FLAGS.noise_cleartext_sz,
        noise_noise_offset=FLAGS.noise_noise_offset,
        # TensorBoard kwargs.
        write_steps_per_second=True,
        update_freq="batch",
        profile_batch=2,
    )

    # Train the model.
    history = model.fit(
        features_dataset,
        labels_dataset,
        epochs=FLAGS.epochs,
        validation_data=val_dataset,
        callbacks=[tb],
    )


if __name__ == "__main__":
    app.run(main)
