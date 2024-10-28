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
import dp_accounting
import signal
import sys

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"

flags.DEFINE_float("learning_rate", 0.15, "Learning rate for training")
flags.DEFINE_float("noise_multiplier", 1.00, "Noise multiplier for DP-SGD")
flags.DEFINE_integer("epochs", 15, "Number of epochs")
flags.DEFINE_enum(
    "party", "b", ["f", "l", "b"], "Which party is this, `f` `l`, or `b`, for feature, label, or both."
)
flags.DEFINE_string(
    "cluster_spec",
    f"""{{
  "{features_party_job}": ["localhost:2222"],
  "{labels_party_job}": ["localhost:2223"],
}}""",
    "Cluster spec",
)
FLAGS = flags.FLAGS


def compute_epsilon(steps, batch_size, num_samples, noise_multiplier, target_delta):
    """Computes epsilon value for given hyperparameters."""
    if FLAGS.noise_multiplier == 0.0:
        return float("inf")
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = batch_size / num_samples
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )

    accountant.compose(event)

    return accountant.get_epsilon(target_delta=target_delta)


def main(_):
    # Allow killing server with control-c
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if FLAGS.party == "b":
        # Override the features and labels party devices.
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0"
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0"

    else:
        # Set up the distributed training environment.
        features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
        labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"

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

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=2**14).batch(2**10)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(32)

    cache_path = "./cache-mnist-dpsgd-conv/"

    # Create the model.
    model = tf_shell_ml.PostScaleSequential(
        layers=[
            keras.layers.Conv2D(
                filters=16,
                kernel_size=8,
                strides=2,
                padding="SAME",
            ),
            keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=1,
            ),
            keras.layers.Conv2D(
                filters=32,
                kernel_size=4,
                strides=2,
            ),
            keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=1,
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(
                16,
                activation=tf.nn.relu,
            ),
            keras.layers.Dense(
                10,
                activation=tf.nn.softmax,
            ),
        ],
        backprop_context_fn=lambda: tf_shell.create_autocontext64(
            log2_cleartext_sz=23,
            scaling_factor=32,
            noise_offset_log2=14,
            cache_path=cache_path,
        ),
        noise_context_fn=lambda: tf_shell.create_autocontext64(
            log2_cleartext_sz=24,
            scaling_factor=1,
            noise_offset_log2=0,
            cache_path=cache_path,
        ),
        labels_party_dev=labels_party_dev,
        features_party_dev=features_party_dev,
        noise_multiplier=FLAGS.noise_multiplier,
        cache_path=cache_path,
        check_overflow_INSECURE=True,
    )

    model.build([None, 28, 28, 1])

    model.compile(
        shell_loss=tf_shell_ml.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/post-scale-{stamp}"
    tb = tf.keras.callbacks.TensorBoard(
        logdir,
        write_steps_per_second=True,
        update_freq="batch",
        profile_batch="2, 3",
    )
    print(f"To start tensorboard, run: tensorboard --logdir ./ --host 0.0.0.0")
    print(f"\ttensorboard profiling requires: pip install tensorboard_plugin_profile")

    # Train the model.
    history = model.fit(
        train_dataset,
        epochs=FLAGS.epochs,
        validation_data=val_dataset,
        callbacks=[tb],
    )

    print("Training complete.")
    batch_size = history.history["num_slots"][0] // 2
    samples_per_epoch = 60000 - (60000 % batch_size)

    # Compute the privacy budget expended.
    eps = compute_epsilon(
        steps=FLAGS.epochs * 60000 // batch_size,
        batch_size=batch_size,
        num_samples=samples_per_epoch,
        noise_multiplier=FLAGS.noise_multiplier,
        target_delta=1e-5,
    )
    print(f"Privacy budget expended: {eps}")


if __name__ == "__main__":
    app.run(main)
