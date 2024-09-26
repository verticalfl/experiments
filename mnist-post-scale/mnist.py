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

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"

flags.DEFINE_float("learning_rate", 0.15, "Learning rate for training")
flags.DEFINE_integer("epochs", 15, "Number of epochs")
flags.DEFINE_bool(
    "use_encryption", True, "Use Homomorphic Encryption (false is plaintext, insecure)"
)
flags.DEFINE_bool("fast_rotate", True, "Use fast rotation protocol")
flags.DEFINE_enum(
    "party", "f", ["f", "l"], "Which party is this, f or l, for features or labels"
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


def main(_):
    # Set up the distributed training environment.
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

        # Allow killing server with control-c
        import signal
        import sys

        def signal_handler(sig, frame):
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.pause()

        # Wait for the features party to finish.
        server.join()
        exit(0)

    # Set up training data.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
    x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=2**14).batch(2**10)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(32)

    # Create the model.
    model = tf_shell_ml.PostScaleSequential(
        [
            tf_shell_ml.ShellDense(
                64,
                activation=tf_shell_ml.relu,
                activation_deriv=tf_shell_ml.relu_deriv,
                use_fast_reduce_sum=FLAGS.fast_rotate,
            ),
            tf_shell_ml.ShellDense(
                10,
                activation=tf.nn.softmax,
                use_fast_reduce_sum=FLAGS.fast_rotate,
            ),
        ],
        # lambda: tf_shell.create_context64(
        #     log_n=12,
        #     main_moduli=[288230376151760897, 288230376152137729],
        #     plaintext_modulus=4294991873,
        #     scaling_factor=3,
        # ),
        lambda: tf_shell.create_autocontext64(
            log2_cleartext_sz=32,
            scaling_factor=3,
            noise_offset_log2=57,
        ),
        use_encryption=FLAGS.use_encryption,
        needs_public_rotation_key=not FLAGS.fast_rotate,
        labels_party_dev=labels_party_dev,
        features_party_dev=features_party_dev,
    )

    model.compile(
        shell_loss=tf_shell_ml.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fast_str = "-fast" if FLAGS.fast_rotate else ""
    logdir = os.path.abspath("") + f"/tflogs/post-scale{fast_str}-{stamp}"
    tb = tf.keras.callbacks.TensorBoard(
        logdir,
        write_steps_per_second=True,
        update_freq="batch",
        profile_batch=2,
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


if __name__ == "__main__":
    app.run(main)
