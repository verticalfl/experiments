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
import network_mon_utils

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"

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
    # x_train, x_test = x_train[:, :300], x_test[:, :300]

    with tf.device(labels_party_dev):
        labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        labels_dataset = labels_dataset.batch(2**10)

    with tf.device(features_party_dev):
        features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        features_dataset = features_dataset.batch(2**10)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        cache_path = "./cache-mnist-postscale/"

        # Create the model. When using post scale, you can use either Shell*
        # layers or standard Keras layers.
        model = tf_shell_ml.PostScaleSequential(
            layers=[
                keras.layers.Dense(100, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax),
            ],
            backprop_context_fn=lambda read_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=23,
                scaling_factor=32,
                noise_offset_log2=14,
                read_from_cache=read_cache,
                cache_path=cache_path,
            ),
            noise_context_fn=lambda read_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=25,
                scaling_factor=1,
                noise_offset_log2=0,
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
        )

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=FLAGS.learning_rate,
            decay_steps=10,
            decay_rate=0.9,
        )

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/post-scale-{stamp}"
    tb = tf.keras.callbacks.TensorBoard(
        logdir,
        write_steps_per_second=True,
        update_freq="batch",
        profile_batch=2,
    )
    # Write some metadata to the logs.
    file_writer = tf.summary.create_file_writer(logdir + "/metadata")
    file_writer.set_as_default()
    tf.summary.scalar("noise_multiplier", FLAGS.noise_multiplier, step=0)
    tf.summary.scalar("learning_rate", FLAGS.learning_rate, step=0)
    tf.summary.text("party", FLAGS.party, step=0)
    tf.summary.scalar("gpu_enabled", FLAGS.gpu, step=0)
    tf.summary.scalar("num_gpus", len(tf.config.list_physical_devices('GPU')), step=0)
    for i, layer in enumerate(model.layers):
        tf.summary.text(f"layer_{i}_type", layer.__class__.__name__, step=0)
        tf.summary.text(f"layer_{i}_config", str(layer.get_config()), step=0)

    # Start network monitoring if the labels party is running on a different machine.
    if FLAGS.party == "f":
        label_party_ip = eval(FLAGS.cluster_spec)[labels_party_job][0].split(":")[0]
        network_mon_utils.setup_traffic_monitoring(ip=label_party_ip)

    # Train the model.
    history = model.fit(
        features_dataset,
        labels_dataset,
        epochs=FLAGS.epochs,
        validation_data=val_dataset,
        callbacks=[tb],
    )

    # Stop network monitoring
    if FLAGS.party == "f":
        bytes_recv, bytes_sent = network_mon_utils.get_byte_count()
        print(f"Network bytes recv, sent: {bytes_recv}, {bytes_sent}")
        tf.summary.scalar("bytes_recv", bytes_recv, step=0)
        tf.summary.scalar("bytes_sent", bytes_sent, step=0)


    print("Training complete.")
    batch_size = 2**12
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
    tf.summary.scalar("dp_epsilon", eps, step=0)


if __name__ == "__main__":
    app.run(main)
