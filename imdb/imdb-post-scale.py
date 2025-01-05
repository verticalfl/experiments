import time
from datetime import datetime
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
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

flags.DEFINE_float("learning_rate", 0.05, "Learning rate for training")
flags.DEFINE_float("noise_multiplier", 1.00, "Noise multiplier for DP-SGD")
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
        labels_party_dev = "/job:localhost/replica:0/task:0/device:CPU:0"
        features_party_dev = "/job:localhost/replica:0/task:0/device:CPU:0"
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

    cache_path = "./cache-imdb-dpsgd/"

    # Set up training data. Split the training set into 60% and 40% to end up
    # with 15,000 examples for training, 10,000 examples for validation and
    # 25,000 examples for testing.
    train_data, val_data, test_data = tfds.load(
        name="imdb_reviews",
        split=("train[:60%]", "train[60%:]", "test"),
        as_supervised=True,
    )
    with tf.device(labels_party_dev):
        # One-hot encode the labels for training data on the features party.
        labels_dataset = train_data.map(lambda x,y: tf.one_hot(tf.cast(y, tf.int32), 2)).batch(2**12)

    with tf.device(features_party_dev):
        features_dataset = train_data.map(lambda x,y: x).batch(2**12)
        val_data = val_data.shuffle(buffer_size=1024).batch(32)
        test_data = test_data.shuffle(buffer_size=1024).batch(32)

        # Create the text vectorization layer.
        vocab_size = 10000  # This dataset has 92061 unique words.
        max_length = 250
        embedding_dim = 16
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
        )
        vectorize_layer.adapt(features_dataset)

        # Add the text vectorization layer as a preprocessing step in the datasets.
        def preprocess_features(text):
            return (vectorize_layer(text))

        # Additionally, one-hot encode the labels for validation data.
        def preprocess_all(text, label):
            return (vectorize_layer(text), tf.one_hot(tf.cast(label, tf.int32), 2))

        features_dataset = features_dataset.map(preprocess_features)
        val_data = val_data.map(preprocess_all)
        test_data = test_data.map(preprocess_all)

        # Create the model.
        model = tf_shell_ml.PostScaleSequential(
            layers=[
                # Note we use Shell's embedding layer here because it allows us
                # to skip the most popular words, to match behavior of the DP-SGD
                # model.
                tf_shell_ml.ShellEmbedding(
                    vocab_size + 1,  # +1 for OOV token.
                    embedding_dim,
                    skip_embeddings_below_index=200,  # Skip the most common words.
                ),
                #tf_shell_ml.ShellDropout(0.5),
                tf_shell_ml.GlobalAveragePooling1D(),
                tf_shell_ml.ShellDropout(0.5),
                tf_shell_ml.ShellDense(
                    2,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=lambda read_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=33,
                scaling_factor=16,
                noise_offset_log2=14,
                read_from_cache=read_cache,
                cache_path=cache_path,
            ),
            noise_context_fn=lambda read_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=36,
                scaling_factor=1,
                noise_offset_log2=0,
                read_from_cache=read_cache,
                cache_path=cache_path,
            ),
            labels_party_dev=labels_party_dev,
            features_party_dev=features_party_dev,
            noise_multiplier=FLAGS.noise_multiplier,
            cache_path=cache_path,
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
    logdir = os.path.abspath("") + f"/tflogs/imdb-dpsgd-{stamp}"
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
        tf.summary.scalar(f"layer_{i}_units", layer.units, step=0)
        tf.summary.text(f"layer_{i}_activation", layer.activation.__name__ if layer.activation is not None else "None", step=0)

    # Train the model.
    history = model.fit(
        features_dataset,
        labels_dataset,
        epochs=FLAGS.epochs,
        validation_data=val_data,
        callbacks=[tb],
    )

    print("Training complete.")
    batch_size = 2**12
    samples_per_epoch = 15000 - (15000 % batch_size)

    # Compute the privacy budget expended.
    eps = compute_epsilon(
        steps=FLAGS.epochs * 15000 // batch_size,
        batch_size=batch_size,
        num_samples=samples_per_epoch,
        noise_multiplier=FLAGS.noise_multiplier,
        target_delta=1e-4,
    )
    print(f"Privacy budget expended: {eps}")
    tf.summary.scalar("dp_epsilon", eps, step=0)


if __name__ == "__main__":
    app.run(main)
