import string
import os
import signal
import sys
from datetime import datetime
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
import tf_shell
import tf_shell_ml
import nltk
from experiment_utils import features_party_job, labels_party_job, ExperimentTensorBoard

nltk.download("stopwords")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
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
flags.DEFINE_integer("backprop_cleartext_sz", 33, "Cleartext size for backpropagation")
flags.DEFINE_integer(
    "backprop_scaling_factor", 16, "Scaling factor for backpropagation"
)
flags.DEFINE_integer("backprop_noise_offset", 14, "Noise offset for backpropagation")
flags.DEFINE_integer("noise_cleartext_sz", 36, "Cleartext size for noise")
flags.DEFINE_integer("noise_noise_offset", 0, "Noise offset for noise")
flags.DEFINE_bool("eager_mode", False, "Eager mode")
flags.DEFINE_bool("plaintext", False, "Run without encryption, noise, or masking.")
FLAGS = flags.FLAGS


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

        # The labels party just runs the server while the training is driven by the
        # features party.
        if this_job == labels_party_job:
            print(f"{this_job} server started.", flush=True)
            server.join()  # Wait for the features party to finish.
            exit(0)

    # Set up training data. Split the training set into 60% and 40% to end up
    # with 15,000 examples for training, 10,000 examples for validation and
    # 25,000 examples for testing.
    train_data, val_data = tfds.load(
        name="imdb_reviews",
        split=("train", "test"),
        as_supervised=True,
    )

    num_examples = int(train_data.cardinality().numpy())
    print("Number of training examples:", num_examples)

    tf.config.run_functions_eagerly(FLAGS.eager_mode)

    with tf.device(labels_party_dev):
        # One-hot encode the labels for training data on the features party.
        labels_dataset = train_data.map(
            lambda x, y: tf.one_hot(tf.cast(y, tf.int32), 2)
        ).batch(2**12)

    with tf.device(features_party_dev):
        features_dataset = train_data.map(lambda x, y: x).batch(2**12)
        val_data = val_data.shuffle(buffer_size=1024).batch(32)

        stop_words = nltk.corpus.stopwords.words("english")

        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            no_punctuation = tf.strings.regex_replace(
                lowercase, f"[{string.punctuation}]", ""
            )
            words = tf.strings.split(no_punctuation)
            stopwords_set = tf.constant(stop_words)
            no_stopwords = tf.ragged.boolean_mask(
                words,
                ~tf.reduce_any(
                    tf.equal(words[:, :, tf.newaxis], stopwords_set), axis=-1
                ),
            )
            return tf.strings.reduce_join(no_stopwords, separator=" ", axis=-1)

        # Create the text vectorization layer.
        vocab_size = 10000  # This dataset has 92061 unique words.
        max_length = 100
        embedding_dim = 16
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            standardize=custom_standardization,
            output_sequence_length=max_length,
        )
        vectorize_layer.adapt(features_dataset)

        # Add the text vectorization layer as a preprocessing step in the datasets.
        def preprocess_features(text):
            return vectorize_layer(text)

        # Additionally, one-hot encode the labels for validation data.
        def preprocess_all(text, label):
            return (vectorize_layer(text), tf.one_hot(tf.cast(label, tf.int32), 2))

        features_dataset = features_dataset.map(preprocess_features)
        val_data = val_data.map(preprocess_all)

        cache_path = "./cache-imdb-post-scale/"

        # Define functions which generate encryption context for the
        # backpropagation and noise parts of the protocol. When not executing
        # eagerly, autocontext can be used to automatically determine the
        # encryption parameters. When executing eagerly, parameters must be
        # specified manually (or simply copied from a previous run which uses
        # autocontext).
        def backprop_context_fn(read_cache):
            if FLAGS.eager_mode:
                return tf_shell.create_context64(
                    log_n=12,
                    main_moduli=[1688880462102529, 2181470596882433],
                    plaintext_modulus=8590090241,
                    scaling_factor=flags.FLAGS.backprop_scaling_factor,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=flags.FLAGS.backprop_cleartext_sz,
                    scaling_factor=flags.FLAGS.backprop_scaling_factor,
                    noise_offset_log2=flags.FLAGS.backprop_noise_offset,
                    read_from_cache=read_cache,
                    cache_path=cache_path,
                )

        def noise_context_fn(read_cache):
            if FLAGS.eager_mode:
                return tf_shell.create_context64(
                    log_n=12,
                    main_moduli=[6192450225922049, 16325550595612673],
                    plaintext_modulus=68719484929,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=flags.FLAGS.noise_cleartext_sz,
                    scaling_factor=1,
                    noise_offset_log2=flags.FLAGS.noise_noise_offset,
                    read_from_cache=read_cache,
                    cache_path=cache_path,
                )

        # Create the model.
        # vectorization layer input: [batch_size, string (variable length)]
        # embedding layer input: [batch_size, int token number (max_length)]
        # pooling layer input: [batch_size, max_length, embedding (embedding_dim)]
        model = tf_shell_ml.PostScaleSequential(
            layers=[
                tf.keras.layers.Embedding(
                    vocab_size + 1,  # +1 for OOV token.
                    embedding_dim,
                ),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=backprop_context_fn,
            noise_context_fn=noise_context_fn,
            labels_party_dev=labels_party_dev,
            features_party_dev=features_party_dev,
            noise_multiplier=FLAGS.noise_multiplier,
            cache_path=cache_path,
            jacobian_devices=jacobian_dev,
            disable_encryption=FLAGS.plaintext,
            disable_masking=FLAGS.plaintext,
            disable_noise=FLAGS.plaintext,
            # check_overflow_INSECURE=True,
        )

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/imdb-post-scale-{stamp}"

    tb = ExperimentTensorBoard(
        log_dir=logdir,
        # ExperimentTensorBoard kwargs.
        noise_multiplier=FLAGS.noise_multiplier,
        learning_rate=FLAGS.learning_rate,
        party=FLAGS.party,
        gpu_enabled=FLAGS.gpu,
        num_gpus=len(tf.config.list_physical_devices("GPU")),
        layers=model.layers,
        cluster_spec=FLAGS.cluster_spec,
        target_delta=1e-5,
        training_num_samples=num_examples,
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
        validation_data=val_data,
        callbacks=[tb],
    )


if __name__ == "__main__":
    app.run(main)
