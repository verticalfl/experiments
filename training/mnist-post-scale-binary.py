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
flags.DEFINE_integer("backprop_cleartext_sz", 33, "Cleartext size for backpropagation")
flags.DEFINE_integer("backprop_scaling_factor", 32, "Scaling factor for backpropagation")
flags.DEFINE_integer("backprop_noise_offset", 14, "Noise offset for backpropagation")
flags.DEFINE_integer("noise_cleartext_sz", 35, "Cleartext size for noise")
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

    # Limit the number of features to reduce the memory footprint for testing.
    # x_train, x_test = x_train[:, :300], x_test[:, :300]

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
            .shuffle(2**14, seed=shuffle_seed)
            .map(lambda x, y: y)
        )
        labels_dataset = labels_dataset.batch(2**12)

    with tf.device(features_party_dev):
        features_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(2**14, seed=shuffle_seed)
            .map(lambda x, y: x)
        )
        features_dataset = features_dataset.batch(2**12)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        cache_path = "./cache-mnist-postscale-binary/"

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

        def noise_context_fn (read_cache):
            if FLAGS.eager_mode:
                return tf_shell.create_context64(
                    log_n=12,
                    main_moduli=[2251800887492609, 9007203549970433],
                    plaintext_modulus=34359754753,
                )
            else:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=flags.FLAGS.noise_cleartext_sz,
                    scaling_factor=1,
                    noise_offset_log2=flags.FLAGS.noise_noise_offset,
                    read_from_cache=read_cache,
                    cache_path=cache_path,
                )

        # Create the model. When using post scale, use standard Keras layers.
        model = tf_shell_ml.PostScaleSequential(
            layers=[
                keras.layers.Dense(100, activation=tf.nn.relu, use_bias=False),
                keras.layers.Dense(2, activation=tf.nn.softmax, use_bias=False),
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

        model.build(input_shape=(None, 784))
        model.summary()

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/post-scale-binary-{stamp}"
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
        target_delta=1e-4,
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
        validation_data=val_dataset,
        callbacks=[tb],
    )


if __name__ == "__main__":
    app.run(main)
