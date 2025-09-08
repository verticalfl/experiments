# https://builtin.com/machine-learning/vgg16
# dataset from https://www.kaggle.com/datasets/salader/dogs-vs-cats/data
# unzip the dataset to the current directory
# mkdir -p cats-and-dogs && unzip -q ../cats-and-dogs.zip
name="dog-cat-post-scale-conv"
import time
from datetime import datetime
import tensorflow as tf
from absl import app
from absl import flags
import math
import keras
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tf_shell
import tf_shell_ml
import os
import json
import hashlib
import signal
import sys
import shutil
import keras_tuner as kt
from experiment_utils import (
    features_party_job,
    labels_party_job,
    TensorBoard,
    LRWarmUp,
    randomized_response_label_flip,
)
from noise_multiplier_finder import search_noise_multiplier

flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for training")
flags.DEFINE_float("beta_1", 0.9, "Beta 1 for Adam optimizer")
flags.DEFINE_float("epsilon", 1.0, "Differential privacy parameter")
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
flags.DEFINE_integer("backprop_scaling_factor", 16, "Scaling factor for backpropagation")
flags.DEFINE_integer("backprop_noise_offset", 14, "Noise offset for backpropagation")
flags.DEFINE_integer("noise_cleartext_sz", 36, "Cleartext size for noise")
flags.DEFINE_integer("noise_noise_offset", 0, "Noise offset for noise")
flags.DEFINE_bool("eager_mode", False, "Eager mode")
flags.DEFINE_bool("dp_sgd", False, "Run without encryption or masking (but with simple additive DP noise).")
flags.DEFINE_bool("rand_resp", False, "Run without encryption or masking, flipping the labels according to randomized response.")
flags.DEFINE_bool("check_overflow", False, "Check for overflow in the protocol.")
flags.DEFINE_bool("tune", False, "Tune hyperparameters (or use default values).")
FLAGS = flags.FLAGS

# from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dropout, BatchNormalization
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import regularizers

def fire_module(input_fire, s1, e1, e3, weight_decay_l2, fireID):
    """
    A wrapper to build fire module with batch normalization

    # Arguments
        input_fire: input activations
        s1: number of filters for squeeze step
        e1: number of filters for 1x1 expansion step
        e3: number of filters for 3x3 expansion step
        weight_decay_l2: weight decay for conv layers
        fireID: ID for the module

    # Return
        Output activations
    """

    # Squezee step
    output_squeeze = Convolution2D(
        s1,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="same",
        name="fire" + str(fireID) + "_squeeze",
        data_format="channels_last",
    )(input_fire)
    # output_squeeze = BatchNormalization(name="fire" + str(fireID) + "_squeeze_bn")(output_squeeze)
    output_squeeze = Activation("relu", name="fire" + str(fireID) + "_squeeze_relu")(output_squeeze)
    
    # Expansion steps
    output_expand1 = Convolution2D(
        e1,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="same",
        name="fire" + str(fireID) + "_expand1",
        data_format="channels_last",
    )(output_squeeze)
    # output_expand1 = BatchNormalization(name="fire" + str(fireID) + "_expand1_bn")(output_expand1)
    output_expand1 = Activation("relu", name="fire" + str(fireID) + "_expand1_relu")(output_expand1)
    
    output_expand2 = Convolution2D(
        e3,
        (3, 3),
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="same",
        name="fire" + str(fireID) + "_expand2",
        data_format="channels_last",
    )(output_squeeze)
    # output_expand2 = BatchNormalization(name="fire" + str(fireID) + "_expand2_bn")(output_expand2)
    output_expand2 = Activation("relu", name="fire" + str(fireID) + "_expand2_relu")(output_expand2)
    
    # Merge expanded activations
    output_fire = Concatenate(axis=3)([output_expand1, output_expand2])
    return output_fire


def SqueezeNetSmall(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 3), residual=False):
    """
    A wrapper to build a small version of the SqueezeNet Model.
    Note the reduced number of filters in each fire module.

    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions
        residual: whether to use residual connections

    # Return
        The input and output layer of the model
    """
    input_img = Input(shape=inputs)

    conv1 = Convolution2D(
        32,
        (7, 7),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        strides=(2, 2),
        padding="same",
        name="conv1",
        data_format="channels_last",
    )(input_img)
    # conv1 = BatchNormalization(name="conv1_bn")(conv1)
    conv1 = Activation("relu", name="conv1_relu")(conv1)

    maxpool1 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool1", data_format="channels_last"
    )(conv1)

    fire2 = fire_module(maxpool1, 8, 16, 16, weight_decay_l2, 2)
    fire3 = fire_module(fire2, 8, 16, 16, weight_decay_l2, 3)
    if residual:
        fire3 = Concatenate(axis=-1, name="fire3_resid")([fire2, fire3])
    fire4 = fire_module(fire3, 16, 32, 32, weight_decay_l2, 4)

    maxpool4 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool4", data_format="channels_last"
    )(fire4)

    fire5 = fire_module(maxpool4, 16, 32, 32, weight_decay_l2, 5)
    if residual:
        fire5 = Concatenate(axis=-1, name="fire5_resid")([maxpool4, fire5])
    fire6 = fire_module(fire5, 32, 64, 64, weight_decay_l2, 6)
    fire7 = fire_module(fire6, 32, 64, 64, weight_decay_l2, 7)
    fire8 = fire_module(fire7, 64, 128, 128, weight_decay_l2, 8)

    maxpool8 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool8", data_format="channels_last"
    )(fire8)

    fire9 = fire_module(maxpool8, 64, 128, 128, weight_decay_l2, 9)
    if residual:
        fire9 = Concatenate(axis=-1, name="fire9_resid")([maxpool8, fire9])
    fire9_dropout = Dropout(0.5, name="fire9_dropout")(fire9)

    conv10 = Convolution2D(
        num_classes,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="valid",
        name="conv10",
        data_format="channels_last",
    )(fire9_dropout)
    # conv10 = BatchNormalization(name="conv10_bn")(conv10)
    conv10 = Activation("relu", name="conv10_relu")(conv10)

    global_avgpool10 = GlobalAveragePooling2D(data_format="channels_last")(conv10)
    softmax = Activation("softmax", name="softmax")(global_avgpool10)
    return input_img, softmax


def SqueezeNetMedium(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 3), residual=False):
    """
    A wrapper to build a medium version of the SqueezeNet Model.
    Note the reduced number of filters in each fire module.

    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions
        residual: whether to use residual connections

    # Return
        The input and output layer of the model
    """
    input_img = Input(shape=inputs)

    conv1 = Convolution2D(
        32,
        (7, 7),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        strides=(2, 2),
        padding="same",
        name="conv1",
        data_format="channels_last",
    )(input_img)
    # conv1 = BatchNormalization(name="conv1_bn")(conv1)
    conv1 = Activation("relu", name="conv1_relu")(conv1)

    maxpool1 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool1", data_format="channels_last"
    )(conv1)

    fire2 = fire_module(maxpool1, 8, 32, 32, weight_decay_l2, 2)
    fire3 = fire_module(fire2, 8, 32, 32, weight_decay_l2, 3)
    if residual:
        fire3 = Concatenate(axis=-1, name="fire3_resid")([fire2, fire3])
    fire4 = fire_module(fire3, 16, 64, 64, weight_decay_l2, 4)

    maxpool4 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool4", data_format="channels_last"
    )(fire4)

    fire5 = fire_module(maxpool4, 16, 64, 64, weight_decay_l2, 5)
    if residual:
        fire5 = Concatenate(axis=-1, name="fire5_resid")([maxpool4, fire5])
    fire6 = fire_module(fire5, 32, 128, 128, weight_decay_l2, 6)
    fire7 = fire_module(fire6, 32, 128, 128, weight_decay_l2, 7)
    fire8 = fire_module(fire7, 48, 192, 192, weight_decay_l2, 8)

    maxpool8 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool8", data_format="channels_last"
    )(fire8)

    fire9 = fire_module(maxpool8, 48, 192, 192, weight_decay_l2, 9)
    if residual:
        fire9 = Concatenate(axis=-1, name="fire9_resid")([maxpool8, fire9])
    fire9_dropout = Dropout(0.5, name="fire9_dropout")(fire9)

    conv10 = Convolution2D(
        num_classes,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="valid",
        name="conv10",
        data_format="channels_last",
    )(fire9_dropout)
    # conv10 = BatchNormalization(name="conv10_bn")(conv10)
    conv10 = Activation("relu", name="conv10_relu")(conv10)

    global_avgpool10 = GlobalAveragePooling2D(data_format="channels_last")(conv10)
    softmax = Activation("softmax", name="softmax")(global_avgpool10)
    return input_img, softmax


def SqueezeNet(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 3), residual=False):
    """
    A wrapper to build the original SqueezeNet Model as described in the paper.

    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions
        residual: whether to use residual connections

    # Return
        A SqueezeNet Keras Model
    """
    input_img = Input(shape=inputs)

    conv1 = Convolution2D(
        96,
        (7, 7),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        strides=(2, 2),
        padding="same",
        name="conv1",
        data_format="channels_last",
    )(input_img)
    # conv1 = BatchNormalization(name="conv1_bn")(conv1)
    conv1 = Activation("relu", name="conv1_relu")(conv1)

    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool1", data_format="channels_last"
    )(conv1)

    fire2 = fire_module(maxpool1, 16, 64, 64, weight_decay_l2, 2)
    fire3 = fire_module(fire2, 16, 64, 64, weight_decay_l2, 3)
    if residual:
        fire3 = Concatenate(axis=-1, name="fire3_resid")([fire2, fire3])
    fire4 = fire_module(fire3, 32, 128, 128, weight_decay_l2, 4)

    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool4", data_format="channels_last"
    )(fire4)

    fire5 = fire_module(maxpool4, 32, 128, 128, weight_decay_l2, 5)
    if residual:
        fire5 = Concatenate(axis=-1, name="fire5_resid")([maxpool4, fire5])
    fire6 = fire_module(fire5, 48, 192, 192, weight_decay_l2, 6)
    fire7 = fire_module(fire6, 48, 192, 192, weight_decay_l2, 7)
    fire8 = fire_module(fire7, 64, 256, 256, weight_decay_l2, 8)

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool8", data_format="channels_last"
    )(fire8)

    fire9 = fire_module(maxpool8, 64, 256, 256, weight_decay_l2, 9)
    if residual:
        fire9 = Concatenate(axis=-1, name="fire9_resid")([maxpool8, fire9])
    fire9_dropout = Dropout(0.5, name="fire9_dropout")(fire9)

    conv10 = Convolution2D(
        num_classes,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="valid",
        name="conv10",
        data_format="channels_last",
    )(fire9_dropout)
    # conv10 = BatchNormalization(name="conv10_bn")(conv10)
    conv10 = Activation("relu", name="conv10_relu")(conv10)

    global_avgpool10 = GlobalAveragePooling2D(data_format="channels_last")(conv10)
    softmax = Activation("softmax", name="softmax")(global_avgpool10)
    return input_img, softmax


def SqueezeNetv1_1Small(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 3), residual=False):
    """
    A wrapper to build SqueezeNet v1.1 Model

    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions

    # Return
        A SqueezeNet Keras Model
    """
    input_img = Input(shape=inputs)

    conv1 = Convolution2D(
        64,
        (3, 3),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        strides=(2, 2),
        padding="valid",
        name="conv1",
        data_format="channels_last",
    )(input_img)
    # conv1 = BatchNormalization(name="conv1_bn")(conv1)
    conv1 = Activation("relu", name="conv1_relu")(conv1)

    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool1", data_format="channels_last"
    )(conv1)

    fire2 = fire_module(maxpool1, 16, 48, 48, weight_decay_l2, 2)
    fire3 = fire_module(fire2, 16, 48, 48, weight_decay_l2, 3)
    if residual:
        fire3 = Concatenate(axis=-1, name="fire3_resid")([fire2, fire3])
    maxpool3 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool4", data_format="channels_last"
    )(fire3)

    fire4 = fire_module(maxpool3, 32, 96, 96, weight_decay_l2, 4)
    fire5 = fire_module(fire4, 32, 96, 96, weight_decay_l2, 5)
    if residual:
        fire5 = Concatenate(axis=-1, name="fire5_resid")([fire4, fire5])
    maxpool5 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool5", data_format="channels_last"
    )(fire5)

    fire6 = fire_module(maxpool5, 48, 128, 128, weight_decay_l2, 6)
    fire7 = fire_module(fire6, 48, 128, 128, weight_decay_l2, 7)
    fire8 = fire_module(fire7, 64, 160, 160, weight_decay_l2, 8)
    fire9 = fire_module(fire8, 64, 160, 160, weight_decay_l2, 9)
    if residual:
        fire9 = Concatenate(axis=-1, name="fire9_resid")([fire8, fire9])
    fire9_dropout = Dropout(0.5, name="fire9_dropout")(fire9)

    conv10 = Convolution2D(
        num_classes,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="valid",
        name="conv10",
        data_format="channels_last",
    )(fire9_dropout)
    # conv10 = BatchNormalization(name="conv10_bn")(conv10)
    conv10 = Activation("relu", name="conv10_relu")(conv10)

    global_avgpool10 = GlobalAveragePooling2D(data_format="channels_last")(conv10)
    softmax = Activation("softmax", name="softmax")(global_avgpool10)
    return input_img, softmax


def SqueezeNetv1_1(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 3), residual=False):
    """
    A wrapper to build SqueezeNet v1.1 Model

    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions

    # Return
        A SqueezeNet Keras Model
    """
    input_img = Input(shape=inputs)

    conv1 = Convolution2D(
        64,
        (3, 3),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        strides=(2, 2),
        padding="valid",
        name="conv1",
        data_format="channels_last",
    )(input_img)
    # conv1 = BatchNormalization(name="conv1_bn")(conv1)
    conv1 = Activation("relu", name="conv1_relu")(conv1)

    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool1", data_format="channels_last"
    )(conv1)

    fire2 = fire_module(maxpool1, 16, 64, 64, weight_decay_l2, 2)
    fire3 = fire_module(fire2, 16, 64, 64, weight_decay_l2, 3)
    if residual:
        fire3 = Concatenate(axis=-1, name="fire3_resid")([fire2, fire3])
    maxpool3 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool4", data_format="channels_last"
    )(fire3)

    fire4 = fire_module(maxpool3, 32, 128, 128, weight_decay_l2, 4)
    fire5 = fire_module(fire4, 32, 128, 128, weight_decay_l2, 5)
    if residual:
        fire5 = Concatenate(axis=-1, name="fire5_resid")([fire4, fire5])
    maxpool5 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name="maxpool5", data_format="channels_last"
    )(fire5)

    fire6 = fire_module(maxpool5, 48, 192, 192, weight_decay_l2, 6)
    fire7 = fire_module(fire6, 48, 192, 192, weight_decay_l2, 7)
    fire8 = fire_module(fire7, 64, 256, 256, weight_decay_l2, 8)
    fire9 = fire_module(fire8, 64, 256, 256, weight_decay_l2, 9)
    if residual:
        fire9 = Concatenate(axis=-1, name="fire9_resid")([fire8, fire9])
    fire9_dropout = Dropout(0.5, name="fire9_dropout")(fire9)

    conv10 = Convolution2D(
        num_classes,
        (1, 1),
        activation=None,
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="valid",
        name="conv10",
        data_format="channels_last",
    )(fire9_dropout)
    # conv10 = BatchNormalization(name="conv10_bn")(conv10)
    conv10 = Activation("relu", name="conv10_relu")(conv10)

    global_avgpool10 = GlobalAveragePooling2D(data_format="channels_last")(conv10)
    softmax = Activation("softmax", name="softmax")(global_avgpool10)
    return input_img, softmax



class HyperModel(kt.HyperModel):
    def __init__(self, labels_party_dev, features_party_dev, jacobian_devs, cache_path, num_examples):
        super().__init__()
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.jacobian_devs = jacobian_devs
        self.cache_path = cache_path
        self.num_examples = num_examples
        self.strategy = tf.distribute.MirroredStrategy(devices=self.jacobian_devs)

    def hp_hash(self, hp_dict):
        """Returns a stable short hash for a dictionary of hyperparameter values."""
        # Convert dict to canonical JSON string and hash it
        hp_dict["epsilon"] = FLAGS.epsilon
        hp_dict["eager_mode"] = FLAGS.eager_mode
        hp_dict["dp-sgd"] = FLAGS.dp_sgd
        hp_dict["rand_resp"] = FLAGS.rand_resp
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
        # Build the model on the feature holding party's CPU (vs GPU) to ensure
        # each GPU can process in parallel.
        with self.strategy.scope():
            # Define functions which generate encryption context for the
            # backpropagation and noise parts of the protocol. When not executing
            # eagerly, autocontext can be used to automatically determine the
            # encryption parameters. When executing eagerly, parameters must be
            # specified manually (or simply copied from a previous run which uses
            # autocontext).
            backprop_cleartext_sz=hp.Int("backprop_cleartext_sz", min_value=20, max_value=34, step=1, default=FLAGS.backprop_cleartext_sz)
            backprop_scaling_factor=hp.Choice("backprop_scaling_factor", values=[2, 4, 8, 16, 32], default=FLAGS.backprop_scaling_factor)
            backprop_noise_offset=hp.Choice("backprop_noise_offset", values=[0, 8, 14, 16, 32, 48], default=FLAGS.backprop_noise_offset)

            noise_cleartext_sz=hp.Int("noise_cleartext_sz", min_value=36, max_value=36, step=1, default=FLAGS.noise_cleartext_sz)
            noise_noise_offset=hp.Choice("noise_noise_offset", values=[0, 40], default=FLAGS.noise_noise_offset)
            # 0 and 40 correspond to ring degree of 2**12 and 2**13

            clip_threshold = hp.Float("clip_threshold", min_value=1.0, max_value=20.0, step=1.0, default=1.0)
            weight_decay = hp.Choice("weight_decay", values=[0.0, 1e-5, 1e-4, 5e-4, 1e-3], default=0.)

            def backprop_context_fn(read_cache):
                if FLAGS.eager_mode:
                    return tf_shell.create_context64(
                        log_n=12,
                        main_moduli=[1688880462102529, 2181470596882433],
                        plaintext_modulus=8590090241,
                        scaling_factor=FLAGS.backprop_scaling_factor,
                    )
                else:
                    return tf_shell.create_autocontext64(
                        log2_cleartext_sz=backprop_cleartext_sz,
                        scaling_factor=backprop_scaling_factor,
                        noise_offset_log2=backprop_noise_offset,
                        read_from_cache=read_cache,
                        cache_path=self.cache_path,
                    )

            def noise_context_fn (read_cache):
                if FLAGS.eager_mode:
                    return tf_shell.create_context64(
                        log_n=12,
                        main_moduli=[6192450225922049, 16325550595612673],
                        plaintext_modulus=68719484929,
                    )
                else:
                    return tf_shell.create_autocontext64(
                        log2_cleartext_sz=noise_cleartext_sz,
                        noise_offset_log2=noise_noise_offset,
                        read_from_cache=read_cache,
                        cache_path=self.cache_path,
                    )

            def noise_multiplier_fn(batch_size):
                # If doing randomized response, we don't need to compute the noise
                # multiplier. Return 0 to disable noise.
                if FLAGS.rand_resp:
                    return 0.0
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

            # Create the model.
            input_shape = (224, 224, 3)
            residual = hp.Choice("residual", values=[True, False], default=False)
            model_arch_str = hp.Choice("model_arch", values=["SqueezeNetSmall", "SqueezeNetMedium", "SqueezeNet", "SqueezeNetv1_1", "SqueezeNetv1_1Small"], default="SqueezeNetv1_1")

            def getModelClass(string):
                if string == "SqueezeNetSmall":
                    return SqueezeNetSmall # Trainable params: 256,818 (1003.20 KB)
                elif string == "SqueezeNetMedium":
                    return SqueezeNetMedium # Trainable params: 346,066 (1.32 MB)
                elif string == "SqueezeNet":
                    return SqueezeNet  # Trainable params: 736,450 (2.81 MB)
                elif string == "SqueezeNetv1_1":
                    return SqueezeNetv1_1  # Trainable params: 723,522 (2.76 MB)
                elif string == "SqueezeNetv1_1Small":
                    return SqueezeNetv1_1Small  # Trainable params: 479,106 (1.83 MB)

            model_arch = getModelClass(model_arch_str)
            inputs, outputs = model_arch(2, weight_decay, inputs=input_shape, residual=residual)

            model = tf_shell_ml.PostScaleModel(
                inputs=inputs,
                outputs=outputs,
                ubatch_per_batch=2**5,  # 8x24GB GPUs
                backprop_context_fn=backprop_context_fn,
                noise_context_fn=noise_context_fn,
                noise_multiplier_fn=noise_multiplier_fn,
                labels_party_dev=self.labels_party_dev,
                features_party_dev=self.features_party_dev,
                cache_path=self.cache_path,
                jacobian_devices=self.jacobian_devs,
                disable_he_backprop_INSECURE=FLAGS.dp_sgd or FLAGS.rand_resp,
                disable_masking_INSECURE=FLAGS.dp_sgd or FLAGS.rand_resp,
                simple_noise_INSECURE= FLAGS.dp_sgd or FLAGS.rand_resp,
                clip_threshold=clip_threshold,
                check_overflow_INSECURE=FLAGS.check_overflow or FLAGS.tune,
                jacobian_strategy=self.strategy,
            )

            model.build((None,) + input_shape)
            model.summary()

            # Learning rate warm up is good practice for large batch sizes.
            # see https://arxiv.org/pdf/1706.02677
            lr = hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6], default=FLAGS.learning_rate)
            # warmup_steps = hp.Choice("warmup_steps", values=[0, 2, 4], default=0)
            # decay_rate = hp.Choice("decay_rate", values=[0.95, 0.97, 0.99, 1.0], default=0.97)
            # lr_schedule = LRWarmUp(
            #     initial_learning_rate=lr,
            #     decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
            #         lr,
            #         decay_steps=1,
            #         decay_rate=decay_rate,
            #     ),
            #     warmup_steps=4,
            # )
            # clipnorm = hp.Choice("clipnorm", values=[0.5, 1.0, 2.0, 5.0], default=1.0)

            beta_1 = hp.Choice("beta_1", values=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95], default=FLAGS.beta_1)
        
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                #optimizer=tf.keras.optimizers.Adam(
                #    learning_rate=lr_schedule, 
                #    beta_1=beta_1, 
                #    # clipnorm=clipnorm
                #),
                optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate, beta_1=beta_1),
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

            # gpus = tf.config.list_physical_devices('GPU')
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #         tf.config.LogicalDeviceConfiguration(memory_limit=1024*1),
            #     ]
            # )
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(logical_gpus)
            # jacobian_dev = [f"/job:localhost/replica:0/task:0/device:GPU:{i}" for i in range(8)]


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
    data_dir = 'cats-and-dogs'
    bs = 2**12
    val_bs = 2**7

    # Next create the training and validation datasets on the feature holding
    # party. Note the validation dataset has both features and labels.
    with tf.device(features_party_dev):
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/train",
            image_size=(224, 224),
            batch_size=bs,
            label_mode='categorical',
            shuffle=False,
        )

        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/test",
            image_size=(224, 224),
            batch_size=val_bs,
            label_mode='categorical',
            shuffle=False,
        )

        # Print the class ratio in the training and validation datasets
        def _compute_class_stats(ds):
            class_names = getattr(ds, 'class_names', None)
            counts = None
            for _, labels in ds:
                batch_counts = tf.reduce_sum(labels, axis=0)
                counts = batch_counts if counts is None else counts + batch_counts
            if counts is None:
                return class_names, None, None, 0
            counts = tf.cast(tf.round(counts), tf.int64)
            total = int(tf.reduce_sum(counts).numpy())
            ratios = tf.cast(counts, tf.float32) / tf.cast(total, tf.float32)
            return class_names, counts.numpy().tolist(), ratios.numpy().tolist(), total

        def _print_class_stats(name, ds):
            class_names, counts, ratios, total = _compute_class_stats(ds)
            if counts is None:
                print(f"{name}: dataset is empty")
                return
            if class_names is None:
                class_names = [str(i) for i in range(len(counts))]
            print(f"{name} total={total}")
            for cname, c, r in zip(class_names, counts, ratios):
                print(f"  {cname}: {c} ({r:.3f})")

        _print_class_stats("Train", train_dataset)
        _print_class_stats("Val", val_dataset)

        # Rescale to [0, 1] and ensure float32 types for compatibility
        def _rescale(images, labels):
            images = tf.cast(images, tf.float32) / 255.0
            labels = tf.cast(labels, tf.float32)
            return images, labels

        train_dataset = train_dataset.map(_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(_rescale, num_parallel_calls=tf.data.AUTOTUNE)

        def _cardinality(dataset, batch_size):
            batches = tf.data.experimental.cardinality(dataset)
            if batches == tf.data.experimental.UNKNOWN_CARDINALITY:
                # Fallback: iterate once to count
                batches = sum(1 for _ in dataset)
            else:
                batches = int(batches.numpy())
            examples = batches * batch_size
            return examples
        num_examples = _cardinality(train_dataset, bs)
        num_val_examples = _cardinality(val_dataset, val_bs)
        print("Number of training examples:", num_examples)
        print("Number of validation examples:", num_val_examples)

        # Features-only dataset to send to the model on the features party
        features_dataset = train_dataset.map(lambda f, l: f, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        features_dataset = features_dataset.prefetch(tf.data.AUTOTUNE)

    # First create the training dataset which on the label holding party.
    # This party does not have the features.
    with tf.device(labels_party_dev):
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/train",
            image_size=(224, 224),
            batch_size=bs,
            label_mode='categorical',
            shuffle=False,
        )

        # Features-only dataset to send to the model on the features party
        labels_dataset = train_dataset.map(lambda f, l: l, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        labels_dataset = labels_dataset.prefetch(tf.data.AUTOTUNE)

    # Write the first few examples to a pdf file to view them for the features
    # and labels dataset.

    tf.config.run_functions_eagerly(FLAGS.eager_mode)

    target_delta = 10**int(math.floor(math.log10(1 / num_examples)))
    print("Target delta:", target_delta)

    hypermodel = HyperModel(
        labels_party_dev=labels_party_dev,
        features_party_dev=features_party_dev,
        jacobian_devs=jacobian_dev,
        cache_path="cache-"+name,
        num_examples=num_examples,
    )

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/dog-cat-post-scale-conv-{stamp}"
    tb = TensorBoard(
        log_dir=logdir,
        # ExperimentTensorBoard kwargs.
        party=FLAGS.party,
        gpu_enabled=FLAGS.gpu,
        num_gpus=len(tf.config.list_physical_devices('GPU')),
        cluster_spec=FLAGS.cluster_spec,
        target_delta=target_delta,
        training_num_samples=num_examples,
        epochs=FLAGS.epochs,
        # TensorBoard kwargs.
        write_steps_per_second=True,
        update_freq="batch",
        # profile_batch=2,
    )

    if FLAGS.tune:
        # Enhanced early stopping with gradient monitoring
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy', 
            patience=8, 
            min_delta=0.001, 
            mode='max',
            restore_best_weights=True
        )

        # Tune the hyperparameters.
        tuner = kt.RandomSearch(
            hypermodel,
            max_trials=80,
            objective=[
                kt.Objective('val_categorical_accuracy', direction='max'),
                # kt.Objective('time', direction='min')
            ],
            directory="kerastuner",
            project_name=name,
            max_consecutive_failed_trials=50,
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
                # kt.Objective('time', direction='min')
            ],
            directory="kerastuner",
            project_name="default_hps",
            max_consecutive_failed_trials=1,
            overwrite=True,  # Always overwrite previous runs.
        )
        trial = tuner.oracle.create_trial("single_run_trial")

        # Remove the cache path to ignore errors from previous runs.
        dirpath = os.path.abspath("") + "/cache-" + name
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)

        tuner.run_trial(
            trial,
            features_dataset,
            labels_dataset,
            epochs=FLAGS.epochs,
            validation_data=val_dataset,
            callbacks=[tb],
        )

    # # SIMPLE TRAIN
    # import time

    # epochs = 2
    # train_dataset = train_dataset.map(lambda x, y: (x, y)).batch(bs)

    # # Create the model.
    # input_shape = (224, 224, 3)
    # inputs, outputs = SqueezeNet(2, 0.0, inputs=input_shape)

    # model = keras.Model(
    #     inputs=inputs,
    #     outputs=outputs,
    # )
    # model.build((None,) + input_shape)
    # model.summary()
    # lr_schedule = LRWarmUp(
    #     initial_learning_rate=0.0001,
    #     decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
    #         0.0001,
    #         # decay_steps=16, # every 1 epoch
    #         decay_steps=32, # every 2 epochs
    #         # decay_steps=64, # every 4 epochs
    #         decay_rate=.95,
    #     ),
    #     # warmup_steps=16*5, # paper above uses 5 epcohs
    #     warmup_steps=4,
    # )
    # model.compile(
    #     loss=tf.keras.losses.CategoricalCrossentropy(),
    #     optimizer=tf.keras.optimizers.Adam(lr_schedule, beta_1=FLAGS.beta_1),
    #     # optimizer=tf.keras.optimizers.Adam(lr, beta_1=beta_1),
    #     # optimizer=tf.keras.optimizers.SGD(lr, momentum=beta_1),
    #     metrics=[tf.keras.metrics.CategoricalAccuracy()],
    # )
    # train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    # val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    # for epoch in range(epochs):
    #     print("\nStart of epoch %d" % (epoch,))
    #     start_time = time.time()

    #     # Iterate over the batches of the dataset.
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             pred = model(x_batch_train, training=True)
    #             loss_value = model.compute_loss(y=y_batch_train, y_pred=pred)
    #         grads = tape.gradient(loss_value, model.trainable_weights)
    #         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #         # Update training metric.
    #         train_acc_metric.update_state(y_batch_train, pred)

    #         # Log every 200 batches.
    #         if step % 10 == 0:
    #             print(
    #                 "Training loss (for one batch) at step %d: %.4f"
    #                 % (step, float(loss_value))
    #             )
    #             print("Seen so far: %d samples" % ((step + 1) * bs))
    #         if step > num_examples // bs:
    #             break

    #     # Display metrics at the end of each epoch.
    #     train_acc = train_acc_metric.result()
    #     print("Training acc over epoch: %.4f" % (float(train_acc),))

    #     # Reset training metrics at the end of each epoch
    #     train_acc_metric.reset_state()

    #     # Run a validation loop at the end of each epoch.
    #     for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
    #         val_logits = model(x_batch_val, training=False)
    #         # Update val metrics
    #         val_acc_metric.update_state(y_batch_val, val_logits)
    #         if step > 28:
    #             break
    #     val_acc = val_acc_metric.result()
    #     val_acc_metric.reset_state()
    #     print("Validation acc: %.4f" % (float(val_acc),))
    #     print("Time taken: %.2fs" % (time.time() - start_time))

if __name__ == "__main__":
    app.run(main)
