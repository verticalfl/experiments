# https://builtin.com/machine-learning/vgg16
# dataset from https://www.kaggle.com/datasets/salader/dogs-vs-cats/data
# unzip the dataset to the current directory
# mkdir -p cats-and-dogs && unzip -q ../cats-and-dogs.zip -d cats-and-dogs
name="dog-cat-tf"
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

# Dataset from
# Jeremy Elson, John R. Douceur, Jon Howell, Jared Saul, Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization, in Proceedings of 14th ACM Conference on Computer and Communications Security (CCS), Association for Computing Machinery, Inc., Oct. 2007

# Hyperparam suggestions:
# https://github.com/chasingbob/squeezenet-keras gets 80% accuracy
# https://github.com/nlml/cats-vs-dogs-classifier-pruned-and-served gets 98% accuracy with l1 reg and pruning, looks like 303 epochs total.
# https://florianbordes.wordpress.com/2016/04/04/cats-vs-dogs-9-squeezenet/ gets 87% after 100k epochs!
# https://obilaniu6266h16.wordpress.com/2016/03/28/first-success-15-42-validation-error-the-benefits-of-cps/
# https://www.researchgate.net/profile/Thomas-Unterthiner/publication/309935608_Speeding_up_Semantic_Segmentation_for_Autonomous_Driving/links/58524adf08ae7d33e01a58a7/Speeding-up-Semantic-Segmentation-for-Autonomous-Driving.pdf
#   Note this paper says training took 22 hours on 2 GPUs, with some weights pre-initialized!

flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for training")
flags.DEFINE_float("beta_1", 0.9, "Beta 1 for Adam optimizer")
flags.DEFINE_float("decay_rate", 1.0, "Learning rate decay.")
flags.DEFINE_integer("epochs", 10, "Number of epochs")
flags.DEFINE_bool("eager_mode", False, "Eager mode")
flags.DEFINE_bool("tune", False, "Tune hyperparameters (or use default values).")
FLAGS = flags.FLAGS
tf.random.set_seed(1337)

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
    def __init__(self, cache_path):
        super().__init__(cache_path)
        self.cache_path = cache_path

    def hp_hash(self, hp_dict):
        """Returns a stable short hash for a dictionary of hyperparameter values."""
        # Convert dict to canonical JSON string and hash it
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
        # Create the model.
        input_shape = (224, 224, 3)
        weight_decay = hp.Choice("weight_decay", values=[0.0, 1e-5, 1e-4, 5e-4, 1e-3], default=0.)
        inputs, outputs = SqueezeNetv1_1Small(2, weight_decay, inputs=input_shape)

        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
        )

        model.build((None,) + input_shape)
        model.summary()

        # Learning rate warm up is good practice for large batch sizes.
        # see https://arxiv.org/pdf/1706.02677
        # lr = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0005, 0.0001, 0.00005, 0.00001], default=FLAGS.learning_rate)
        # # decay_rate = hp.Choice("decay_rate", values=[0.9, 0.95, 0.98, 1.0], default=FLAGS.decay_rate)
        # lr_schedule = LRWarmUp(
        #     initial_learning_rate=lr,
        #     decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
        #         lr,
        #         # decay_steps=160 * 2, # every 2 epochs
        #         # decay_rate=decay_rate,
        #         decay_steps=1,
        #         decay_rate=1.0,
        #     ),
        #     warmup_steps=16*5, # paper above uses 5 epcohs
        #     warmup_steps=160,
        # )

        beta_1 = hp.Choice("beta_1", values=[0.7, 0.8, 0.9, 0.95, 0.99], default=FLAGS.beta_1)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            # optimizer=tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1),
            optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate, beta_1=beta_1),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        return model

def main(_):
    # Set up training data.
    data_dir = 'cats-and-dogs'
    bs = 2**9  # Note this is smaller than PostScale protocol uses (2**12)
    val_bs = 2**5

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=(224, 224),
        batch_size=bs,
        label_mode='categorical',
        shuffle=False,
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/test",
        validation_split=0.18,
        subset='validation',
        image_size=(224, 224),
        batch_size=val_bs,
        label_mode='categorical',
        shuffle=False,
    )

    # Rescale to [0, 1] and ensure float32 types for compatibility
    def _rescale(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        labels = tf.cast(labels, tf.float32)
        return images, labels

    train_dataset = train_dataset.map(_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(_rescale, num_parallel_calls=tf.data.AUTOTUNE)

    # Derive validation example count from dataset cardinality
    _val_batches = tf.data.experimental.cardinality(val_dataset)
    if _val_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
        # Fallback: iterate once to count
        _val_batches = sum(1 for _ in val_dataset)
    else:
        _val_batches = int(_val_batches.numpy())
    num_val_examples = _val_batches * val_bs

    # Features-only dataset to send to the model on the features party
    features_dataset = train_dataset.map(lambda f, l: f, num_parallel_calls=tf.data.AUTOTUNE)

    labels_dataset = train_dataset.map(lambda f, l: l, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    # features_dataset = features_dataset.prefetch(tf.data.AUTOTUNE)
    # labels_dataset = labels_dataset.prefetch(tf.data.AUTOTUNE)

    # Defining data generator with Data Augmentation
    data_gen_augmented = ImageDataGenerator(rescale = 1/255., 
                                            validation_split = 0.18,
                                            # zoom_range = 0.2,
                                            # horizontal_flip= True,
                                            # rotation_range = 20,
                                            # width_shift_range=0.2,
                                            # height_shift_range=0.2
                                            )
    print('Augmented training Images:')
    train_iterator = data_gen_augmented.flow_from_directory(data_dir, 
                                                      target_size = (224, 224), 
                                                      batch_size = bs,
                                                      subset = 'training',
                                                      class_mode = 'categorical',
                                                      shuffle = True,
                                                      )

    num_examples = train_iterator.samples
    print("Number of training examples:", num_examples)

    # Testing Augmented Data
    # Defining Validation_generator withour Data Augmentation
    data_gen = ImageDataGenerator(rescale = 1/255., validation_split = 0.18)

    print('Unchanged Validation Images:')
    val_iterator = data_gen.flow_from_directory(data_dir, 
                                            target_size = (224, 224), 
                                            batch_size = val_bs,
                                            subset = 'validation',
                                            class_mode = 'categorical')

    # Define output signature for the generator
    # The DirectoryIterator yields (batch_of_images, batch_of_labels)
    # Images: (batch_size, height, width, channels), dtype=float32
    # Labels: (batch_size, 2), dtype=float32
    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )

    # Create tf.data.Dataset from the training iterator
    # The lambda ensures the iterator is obtained fresh if from_generator is called multiple times
    # or if the dataset is re-iterated in some contexts. Keras iterators are usually epoch-aware.
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_iterator,
        output_signature=output_signature
    ).unbatch().batch(bs, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prepare validation dataset
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_iterator,
        output_signature=output_signature
    ).unbatch().batch(val_bs, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    tf.config.run_functions_eagerly(FLAGS.eager_mode)

    hypermodel = HyperModel(cache_path="cache-"+name)

    # Set up tensorboard logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.abspath("") + f"/tflogs/dog-cat-tf-{stamp}"
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        write_steps_per_second=True,
        update_freq="batch",
        #profile_batch=2,
    )

    if FLAGS.tune:
        # Tune the hyperparameters.
        tuner = kt.RandomSearch(
            hypermodel,
            max_trials=60,
            objective=[
                kt.Objective('val_categorical_accuracy', direction='max'),
            ],
            directory="kerastuner",
            project_name=name,
            max_consecutive_failed_trials=30,
        )
        tuner.search_space_summary()

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_categorical_accuracy", patience=5, min_delta=0.001, mode='max')

        tuner.search(
            train_dataset,
            # steps_per_epoch=num_examples // bs,
            steps_per_epoch=10,  # shorten to just 10
            epochs=FLAGS.epochs,
            validation_data=val_dataset,
            validation_steps=num_val_examples // val_bs,
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
            train_dataset,
            # steps_per_epoch=num_examples // bs,
            steps_per_epoch=10,  # shorten to just 10
            epochs=FLAGS.epochs,
            validation_data=val_dataset,
            validation_steps=num_val_examples // val_bs,
            callbacks=[tb],
            # callbacks=[tb, lr_plateau], # where lr_plateau = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, cooldown=0, verbose=1)
        )

    # # SIMPLE TRAIN
    # import time

    # epochs = 2

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
    #     initial_learning_rate=FLAGS.learning_rate,
    #     decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
    #         FLAGS.learning_rate,
    #         # decay_steps=16, # every 1 epoch
    #         decay_steps=32, # every 2 epochs
    #         # decay_steps=64, # every 4 epochs
    #         decay_rate=FLAGS.decay_rate,
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
