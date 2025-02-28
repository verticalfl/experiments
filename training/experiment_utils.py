import subprocess
import atexit
import sys
import dp_accounting
import tensorflow as tf
from typing import Callable
import keras
from noise_multiplier_finder import compute_epsilon
from netio_mon import setup_traffic_monitoring, get_byte_count

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"


# Note: This class must override the Keras TensorBoard callback, not the
# TensorFlow TensorBoard callback as the latter does not correctly store
# hyperparameters when using keras tuner.
class ExperimentTensorBoard(keras.callbacks.TensorBoard):
    def __init__(
        self,
        log_dir,
        noise_multiplier,
        party,
        gpu_enabled,
        num_gpus,
        layers,
        cluster_spec,
        target_delta,
        training_num_samples,
        epochs,
        **kwargs,
    ):
        super().__init__(log_dir, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.party = party
        self.gpu_enabled = gpu_enabled
        self.num_gpus = num_gpus
        self.layers = layers
        self.cluster_spec = cluster_spec
        self.target_delta = target_delta
        self.training_num_samples = training_num_samples
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        # Write metadata to the validation logs, as this is where the accuracy
        # is computed (per batch accuracy not logged when encryption enabled).
        with self._val_writer.as_default():
            tf.summary.scalar("noise_multiplier", self.noise_multiplier, step=0)
            tf.summary.text("party", self.party, step=0)
            tf.summary.scalar("gpu_enabled", self.gpu_enabled, step=0)
            tf.summary.scalar("num_gpus", self.num_gpus, step=0)
            for i, layer in enumerate(self.layers):
                tf.summary.text(f"layer_{i}_type", layer.__class__.__name__, step=0)
                tf.summary.text(f"layer_{i}_config", str(layer.get_config()), step=0)
            tf.summary.text("cluster_spec", self.cluster_spec, step=0)
            tf.summary.scalar("target_delta", self.target_delta, step=0)
            tf.summary.scalar("training_num_samples", self.training_num_samples, step=0)
            tf.summary.scalar("planned_epochs", self.epochs, step=0)
            tf.summary.scalar("eager_mode", tf.config.functions_run_eagerly(), step=0)
            tf.summary.scalar(
                "check_overflow_INSECURE", self.model.check_overflow_INSECURE, step=0
            )
            tf.summary.scalar(
                "disable_encryption", self.model.disable_encryption, step=0
            )
            tf.summary.scalar("disable_masking", self.model.disable_masking, step=0)
            tf.summary.scalar("disable_noise", self.model.disable_noise, step=0)

        # Start network monitoring if the labels party is running on a different
        # machine.
        if self.party == "f":
            label_party_ip = eval(self.cluster_spec)[labels_party_job][0].split(":")[0]
            setup_traffic_monitoring(ip=label_party_ip)

    def on_train_end(self, logs=None):
        # Stop network monitoring
        with self._val_writer.as_default():
            tf.summary.scalar("batch_size", self.model.batch_size, step=0)

            if self.party == "f":
                bytes_recv, bytes_sent = get_byte_count()
                print(f"Network bytes recv, sent: {bytes_recv}, {bytes_sent}")
                tf.summary.scalar("bytes_recv", bytes_recv, step=0)
                tf.summary.scalar("bytes_sent", bytes_sent, step=0)

            # Compute the privacy budget expended.
            if self.noise_multiplier == 0.0:
                eps = float("inf")
            else:
                eps = compute_epsilon(
                    steps=self.epochs
                    * (
                        self.training_num_samples // self.model.batch_size
                    ),  # always drops remainder
                    batch_size=self.model.batch_size,
                    training_num_samples=self.training_num_samples,
                    noise_multiplier=self.model.noise_multiplier,
                    target_delta=self.target_delta,
                )
            print(f"Privacy budget expended: {eps}")
            tf.summary.scalar("dp_epsilon", eps, step=0)

        super().on_train_end(logs)


# Based on Transformers implementation.
class LRWarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
