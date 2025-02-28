import subprocess
import atexit
import sys
import dp_accounting
import tensorflow as tf
from typing import Callable
import keras

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"


def compute_epsilon(
    steps, batch_size, training_num_samples, noise_multiplier, target_delta
):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float("inf")
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = batch_size / training_num_samples
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )

    accountant.compose(event)

    return accountant.get_epsilon(target_delta=target_delta)


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


# Create unique chain names for each direction
outbound_chain = "TRAFFIC_MONITOR_OUT"
inbound_chain = "TRAFFIC_MONITOR_IN"


def setup_traffic_monitoring(port=None, ip=None):
    """
    Sets up iptables rules to monitor both inbound and outbound traffic.

    Args:
        port (int, optional): Specific port to monitor
        ip (str, optional): Specific IP to monitor
    """

    def cleanup():
        # Remove from OUTPUT and INPUT chains
        subprocess.run(
            ["sudo", "iptables", "-D", "OUTPUT", "-j", outbound_chain], check=False
        )
        subprocess.run(
            ["sudo", "iptables", "-D", "INPUT", "-j", inbound_chain], check=False
        )
        # Flush and delete our chains
        for chain in [outbound_chain, inbound_chain]:
            subprocess.run(["sudo", "iptables", "-F", chain], check=False)
            subprocess.run(["sudo", "iptables", "-X", chain], check=False)

    # If the chains already exist from a previous run, clean
    cleanup()

    try:
        # Create new chains
        for chain in [outbound_chain, inbound_chain]:
            subprocess.run(["sudo", "iptables", "-N", chain], check=True)

        # Create base rule for inbound chain
        inbound_rule = []
        if ip:
            inbound_rule.extend(["-s", ip])
        if port:
            inbound_rule.extend(["-p", "tcp", "--dport", str(port)])

        # Create base rule for outbound chain
        outbound_rule = []
        if ip:
            outbound_rule.extend(["-d", ip])
        if port:
            outbound_rule.extend(["-p", "tcp", "--dport", str(port)])

        # Add rules to each chain
        subprocess.run(
            ["sudo", "iptables", "-A", inbound_chain] + inbound_rule, check=True
        )
        subprocess.run(
            ["sudo", "iptables", "-A", outbound_chain] + outbound_rule, check=True
        )

        # Link to OUTPUT and INPUT chains
        subprocess.run(
            ["sudo", "iptables", "-I", "OUTPUT", "1", "-j", outbound_chain], check=True
        )
        subprocess.run(
            ["sudo", "iptables", "-I", "INPUT", "1", "-j", inbound_chain], check=True
        )

        # Register cleanup function
        atexit.register(cleanup)

    except subprocess.CalledProcessError as e:
        print(f"Error setting up iptables: {e}", file=sys.stderr)
        sys.exit(1)


def get_byte_count(direction="both"):
    """
    Returns the number of bytes counted by the monitoring chains.

    Args:
        direction (str): "in", "out", or "both"

    Returns:
        int or tuple: Byte count(s) depending on direction specified
    """
    try:

        def get_chain_bytes(chain):
            result = subprocess.run(
                ["sudo", "iptables", "-L", chain, "-v", "-n", "-x"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.split("\n")
            if len(lines) >= 3:
                fields = lines[2].split()
                if len(fields) >= 2:
                    return int(fields[1])
            return 0

        if direction == "out":
            return get_chain_bytes(outbound_chain)
        elif direction == "in":
            return get_chain_bytes(inbound_chain)
        else:  # both
            return (get_chain_bytes(inbound_chain), get_chain_bytes(outbound_chain))

    except subprocess.CalledProcessError as e:
        print(f"Error getting byte count: {e}", file=sys.stderr)
        return 0 if direction != "both" else (0, 0)

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
