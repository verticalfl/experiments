import subprocess
import atexit
import sys
import dp_accounting
import tensorflow as tf


def compute_epsilon(steps, batch_size, training_num_samples, noise_multiplier, target_delta):
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


class ExperimentTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(
        self,
        log_dir,
        noise_multiplier,
        learning_rate,
        party,
        gpu_enabled,
        num_gpus,
        layers,
        cluster_spec,
        target_delta,
        training_num_samples,
        epochs,
        backprop_cleartext_sz,
        backprop_scaling_factor,
        backprop_noise_offset,
        noise_cleartext_sz,
        noise_noise_offset,
        **kwargs,
    ):
        super().__init__(log_dir=log_dir, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.learning_rate = learning_rate
        self.party = party
        self.gpu_enabled = gpu_enabled
        self.num_gpus = num_gpus
        self.layers = layers
        self.cluster_spec = cluster_spec
        self.target_delta = target_delta
        self.training_num_samples = training_num_samples
        self.epochs = epochs
        self.backprop_cleartext_sz = backprop_cleartext_sz
        self.backprop_scaling_factor = backprop_scaling_factor
        self.backprop_noise_offset = backprop_noise_offset
        self.noise_cleartext_sz = noise_cleartext_sz
        self.noise_noise_offset = noise_noise_offset

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        # Write metadata to the validation logs, as this is where the accuracy
        # is computed (per batch accuracy not logged when encryption enabled).
        with self._val_writer.as_default():
            tf.summary.scalar("noise_multiplier", self.noise_multiplier, step=0)
            tf.summary.scalar("learning_rate", self.learning_rate, step=0)
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
            tf.summary.scalar("backprop_cleartext_sz", self.backprop_cleartext_sz, step=0)
            tf.summary.scalar("backprop_scaling_factor", self.backprop_scaling_factor, step=0)
            tf.summary.scalar("backprop_noise_offset", self.backprop_noise_offset, step=0)
            tf.summary.scalar("noise_cleartext_sz", self.noise_cleartext_sz, step=0)
            tf.summary.scalar("noise_noise_offset", self.noise_noise_offset, step=0)

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
                    steps=self.epochs * (self.training_num_samples // self.model.batch_size),  # always drops remainder
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
