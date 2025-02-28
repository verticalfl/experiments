from absl import app
from absl import flags
import dp_accounting
import math

flags.DEFINE_integer("target_epochs", 10, "Number of epochs")
flags.DEFINE_integer("samples", 60000, "Number of available training samples.")
flags.DEFINE_integer("log2_batch_size", 12, "Number of samples per batch")
flags.DEFINE_float("target_delta", 1e-5, "Target delta")
FLAGS = flags.FLAGS


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


def search_noise_multiplier(target_epsilon, target_delta, epochs, training_num_samples, batch_size):
    """
    Binary search for noise multiplier that achieves target epsilon.
    """
    noise_l = 1.0
    noise_r = 100.0
    noise = (noise_l + noise_r) / 2

    while True:
        epsilon = compute_epsilon(
            # Dataset batches use drop_remainder, so use // to compute steps.
            steps=epochs * (training_num_samples // batch_size),
            batch_size=batch_size,
            training_num_samples=training_num_samples,
            noise_multiplier=noise,
            target_delta=target_delta,
        )

        # Stop if noise is found.
        if math.isclose(epsilon, target_epsilon, abs_tol=1e-3):
            return noise

        if epsilon > target_epsilon:  # Too little privacy, add more noise
            noise_l = noise
            noise = (noise_l + noise_r) / 2
        else:  # Too much privacy, reduce noise
            noise_r = noise
            noise = (noise_l + noise_r) / 2

        # Stop if noise not found.
        if math.isclose(noise_l, noise_r, abs_tol=1e-3):
            return None


def main(_):
    search_epsilons = list(range(2, 11, 1))
    search_epsilons = [s / 10 for s in search_epsilons]
    batch_size = 2**FLAGS.log2_batch_size

    noise_multipliers_found = []

    for e in search_epsilons:
        nm = search_noise_multiplier(
            target_epsilon=e,
            target_delta=FLAGS.target_delta,
            epochs=FLAGS.epochs,
            training_num_samples=FLAGS.samples,
            batch_size=batch_size,
        )
        noise_multipliers_found.append(nm)

    print(f"Epochs: {FLAGS.epochs}")
    print(f"Samples: {FLAGS.samples}")
    print(f"Batch size: 2**{FLAGS.log2_batch_size}")
    for n, e in zip(noise_multipliers_found, search_epsilons):
        print(f"epsilon: {e}\tnoise_multiplier: {n}")


if __name__ == "__main__":
    app.run(main)
