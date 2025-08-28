import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulation Parameters ---
# Let's assume our gradients are 100-dimensional vectors
GRADIENT_DIM = 100
# We'll simulate a batch of gradients
NUM_GRADIENTS = 2**10
# When simulating the jacobian, there are this many output classes
NUM_CLASSES = 10
# Assume each component of the gradient vector is drawn from N(0, 0.5^2)
# This creates a distribution of gradient norms we can work with.
GRADIENT_STD = 0.5

# --- 2. Generate Synthetic Gradients ---
# Create a synthetic dataset of gradients
# Shape: (NUM_GRADIENTS, GRADIENT_DIM)
np.random.seed(42)
true_gradients = np.random.normal(0, GRADIENT_STD, (NUM_GRADIENTS, GRADIENT_DIM))
true_jacobian = np.random.normal(0, GRADIENT_STD, (NUM_GRADIENTS, NUM_CLASSES, GRADIENT_DIM))
# Calculate the L2 norm of each gradient vector
gradient_norms = np.linalg.norm(true_gradients, axis=1)


# --- 3. Define Clipping and Analysis Functions ---

def standard_clip(gradients, C):
    """
    Applies standard per-example gradient clipping.
    The sensitivity of the sum is C.
    """
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    scale_factors = np.minimum(1.0, C / norms)
    clipped_gradients = gradients * scale_factors
    return clipped_gradients, scale_factors

def novel_clip(jacobians, C):
    """
    Applies standard per-example gradient clipping.
    The sensitivity of the sum is C.
    """
    # choose a random class as the "true" gradient
    real_classes = np.random.randint(0, NUM_CLASSES, (NUM_GRADIENTS))
    gradients = np.array(jacobians)[np.arange(NUM_GRADIENTS), real_classes]

    class_norms = np.linalg.norm(jacobians, axis=2, keepdims=True)
    max_class_norms = np.max(class_norms, axis=1, keepdims=True)
    scale_factors = np.minimum(1.0, C / max_class_norms)
    clipped_gradients = gradients * scale_factors
    return gradients, clipped_gradients, scale_factors

def calculate_bias(original_gradients, processed_gradients):
    """
    Calculates the bias as the average L2 norm of the distortion vector.
    A larger value means more bias.
    """
    distortion = original_gradients - processed_gradients
    avg_distortion_norm = np.mean(np.linalg.norm(distortion, axis=1))
    return avg_distortion_norm

def calculate_variance_proxy(sensitivity):
    """
    The noise variance is proportional to sensitivity^2.
    We'll use sensitivity^2 as a direct proxy for variance.
    """
    return sensitivity**2

# --- 4. Run the Simulation ---

# Part A: Analyze the standard DP-SGD trade-off
C_low = 0.5
C_high = 6
standard_thresholds = np.linspace(C_low, C_high, 25)
standard_bias = []
standard_variance = []

for C in standard_thresholds:
    # Clip gradients
    clipped_g, _ = standard_clip(true_gradients, C)
    # Calculate bias
    bias = calculate_bias(true_gradients, clipped_g)
    # Variance is determined by the clipping threshold C
    variance = calculate_variance_proxy(C)

    standard_bias.append(bias)
    standard_variance.append(variance)

# Part B: Analyze the novel method, but without per-gradient noise calibration
novel_bias = []
novel_variance = []

for C in standard_thresholds:
    # Clip gradients
    gradients, clipped_g, _ = novel_clip(true_jacobian, C)
    # Calculate bias
    bias = calculate_bias(gradients, clipped_g)
    # Variance is determined by the clipping threshold C
    variance = calculate_variance_proxy(C)

    novel_bias.append(bias)
    novel_variance.append(variance)

# Part C: Analyze the novel method *with* per-gradient noise calibration
novel_star_bias = []
novel_star_variance = []

for C in standard_thresholds:
    # Clip gradients
    gradients, clipped_g, norms = novel_clip(true_jacobian, C)
    # Calculate bias
    bias = calculate_bias(gradients, clipped_g)
    # Variance is determined by the clipping threshold C
    variance = calculate_variance_proxy(np.mean(norms))

    novel_star_bias.append(bias)
    novel_star_variance.append(variance)


# --- 5. Plot the Results ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4), dpi=400)

# Plot the curve for the standard clipping method
plt.plot(standard_bias, standard_variance, marker='', linestyle='--', label='Clip by Norm (DP-SGD)', markersize=4, color='black')

# Plot the novel clipping method
plt.plot(novel_bias, novel_variance, marker='', linestyle='--', label='Clip by Maximum Per-class Norm\n(Intermediate Method)', markersize=4, color='gray')

# Plot the novel clipping method
plt.plot(novel_star_bias, novel_star_variance, marker='', linestyle='--', label='Clip by Maximum Per-class Norm\nwith Dynamic Noise Calibration (This Work)', markersize=4, color='lightgray')

# Add annotations and labels
plt.title('Clipping Bias-Variance Trade-off', fontsize=16)
plt.xlabel('Bias', fontsize=12)
plt.ylabel('Variance (Proxy: Sensitivity²)', fontsize=12)
#plt.style.use('grayscale')
#plt.legend(fontsize=11)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Annotate the direction of the trade-off for the standard method
left_x, left_y = 1.0, 18
plt.annotate('Higher Clipping Threshold\n(Less Bias, More Noise)',
             xy=(left_x, left_y),
             xytext=(left_x, left_y),
             #xytext=(standard_bias[-5] + 0.1, standard_variance[-5] - 1.5),
             #xy=(standard_bias[-5], standard_variance[-5]),
             #arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.1'),
             fontsize=10)
right_x, right_y = 4, 3
plt.annotate('Lower Clipping Threshold\n(More Bias, Less Noise)',
             xy=(right_x, right_y),
             xytext=(right_x, right_y),
             #arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.1'),
             fontsize=10)

## Annotate the direction of the trade-off for the novel method
#plt.annotate('Lower C\n(More Bias, Less Noise)', xy=(novel_bias[-5], novel_variance[-5]),
#             xytext=(novel_bias[-5] + 0.1, novel_variance[-5] - 1.5),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
#             fontsize=10)
#plt.annotate('Higher C\n(Less Bias, More Noise)', xy=(novel_bias[5], novel_variance[5]),
#             xytext=(novel_bias[5] - 0.2, novel_variance[5] + 1.5),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
#             fontsize=10)
#
## Annotate the direction of the trade-off for the novel star method
#plt.annotate('Lower C\n(More Bias, Less Noise)', xy=(novel_star_bias[-5], novel_star_variance[-5]),
#             xytext=(novel_star_bias[-5] + 0.1, novel_star_variance[-5] - 1.5),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
#             fontsize=10)
#plt.annotate('Higher C\n(Less Bias, More Noise)', xy=(novel_star_bias[5], novel_star_variance[5]),
#             xytext=(novel_star_bias[5] - 0.2, novel_star_variance[5] + 1.5),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
#             fontsize=10)


plt.show()
plt.savefig('bv_tradeoff.png')

# Optional: Plot the distribution of gradient norms to understand the effect of clipping
plt.figure(figsize=(10, 5))
sns.histplot(gradient_norms, bins=50, kde=True)
plt.axvline(C_low, color='red', linestyle='--', label=f'Low Clipping Threshold')
plt.axvline(C_high, color='red', linestyle='--', label=f'High Clipping Threshold')
plt.title('Distribution of Gradient Norms')
plt.xlabel('L2 Norm')
plt.ylabel('Frequency')
plt.legend()
plt.show()
plt.savefig('grad_norms.png')

