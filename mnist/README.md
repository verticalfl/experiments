# MNIST Model Tests
Here are some notes about tunable hyperparameters.

## Resource Requirements

mnist-dpsgd.py requires 286GB of RAM.
mnist-post-scale.py requires 380GB of RAM.
When using a GPU, 24GB of VRAM is good, 16GB is too small.

## Noise Multiplier
To choose the appropriate noise multiplier, see `noise_multiplier_finder.py`.
Example usage:
```bash
python noise_multiplier_finder.py --epochs 10 --samples 60000 --log2_batch_size 12

Epochs: 10
Samples: 60000
Batch size: 2**12
epsilon: 0.2    noise_multiplier: None
epsilon: 0.3    noise_multiplier: 10.7890625
epsilon: 0.4    noise_multiplier: 8.328125
epsilon: 0.5    noise_multiplier: 6.796875
epsilon: 0.6    noise_multiplier: 5.78515625
epsilon: 0.7    noise_multiplier: 5.046875
epsilon: 0.8    noise_multiplier: 4.4931640625
epsilon: 0.9    noise_multiplier: 4.0556640625
epsilon: 1.0    noise_multiplier: 3.70703125
```

Note: all models have batch size 2**12, except `mnist_dpsgd_conv.py`, which uses
2**13.

## Learning Rate

At a epsilon = .5 with 10 epochs, here are some data points on learning rate and
accuracy:
mnist-dpsgd.py: .15 = 75% val accuracy
mnist-post-scale.py: .15 = 61% accuracy

