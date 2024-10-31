# Hadal Experiments

## Setup

Use python 3.10 or 3.11 (3.12 is not supported yet because tensorflow-privacy
depends on an older version of tensorflow which does not support 3.12. Note
Ubuntu 24.04 uses python 3.12 by default which is not compatible).

Create a python environment:
```bash
sudo apt install python3-venv
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running

The experiments can be run on a single machine or multiple.

To run the experiments on the same machine on the same TensorFlow process
(simplest):
```bash
python ./mnist.py --party b
```

To run the experiments on a single machine on multiple TensorFlow processes:
```bash
cd mnist-post-scale
python ./mnist.py --party l
python ./mnist.py --party f # In another terminal
```

To run the experiments on multiple machines:

```bash
python ./mnist.py --party f --cluster_spec '{ "tfshellfeatures": ["localhost:2222"], "tfshelllabels": ["localhost:2223"], }'
python ./mnist.py --party l --cluster_spec '{ "tfshellfeatures": ["localhost:2222"], "tfshelllabels": ["localhost:2223"], }' # In another terminal
```

Tips:

- TensorFlow does not interrupt operations. This is problems when you want to
interrupt a running test because some operations take a long time. If you are
running the test with bash, you can use `Ctrl + Z` to suspend the process,
`kill %1` to kill it, and `fg` to clear it from the list.

- The cryptographic techniques can be disabled independently for debugging
purposes. The PostScaleSequential and DpSgdSequential models accept
arguments: `disable_encryption`, `disable_masking`, `disable_noise`.
Each can be disabled independently (with the exception of disable encryption
which also disabled masking).

- The output of the models can be checked for overflow by passing the argument
`check_overflow_INSECURE=True`. Note this requires decrypting intermediate
results and breaks the security guarantees.

- Choosing parameters in the backprop and noise encryption context functions
(e.g. log2_cleartext_sz, scaling_factor) requires care. It is useful to debug
the backprop parameters first by disabling the noise protocol and enabling the
overflow checks. Then, set a large `noise_offset_log2` and choose an appropriate
plaintext size and scaling factor. Then, reduce the noise offset. The same
process can be repeated for the noise encryption context.

- Tensorboard profiler has a bug where it writes data to the wrong directory.
If you want to see profiling information, `cd tflogs/.../train/`, you should
see a file that starts with `events.out...`. Then `mv ../plugins ./`