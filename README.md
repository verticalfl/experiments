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
