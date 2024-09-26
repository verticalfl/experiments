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

The experiments can be run on a single machine or multiple (note they require
logs of memory so a single machine may not always be sufficient).

To run the experiments on a single machine:
```bash
cd mnist-post-scale
python ./mnist.py --party l
```

In another terminal:
```bash
cd mnist-post-scale
python ./mnist.py --party f
```

To run the experiments on multiple machines, specify a Tensorflow cluster spec:

```bash
python ./mnist.py --party f --cluster_spec '{ "tfshellfeatures": ["localhost:2222"], "tfshelllabels": ["localhost:2223"], }'
```