# TF-Shell Experiments
This project contains scripts to train models in a privacy preserving way using the tf-shell library.

## General Instructions

- If running a python script fails because dependencies are not installed, the issue is likely because the devcontainer is not being used.
- The devcontainer is defined in ./.devcontainer, using the devcontainer cli. For example:
    ```bash
    devcontainer up --workspace-folder .
    devcontainer exec --workspace-folder . /bin/bash
    ```
- Inside the devcontainer, the training scripts can be run with
    ```bash
    source .venv/bin/activate
    cd training
    python ./training-script-here.py
    ```
- If the container needs to be recreated, use the command:
    ```bash
    devcontainer up --workspace-folder . --remove-existing-container
  Use with caution, as it takes a long time. When adding python dependencies, prefer requirements.txt.

