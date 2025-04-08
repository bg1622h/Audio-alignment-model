# Audio-alignment-model

## Description

This repository contains tools for training a model for note recognition and temporal alignment of music audio recordings using MCTC-loss, including scripts for data preprocessing.

## Installation

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

The default config is `src/configs/baseline.yaml`

Model weights and checkpoints are `./saved` in the saved folder

Logging with `wandb` or `cometml` is supported

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

The default config is `src/configs/inference.yaml`

Model predictions are saved in `./data/saved/{save_path}`.

## Useful Links:

Model weights used in the file predict_with_pretrained_model.ipynb: [Unet weights](https://drive.google.com/file/d/1O0mRcNhxBOKMCUd4bHCJY0OBuThvMQ4j/view?usp=sharing)

## Credits

This repository is based on [asr_project_template](https://github.com/Blinorot/pytorch_project_template/tree/example/asr)

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
