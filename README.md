# Sparse Circuit Discovery
Automatic circuit discovery in large language models, using sparse coding.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Status](#project-status)

## Installation
To manually install, just run these commands in the shell:
```
git clone https://github.com/DavidUdell/sparse_circuit_discovery

cd sparse_circuit_discovery

pip install --editable sparse_circuit_discovery
```
_Alternatively,_ I have a Docker image [hosted on
DockerHub](https://hub.docker.com/r/davidudell/sparse_circuit_discovery).
The Docker image is especially good for pulling to a remote server.

## Usage
To train and interpret a sparse autoencoder, go to
`sparse_coding/config/central_config.yaml`. In that YAML, set your

1. HuggingFace Transformers model
repository (`MODEL_DIR`),
2. layer to collect model activations at
(`ACTS_LAYER`), and
3. autoencoder training hyperparameters (`LAMBDA_L1`,
`LEARNING_RATE`, `PROJECTION_FACTOR`).

To help get you started, here are decent starting values for a few HuggingFace models:

|`MODEL_DIR`|`ACTS_LAYER`|`LAMBDA_L1`|`LEARNING_RATE`| `PROJECTION_FACTOR`|
|---|:---:|:---:|:---:|:---:|
|EleutherAI/pythia-70m | 2 | 1.0e-2 | 1.0e-2 | 10 |
|meta-llama/Llama-2-7b-hf | 13 | 1.0 | 1.0e-3 | 10 |
|meta-llama/Llama-2-70b-hf | 32 | 3.0 | 1.0e-3 | 10 |

Once you've set YAML values, run the activation data collection, autoencoder
training, and autoencoder interpretation pipeline running with:
```
cd sparse_coding

python3 pipe.py
```

### Notes:
A highly interpretable sparse autoencoder will have an L^0 value of 10-100 at
convergence. Manually tune the `LAMBDA_L1` and `LEARNING_RATE` hyperparameters
to achieve this value.

Small models like Pythia 70M should be run with `LARGE_MODEL_MODE: False`.

If you're trying to access a gated HuggingFace model repo, you'll have to
provide a corresponding HuggingFace access token in
`sparse_coding/act_access.yaml`.

## Project Status
Project is currently WIP. Current version is 0.1.2.
