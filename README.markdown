# Sparse Circuit Discovery
![Feature graph](header.png)

Automatic circuit discovery in GPT-2 small, using sparse coding.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Status](#project-status)

## Installation
To manually install, just run these commands in the shell:

`git clone https://github.com/DavidUdell/sparse_circuit_discovery`

`cd sparse_circuit_discovery`

`pip install --editable sparse_circuit_discovery`

_Alternatively,_ I have a Docker image [hosted on
DockerHub](https://hub.docker.com/r/davidudell/sparse_circuit_discovery). The
Docker image is especially good for pulling to a remote server.

## Usage
To train and interpret a sparse autoencoder, go to
`sparse_coding/config/central_config.yaml`. There, set your

1. HuggingFace model
repo (`MODEL_DIR`),
2. layer indexes to collect activation data from
(`ACT_LAYERS_SLICE`), and
3. autoencoder training hyperparameters values (`LAMBDA_L1`,
`LEARNING_RATE`, `PROJECTION_FACTOR`).

Acceptable starting values for a range of models are:

|`MODEL_DIR`|EleutherAI/pythia-70m|meta-llama/Llama-2-7b-hf|meta-llama/Llama-2-70b-hf|
|:---:|:---:|:---:|:---:|
|`ACTS_LAYERS_SLICE`| "1:3" | "12:14" | "31:33" |
|`LAMBDA_L1` | 1e-2 | 1 | 3 |
|`LEARNING_RATE` | 3e-3 | 1e-3 | 1e-3 |
|`PROJECTION_FACTOR` | 10 | 10 | 10 |

Once you've saved `central_config.yaml`, run the main interpretability pipeline
with:

`cd sparse_coding`

`python3 pipe.py`

### Notes:
- For the time being, only GPT-2 small and a projection factor of 32 are
  supported, to use preexisting sparse autoencoders.

- A highly interpretable sparse autoencoder will have an L^0 value of 10-100 at
  convergence. Manually tune the `LAMBDA_L1` and `LEARNING_RATE` training
  hyperparameters to get this L^0.

- If you're trying to access a gated HuggingFace model repo, you'll have to
  provide the needed HuggingFace access token in `config/hf_access.yaml`. The
  repo will create this YAML if needed.

- If you're encountering cryptic env variable bugs, ensure you're running CUDA
  Toolkit 12.2 or newer.

## Project Status
Current version is 1.0.0.
