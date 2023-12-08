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
`sparse_coding/config/central_config.yaml`. There, set your

1. HuggingFace model
repo (`MODEL_DIR`),
2. layer indexes to collect activation data from
(`ACT_LAYERS_SLICE`), and
3. autoencoder training hyperparameters values (`LAMBDA_L1`,
`LEARNING_RATE`, `PROJECTION_FACTOR`).

Acceptable starting values for a range of models are:

|`MODEL_DIR`|`ACT_LAYERS_SLICE`|`LAMBDA_L1`|`LEARNING_RATE`| `PROJECTION_FACTOR`|
|---|:---:|:---:|:---:|:---:|
|EleutherAI/pythia-70m | "2:3" | 1.0e-2 | 1.0e-2 | 10 |
|meta-llama/Llama-2-7b-hf | "13:14" | 1.0 | 1.0e-3 | 10 |
|meta-llama/Llama-2-70b-hf | "32:33" | 3.0 | 1.0e-3 | 10 |

Once you've saved the YAML, run the main interpretability pipeline with:
```
cd sparse_coding

python3 pipe.py
```

### Notes:
- A highly interpretable sparse autoencoder will have an L^0 value of 10-100 at
  convergence. Manually tune the `LAMBDA_L1` and `LEARNING_RATE` training
h  yperparameters to get this L^0.

- Small models like Pythia 70M should be run with `LARGE_MODEL_MODE: False`.
  This solves device issues with `accelerate` on too-small models.

- Try to run this on CUDA 12.2 or better. I have ever had env variable bugs on
  CUDA 12.0; I haven't looked into this in great detail, but I notice this fix
  works.

- If you're trying to access a gated HuggingFace model repo, you'll have to
  provide the needed HuggingFace access token in
  `sparse_coding/act_access.yaml`. The script will create this YAML if needed.

## Project Status
Project is currently a WIP. Current version: 0.1.2.
