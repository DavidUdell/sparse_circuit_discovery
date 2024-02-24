# Sparse Circuit Discovery
![Feature graph](header.png)

Circuit discovery in GPT-2 small, using sparse autoencoding

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Status](#project-status)

## Installation
To manually install, just run these commands in the shell:

`git clone https://github.com/DavidUdell/sparse_circuit_discovery`

`cd sparse_circuit_discovery`

`pip install -e .`

_Alternatively,_ I have a Docker image [hosted on
DockerHub](https://hub.docker.com/r/davidudell/sparse_circuit_discovery). The
Docker image is especially good for pulling to a remote server.

## Usage
To train and interpret a sparse autoencoder, go to
`sparse_coding/config/central_config.yaml`. There, set your layer indexes to
collect activation data (`ACT_LAYERS_SLICE`). (Leave other hyperparameters as
they are.)

Once you save `central_config.yaml`, run the interpretability pipeline with:

`cd sparse_coding`

`python3 pipe.py`

### Notes:
- For the time being, only GPT-2 small and a projection factor of 32 are
  supported, to take advantage of set of preexisting sparse autoencoders for
  those values. Additionally, only ablations (rather than feature scaling by
  arbitrary coefficients) are supported.

- If you're encountering cryptic env variable bugs, ensure you're running CUDA
  Toolkit 12.2 or newer.

## Project Status
Current version is 0.2.1.

The `sae_training` sub-directory is Joseph Bloom's, a dependency for importing
his pretrained sparse autoencoders from HF Hub.
