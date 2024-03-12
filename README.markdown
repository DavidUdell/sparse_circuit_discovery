# Sparse Circuit Discovery
![Feature graph](header.png)

Circuit discovery in `GPT-2 small`, using sparse autoencoding

### Table of Contents
- [Installation](#installation)
- [User's Guide](#users-guide)
- [How to Read the Graphs](#how-to-read-the-graphs)
- [Errors](#errors)
- [Project Status](#project-status)

## Installation
To manually install, just run these commands in the shell:

`git clone https://github.com/DavidUdell/sparse_circuit_discovery`

`cd sparse_circuit_discovery`

`pip install -e .`

Alternatively, I have a Docker image [on
DockerHub](https://hub.docker.com/r/davidudell/sparse_circuit_discovery). The
Docker image is especially good for pulling to a remote server.

## User's Guide
Your base of operations is `sparse_coding/config/central_config.yaml`.
The most important hyperparameters are clustered up top:

```
## Config Notes
# Throughout, leave out entries for None. Writing in `None` values will get
# you the string "None". Key params here:
ACTS_LAYERS_SLICE: "4:6"
INIT_THINNING_FACTOR: 0.01
NUM_SEQUENCES_INTERPED: 200
SEQ_PER_DIM_CAP: 100

DIMS_PINNED:
 4: 112
```

In order:
1. `ACTS_LAYERS_SLICE` is a Python slice formatted as a string. It sets which
  layers of the `GPT-2 small` model you'll interpret activations at.
2. `INIT_THINNING_FACTOR` is the fraction of features at the first layer in
   your slice you'll plot. I.e., a fraction of `1` will try to plot every
   feature in the layer.
3. `NUM_SEQUENCES_INTERPED` is the number of token sequences used during
   plotting, for the purpose of caluculating logit effects and downstream
   feature effects.
4. `SEQ_PER_DIM_CAP` is the maximum number of top-activating sequences a
   feature can have. I.e., when it equals `NUM_SEQUENCES_INTERPED`, you're
   saying that any feature that fired at every sequence should be interpreted
   over all of those sequences. For computational reasons, we basically want to
   set `NUM_SEQUENCES_INTERPED` as high as we can, and then set this value
   relatively low, so that our interpretability calculations are tractable.
5. `DIMS_PINNED` is a dictionary of layer indices followed by a single feature
   index each. If set for the first layer, it will completely override
   `INIT_THINNING_FACTOR`.

Set these values, save `central_config.yaml`, then run interpretability with:

`cd sparse_coding`

`python3 pipe.py`

Data appears in `sparse_coding/data/`.

The last cognition graph you generated is saved as both a `.svg` for you and as
a `.dot` for the computer. If you run the interpretability pipeline again, the
new data will expand upon that old `.dot` file. This way, you can progressively
trace out circuits as you go.

## How to Read the Graphs
Consider the cognition graph at the top of this page. Each _box_ with a label
like `4.112` is a feature in a sparse autoencoder. `4` is its layer index,
while `112` is its column index in that layer's autoencoder. You can
cross-reference more comprehensive interpretability data for any given feature
on [Neuronpedia](https://www.neuronpedia.org/gpt2-small).

_Blue tokens_ in each box represent top feature activations in their contextual
sequences, to a specified length out to either side.

_Red tokens_ in individual boxes at the bottom are the logits most downweighted
by that ablation.

_Arrows_ between boxes represent downstream ablation effects on other features.
Red arrows indicate downweighting, blue arrows indicate upweighting, and degree
is indicated by color transparency.

## Errors
- I've gimped a lot of repository functionality for now: only `GPT-2 small` and
  a projection factor of 32 are supported, to take advantage of a set of
  preexisting sparse autoencoders.

- When there is an "exactly `0.0` effect from ablations" error, check whether
  your layers slice is compatible with your pinned dim.

- If you're encountering cryptic env variable bugs, ensure you're running CUDA
  Toolkit 12.2 or newer.

- As the shell syntax suggests, Unix-like paths (on MacOS or Linux) are
  currently required, and Windows pathing will probably not play nice with the
  repo.

## Project Status
Current version is 0.3.1.

The `sae_training` sub-directory is Joseph Bloom's, a dependency for importing
his pretrained sparse autoencoders from HF Hub. The Neuronpedia wiki is Johnny
Lin's.
