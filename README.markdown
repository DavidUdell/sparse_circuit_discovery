![Feature graph](header.png)

# Sparse Circuit Discovery
[![Basic CI; testing and
linting](https://github.com/DavidUdell/sparse_circuit_discovery/actions/workflows/CI.yaml/badge.svg)](https://github.com/DavidUdell/sparse_circuit_discovery/actions/workflows/CI.yaml)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
### Naive Algorithm
Your base of operations is `sparse_coding/config/central_config.yaml`.
The most important hyperparameters are clustered up top:

```
## Key Params
# Throughout, leave out entries for None. Writing in `None` values will get
# you the string "None".
ACTS_LAYERS_SLICE: "3:5"
INIT_THINNING_FACTOR: 1.0
NUM_SEQUENCES_INTERPED: 1
THRESHOLD_EXP: 5.0

# Only pin single dims per layer.
DIMS_PINNED:
  3: [331]
```

In order:
1. `ACTS_LAYERS_SLICE` is a Python slice formatted as a string. It sets which
   layers of the `GPT-2 small` model you'll interpret activations at.
2. `INIT_THINNING_FACTOR` is the fraction of features at the first layer in
   your slice you'll plot. E.g., a fraction of `1.0` will try to plot every
   feature in the layer.
3. `NUM_SEQUENCES_INTERPED` is the number of token sequences used during
   plotting, for the purpose of calculating logit effects and downstream
   feature effects.
4. `THRESHOLD_EXP` is the threshold value exponent for activation differences
   plotted. Smaller differences in activation magnitude than `2**THRESHOLD_EXP`
   are dropped. To plot every non-zero effect, comment out this line.
5. `DIMS_PINNED` is a dictionary of layer indices followed by singleton lists
   of feature indices. If set for the first layer, it will completely override
   `INIT_THINNING_FACTOR`.

Set these values, save `central_config.yaml`, then run interpretability with:

`cd sparse_coding`

`python3 pipe.py`

All data appears in `sparse_coding/data/`.

The last cognition graph you generated is saved as both a `.svg` for you and as
a `.dot` for the computer. If you run the interpretability pipeline again, the
new data will expand upon that old `.dot` file. This way, you can progressively
trace out circuits as you go.

### Gradient-Based Algorithm
There is also a gradient-based algorithm, an implementation of [Marks et al.
(2024).](https://arxiv.org/abs/2403.19647) This algorithm has the advantage of
plotting contributions to the cross-entropy loss _directly_, rather than
plotting contributions to intermediate activation magnitudes. Its
implementation here also extends to GPT-2's sublayers, not just the model's
residual stream.

Key hyperparameters:
1. `ACTS_LAYERS_SLICE` works as above.
2. `NUM_DOWN_NODES` is the number of nodes per sublayer to plot edges _for_. The
   number of plotted notes is twice this value (top-k and bottom-k).
3. `NUM_UP_NODES` is the number of nodes per sublayer to plot edges _to_, from
   down nodes. The number of plotted edges per down node will be twice this
   value.

Save these values in `central_config.yaml`, then run interpretability:

```cd sparse_coding```

```python3 fast.py```

Data appears as it does with the naive algorithm.

However, here you can also choose to render graphs as `.png` files. Change the
extension of `GRADS_FILE` in `central_config.yaml` from `.svg` to `.png` for
that. I separately use [PosteRazor](https://posterazor.sourceforge.io/) to tile
print large `.png` graph files.

### Validating Circuits
There's also an independent circuit validation pipeline, `val.py`. This script
simultaneously ablates all the features that comprise a circuit, to see how the
_overall_ circuit behaves under ablation (rather than just looking at separate
features under independent ablations, the way `pipe.py` cognition graphs do).

To set this up, first set `ACTS_LAYERS_SLICE` to encompass the relevant layers
in GPT-2 small, including one extra layer after
```
## Key Params
# Throughout, leave out entries for None. Writing in `None` values will get
# you the string "None".
ACTS_LAYERS_SLICE: "6:9"
```
and then pin all the features that comprise a given circuit in
`VALIDATION_DIMS_PINNED`.
```
# Here you can freely pin multiple dims per layer.
VALIDATION_DIMS_PINNED:
  6: [8339, 14104, 18854]
  7: [2118]
```
Now run validation with:

`python3 val.py`

## How to Read the Graphs
Consider the cognition graph at the top of this page. Each _box_ with a label
like `4.112` is a feature in a sparse autoencoder. `4` is its layer index,
while `112` is its column index in that layer's autoencoder. You can
cross-reference more comprehensive interpretability data for any given feature
on [Neuronpedia](https://www.neuronpedia.org/gpt2-small).

_Blue tokens_ in sequences in each box represent top feature activations in
their contexts, to a specified length out to either side.

_Blue and red tokens_ in individual boxes at the bottom are the logits most
upweighted/downweighted by that dimension. (_Gray_ is the 0.0 effect edge case.)

_Arrows_ between boxes represent downstream ablation effects on other features.
Red arrows represent downweighting, blue (green in `grad_graph.py`) arrows
represent upweighting, and arrow transparency represents magnitude. E.g., a
pale red arrow is a minor downweighting effect.

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

- You're safe ignoring an `Exception ignored TypeError` _at cleanup_ after
  running `fast.py`, about an `AGraph` object and a `NoneType`.

- `fast.py` uses a unique pruning strategy: it will take autoencoder dims in
  the final `GPT-2 small` layer and prune up from them. So you should start
  from the bottom of the model and progressively plot up from there.

## Project Status
Current version is 1.2.0

The `sae_training` sub-directory is Joseph Bloom's, a dependency for importing
his pretrained sparse autoencoders from HF Hub.

The Neuronpedia wiki is Johnny Lin's.
