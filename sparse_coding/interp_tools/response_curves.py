# %%
"""Plot the effects of different pinning values on a downstream feature."""


import warnings

import torch as t
from matplotlib import pyplot as plt
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
)
from sparse_coding.utils.tasks import recursive_defaultdict


_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
SEED = config.get("SEED")
# dict[int, list[int]]. Use syntax for ablation dim pinning.
PINNED_ABLATION_DIM = {3: [953]}
PINNED_CACHE_DIM = {4: [7780]}
COEFFICIENTS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

ABLATION_LAYER: int = list(PINNED_ABLATION_DIM.keys())[0]
RANGE = range(
    ABLATION_LAYER,
    list(PINNED_CACHE_DIM.keys())[0] + 1,
)
ABLATION_DIM: int = list(PINNED_ABLATION_DIM.values())[0][0]
CACHE_DIM: int = list(PINNED_CACHE_DIM.values())[0][0]

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Load model, etc.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator: Accelerator = Accelerator()

# %%
# Prepare relevant autoencoders and the prompt.
layer_encoders, layer_dim_indices = prepare_autoencoder_and_indices(
    RANGE,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

layer_decoders, _ = prepare_autoencoder_and_indices(
    RANGE,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

inputs = tokenizer(PROMPT, return_tensors="pt")

# %%
# Base case run.
base_effects = recursive_defaultdict()

with hooks_manager(
    ABLATION_LAYER,
    ABLATION_DIM,
    RANGE,
    PINNED_CACHE_DIM,
    model,
    layer_encoders,
    layer_decoders,
    base_effects,
    ablate_during_run=False,
):
    _ = model(**inputs)

# %%
# Pinned case run, for each coefficient.
diffs = []
for coefficient in COEFFICIENTS:
    pinned_effects = recursive_defaultdict()

    with hooks_manager(
        ABLATION_LAYER,
        ABLATION_DIM,
        RANGE,
        PINNED_CACHE_DIM,
        model,
        layer_encoders,
        layer_decoders,
        pinned_effects,
        ablate_during_run=True,
        coefficient=coefficient,
    ):
        _ = model(**inputs)

    # Compute and print effect.
    diff = (
        base_effects[ABLATION_LAYER][ABLATION_DIM][CACHE_DIM]
        - pinned_effects[ABLATION_LAYER][ABLATION_DIM][CACHE_DIM]
    )

    diffs.append(diff.item())

# %%
# Plot in graph.
plt.plot(COEFFICIENTS, diffs)
plt.title("Pinning strength effect on downstream feature")
plt.xlabel("Pinning Coefficient")
plt.ylabel("Downstream Activation Difference")
plt.show()
