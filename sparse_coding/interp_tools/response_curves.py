# %%
"""Plot the effects of different pinning values on a downstream feature."""


import warnings
import torch as t
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
# dict[int, list[int]]
PINNED_ABLATION_DIM = {1: [1]}
PINNED_CACHE_DIM = {2: [1]}
COEFFICIENT: float = 0.0

RANGE = range(
    PINNED_ABLATION_DIM.keys()[0],
    PINNED_CACHE_DIM.keys()[0],
)

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Load model, etc.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR
    )
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)
accelerator: Accelerator = Accelerator()

# %%
# Prepare relevant autoencoders.
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
