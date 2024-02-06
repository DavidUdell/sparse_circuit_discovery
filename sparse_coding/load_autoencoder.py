# %%
"""Load sparse autoencoders from HuggingFace."""


import torch as t
import transformers

from sparse_coding.utils.interface import load_yaml_constants, parse_slice

# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

BIASES_FILE = config.get("BIASES_FILE")
ENCODER_FILE = config.get("ENCODER_FILE")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))


# %%
# Loading functionality.
def load_autoencoder(
    autoencoder_repository: str,
    encoder_file,
    biases_file,
    model_dir,
    acts_layers_slice,
    base_file: str,
) -> None:
    """Save a sparse autoencoder directly to disk."""


# %%
# Call
load_autoencoder()
