# %%
"""Create a rasp model and cache its weights and biases in the interface."""


from textwrap import dedent

import torch as t

from sparse_coding.utils.configure import load_yaml_constants
from sparse_coding.utils.caching import parse_slice


# %%
# Load up constants.
_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))

if MODEL_DIR != "rasp":
    raise ValueError(
        dedent(
            f"""
            `rasp_cache.py` requires that MODEL_DIR be set to `rasp`, not
            {MODEL_DIR}.
            """
        )
    )
# %%
# Define the rasp model.
