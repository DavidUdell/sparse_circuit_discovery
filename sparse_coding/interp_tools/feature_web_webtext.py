# %%
"""
Mess with autoencoder activation dims during `webtext` and graph effects.

`feature_web_webtext` identifies the sequence positions that most excited each
autoencoder dimension and plots ablation effects at those positions. It relies
on prior cached data from `pipe.py`.

You may need to have logged a HF access token, if applicable.
"""


from collections import defaultdict
import warnings

import numpy as np
import torch as t
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm

from sparse_coding.interp_tools.utils.hooks import hooks_lifecycle
from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_seq,
    load_yaml_constants,
    save_paths,
    sanitize_model_name,
    load_layer_tensors,
    load_layer_feature_indices,
)
from sparse_coding.interp_tools.utils.graphs import graph_causal_effects


# %%
# Load constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
ABLATION_DIM_INDICES_PLOTTED = config.get("ABLATION_DIM_INDICES_PLOTTED", None)
SEED = config.get("SEED", 0)
