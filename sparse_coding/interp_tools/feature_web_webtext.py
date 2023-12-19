# %%
"""
Mess with autoencoder activation dims during `webtext` and graph effects.

`feature_web_webtext` identifies the sequence positions that most excited each
autoencoder dimension and plots ablation effects at those positions. It relies
on prior cached data from `pipe.py`.

You may need to have logged a HF access token, if applicable.
"""


import gc
import warnings
from collections import defaultdict

import numpy as np
import torch as t
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm

from sparse_coding.interp_tools.utils.graphs import graph_causal_effects
from sparse_coding.interp_tools.utils.hooks import hooks_lifecycle
from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_range,
    load_yaml_constants,
    save_paths,
    sanitize_model_name,
    load_layer_tensors,
    load_layer_feature_indices,
)
from sparse_coding.utils.tasks import recursive_defaultdict


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

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Load model, etc.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)
ablate_layer_range: range = layer_range[:-1]
cache_layer_range: range = layer_range[1:]

# %%
# Load the complementary `openwebtext` dataset subset, relative to
# `collect_acts_webtext`.
dataset: list[list[str]] = load_dataset(
    "Elriggs/openwebtext-100k",
    split="train",
)["text"]
dataset_indices: np.ndarray = np.random.choice(
    len(dataset), size=len(dataset), replace=False
)
eval_indices: np.ndarray = dataset_indices[:NUM_SEQUENCES_INTERPED:-1]
eval_set: list[list[str]] = [
    dataset[i] for i in eval_indices
]

# %%
# Run interp.
base_activations = defaultdict(recursive_defaultdict)

for ablate_layer_idx in ablate_layer_range:
    for cache_layer_idx in cache_layer_range:
        cache_layer_encoder, cache_layer_bias = load_layer_tensors(
            MODEL_DIR,
            cache_layer_idx,
            ENCODER_FILE,
            BIASES_FILE,
            __file__,
        )
        tensors_per_layer = {
            cache_layer_idx: (
                cache_layer_encoder,
                cache_layer_bias,
            ),
        }
        cache_dims = load_layer_feature_indices(
            MODEL_DIR,
            cache_layer_idx,
            TOP_K_INFO_FILE,
            __file__,
            [],
        )
        cache_dim_indices: dict[int, list[int]] = {
            cache_layer_idx: cache_dims,
        }
        np.random.seed(SEED)
        # Base run, to determine top activating sequence positions.
        with hooks_lifecycle(ablate_layer_idx,
                            None,
                            layer_range,
                            cache_dim_indices,
                            model,
                            tensors_per_layer,
                            base_activations,
                            ablate_during_run=False):
            for sequence in tqdm(eval_set):
                try:
                    inputs = tokenizer(
                        sequence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_SEQ_INTERPED_LEN,
                    ).to(model.device)
                    model(**inputs)
                except RuntimeError:
                    # Manually clear memory and retry.
                    gc.collect()
                    inputs = tokenizer(
                        sequence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_SEQ_INTERPED_LEN,
                    ).to(model.device)
                    model(**inputs)

# Pare down to each dimension's top activating sequence position.
favorite_sequence_positions = {}
for i in base_activations:
    for j in base_activations[i]:
        for k in base_activations[i][j]:
            favorite_sequence_positions[(i, j, k)] = np.argmax(
                base_activations[i][j][k]
            )
            print(base_activations[i][j][k].shape)

# Now run ablations at favorite sequence positions.
ablated_activations = defaultdict(recursive_defaultdict)
for ablate_layer_idx in ablate_layer_range:
    ablate_layer_encoder, ablate_layer_bias = load_layer_tensors(
        MODEL_DIR,
        ablate_layer_idx,
        ENCODER_FILE,
        BIASES_FILE,
        __file__,
    )
    ablate_dim_indices = load_layer_feature_indices(
        MODEL_DIR,
        ablate_layer_idx,
        TOP_K_INFO_FILE,
        __file__,
        [],
    )
    for cache_layer_idx in cache_layer_range:
        cache_layer_encoder, cache_layer_bias = load_layer_tensors(
            MODEL_DIR,
            cache_layer_idx,
            ENCODER_FILE,
            BIASES_FILE,
            __file__,
        )
        tensors_per_layer: dict[int, tuple[t.Tensor]] = {
            ablate_layer_idx: (
                ablate_layer_encoder,
                ablate_layer_bias,
            ),
            cache_layer_idx: (
                cache_layer_encoder,
                cache_layer_bias,
            ),
        }
        cache_dims = load_layer_feature_indices(
            MODEL_DIR,
            cache_layer_idx,
            TOP_K_INFO_FILE,
            __file__,
            [],
        )
        np.random.seed(SEED)
        # Base run, to determine top activating sequence positions.
        with hooks_lifecycle(ablate_layer_idx,
                            None,
                            layer_range,
                            cache_dims,
                            model,
                            tensors_per_layer,
                            ablated_activations,
                            ablate_during_run=True):
            for idx, sequence in enumerate(tqdm(eval_set)):
                try:
                    sequence = tokenizer(
                        sequence,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_SEQ_INTERPED_LEN,
                    ).to(model.device)
                    model(**sequence)
                except RuntimeError:
                    # Manually clear memory and retry.
                    gc.collect()
                    sequence = tokenizer(
                        sequence,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_SEQ_INTERPED_LEN,
                    ).to(model.device)
                    model(**sequence)
