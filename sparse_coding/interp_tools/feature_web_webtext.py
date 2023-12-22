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
from textwrap import dedent

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
STARTING_META_IDX: int = len(dataset) - NUM_SEQUENCES_INTERPED
eval_indices: np.ndarray = dataset_indices[STARTING_META_IDX:]
eval_set: list[list[str]] = [dataset[i] for i in eval_indices]

# %%
# Collect base case data.
base_activations_all_positions = defaultdict(recursive_defaultdict)

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
        with hooks_lifecycle(
            ablate_layer_idx,
            None,
            layer_range,
            cache_dim_indices,
            model,
            tensors_per_layer,
            base_activations_all_positions,
            ablate_during_run=False,
        ):
            for sequence in eval_set:
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

# %%
# Pare down to each dimension's top activating sequence position.
# base_activation indices:
# [ablation_layer_idx][ablated_dim_idx][downstream_dim]
favorite_sequence_positions = {}

for i in base_activations_all_positions:
    for j in base_activations_all_positions[i]:
        for k in base_activations_all_positions[i][j]:
            favorite_sequence_positions[(i, j, k)] = np.argmax(
                base_activations_all_positions[i][j][k]
            )
# favorite_sequence_position indices are now (ablate_layer_idx, None,
# cache_dim)

# %%
# Run ablations at favorite sequence positions.
ablated_activations = defaultdict(recursive_defaultdict)
base_activations_select = defaultdict(recursive_defaultdict)

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
    # Optionally pare down to a target subset of ablate dims.
    if ABLATION_DIM_INDICES_PLOTTED is not None:
        for i in ABLATION_DIM_INDICES_PLOTTED:
            assert i in ablate_dim_indices, dedent(
                f"Index {i} not in `ablate_dim_indices`."
            )
        ablate_dim_indices = ABLATION_DIM_INDICES_PLOTTED

    for ablate_dim in tqdm(ablate_dim_indices, desc="Dim Ablations Progress"):
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
            cache_dim_indices: dict[int, list[int]] = {
                cache_layer_idx: cache_dims,
            }
            for cache_dim in cache_dims:
                np.random.seed(SEED)
                # Ablation run at top activating sequence positions.
                top_seq_position = favorite_sequence_positions[
                    ablate_layer_idx, None, cache_dim
                ]
                abs_top_seq_position = top_seq_position
                # top_seq_position is on flattened eval_set.
                for sequence_idx, position_idx in enumerate(eval_set):
                    if len(position_idx) < top_seq_position:
                        top_seq_position = top_seq_position - len(position_idx)
                        continue

                    dim_token_stream = eval_set[sequence_idx][top_seq_position]
                    base_activations_select[
                        ablate_layer_idx][ablate_dim][cache_dim
                    ] = base_activations_all_positions[
                            ablate_layer_idx][None][cache_dim
                        ][:, abs_top_seq_position, :]
                    break

                with hooks_lifecycle(
                    ablate_layer_idx,
                    ablate_dim,
                    layer_range,
                    cache_dim_indices,
                    model,
                    tensors_per_layer,
                    ablated_activations,
                    ablate_during_run=False,  # Set to True.
                ):
                    try:
                        sequence = tokenizer(
                            dim_token_stream,
                            return_tensors="pt",
                            truncation=True,
                            max_length=MAX_SEQ_INTERPED_LEN,
                        ).to(model.device)
                        model(**sequence)
                    except RuntimeError:
                        # Manually clear memory and retry.
                        gc.collect()
                        sequence = tokenizer(
                            dim_token_stream,
                            return_tensors="pt",
                            truncation=True,
                            max_length=MAX_SEQ_INTERPED_LEN,
                        ).to(model.device)
                        model(**sequence)

# %%
# Compute ablated effects minus base effects. Recursive defaultdict indices
# are: [ablation_layer_idx][ablated_dim_idx][downstream_dim]
activation_diffs = {}
for (
    i
) in ablated_activations.keys():  # pylint: disable=consider-using-dict-items
    for j in ablated_activations[i].keys():
        for k in ablated_activations[i][j].keys():
            assert (
                ablated_activations[i][j][k].shape
                == base_activations_select[i][j][k].shape
            ), dedent(
                f"Shape mismatch between ablated and base activations for "
                f"ablate layer {i}, ablate dim {j}, and downstream dim {k}."
            )
            activation_diffs[i, j, k] = (
                ablated_activations[i][j][k].sum(axis=1).squeeze()
                - base_activations_select[i][j][k].sum(axis=1).squeeze()
            )
# Check that there was any overall effect.
HOOK_EFFECTS_CHECKSUM = 0.0
for i, j, k in activation_diffs:
    HOOK_EFFECTS_CHECKSUM += activation_diffs[i, j, k].sum().item()
assert HOOK_EFFECTS_CHECKSUM != 0.0, "Ablate hook effects sum to exactly zero."

sorted_diffs = dict(
    sorted(activation_diffs.items(), key=lambda x: x[-1].item())
)
graph_causal_effects(
    sorted_diffs,
    MODEL_DIR,
    TOP_K_INFO_FILE,
    __file__,
).draw(
    save_paths(__file__, f"{sanitize_model_name(MODEL_DIR)}/feature_web.svg"),
    format="svg",
    prog="dot",
)
