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
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm

from sparse_coding.interp_tools.utils.graphs import graph_causal_effects
from sparse_coding.interp_tools.utils.hooks import hooks_manager
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
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
ABLATION_DIM_INDICES_PLOTTED = config.get("ABLATION_DIM_INDICES_PLOTTED", None)
N_EFFECTS = config.get("N_EFFECTS")
SEED = config.get("SEED", 0)

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config to wandb.
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)

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
    ablate_layer_encoder, ablate_layer_bias = load_layer_tensors(
        MODEL_DIR,
        ablate_layer_idx,
        ENCODER_FILE,
        BIASES_FILE,
        __file__,
    )
    ablation_layer_autoencoder = {
        ablate_layer_idx: (
            ablate_layer_encoder,
            ablate_layer_bias,
        ),
    }
    ablate_dims = load_layer_feature_indices(
        MODEL_DIR,
        ablate_layer_idx,
        TOP_K_INFO_FILE,
        __file__,
        [],
    )
    base_cache_dim_index: dict[int, list[int]] = {
        ablate_layer_idx: ablate_dims
    }
    # Base run, to determine top activating sequence positions. I'm
    # repurposing the hooks_lifecycle to cache _at_ the would-be ablated
    # layer, by using its interface in a hacky way.
    with hooks_manager(
        ablate_layer_idx - 1,
        None,
        [ablate_layer_idx],
        base_cache_dim_index,
        model,
        ablation_layer_autoencoder,
        base_activations_all_positions,
        ablate_during_run=False,
    ):
        for sequence in eval_set:
            _ = t.manual_seed(SEED)
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_INTERPED_LEN,
            ).to(model.device)

            try:
                model(**inputs)
            except RuntimeError:
                # Manually clear memory and retry.
                gc.collect()
                model(**inputs)

# %%
# Find each dim's top activation value, and select all positions in the range
# [activation/2, activation], a la Cunningham et al. 2023.
# base_activation indices:
# [ablate_layer_index][None][base_cache_dim_index]
favorite_sequence_positions: dict[tuple[int, int, int], list[int]] = {}

for i in base_activations_all_positions:
    for j in base_activations_all_positions[i]:
        assert j is None, f"Middle index {j} should have been None."
        for k in base_activations_all_positions[i][j]:
            # The t.argmax here finds the top sequence position for each dict
            # index tuple. # favorite_sequence_position indices are now the
            # tuple (ablate_layer_idx, None, base_cache_dim_index).
            activations_tensor = base_activations_all_positions[i][j][k]

            favorite_seq_pos = t.argmax(activations_tensor, dim=1).squeeze()
            max_val = activations_tensor[:, favorite_seq_pos, :].unsqueeze(1)
            min_val = max_val / 2.0
            mask = (activations_tensor >= min_val) & (
                activations_tensor <= max_val
            )

            top_indices: t.Tensor = t.nonzero(mask)[:, 1]
            favorite_sequence_positions[i, j, k] = top_indices.tolist()


# %%
# Run ablations at top sequence positions.
ablated_activations = defaultdict(recursive_defaultdict)
base_activations_top_positions = defaultdict(recursive_defaultdict)

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
        # This inner loop is all setup; it doesn't loop over the forward
        # passes.
        for cache_layer_idx in cache_layer_range:
            ablate_layer_encoder, ablate_layer_bias = load_layer_tensors(
                MODEL_DIR,
                cache_layer_idx,
                ENCODER_FILE,
                BIASES_FILE,
                __file__,
            )
            per_layer_autoencoders: dict[int, tuple[t.Tensor]] = {
                ablate_layer_idx: (
                    ablate_layer_encoder,
                    ablate_layer_bias,
                ),
                cache_layer_idx: (
                    ablate_layer_encoder,
                    ablate_layer_bias,
                ),
            }
            cache_dims = load_layer_feature_indices(
                MODEL_DIR,
                cache_layer_idx,
                TOP_K_INFO_FILE,
                __file__,
                [],
            )
            base_cache_dim_index: dict[int, list[int]] = {
                cache_layer_idx: cache_dims,
            }

        # Ablation run at top activating sequence positions. We use the -1
        # index from the initial top position collection.
        per_seq_positions: list[int] = favorite_sequence_positions[
            ablate_layer_idx - 1, None, ablate_dim
        ]
        truncated_tok_seqs = []
        truncated_seqs_final_indices: list[int] = []
        for per_seq_position in per_seq_positions:
            # per_seq_position is an int idx for a flattened eval_set.
            for sequence_idx, seq in enumerate(eval_set):
                # The tokenizer also takes care of MAX_SEQ_INTERPED_LEN.
                tok_seq = tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_INTERPED_LEN,
                )
                if len(tok_seq["input_ids"]) < per_seq_position:
                    per_seq_position = per_seq_position - len(
                        tok_seq["input_ids"]
                    )
                    continue
                truncated_tok_seqs.append(tok_seq)
                truncated_seqs_final_indices.append(per_seq_position)
                break

        # This is a conventional use of hooks_lifecycle, but we're only passing
        # in as input to the model the top activating sequence, truncated. We
        # run one ablated and once not.
        with hooks_manager(
            ablate_layer_idx,
            ablate_dim,
            layer_range,
            base_cache_dim_index,
            model,
            per_layer_autoencoders,
            base_activations_top_positions,
            ablate_during_run=False,
        ):
            for s in truncated_tok_seqs:
                top_input = s.to(model.device)
                _ = t.manual_seed(SEED)
                try:
                    model(**top_input)
                except RuntimeError:
                    gc.collect()
                    model(**top_input)

        with hooks_manager(
            ablate_layer_idx,
            ablate_dim,
            layer_range,
            base_cache_dim_index,
            model,
            per_layer_autoencoders,
            ablated_activations,
            ablate_during_run=True,
        ):
            for s in truncated_tok_seqs:
                top_input = s.to(model.device)
                _ = t.manual_seed(SEED)
                try:
                    model(**top_input)
                except RuntimeError:
                    gc.collect()
                    model(**top_input)

# %%
# Compute diffs. Recursive defaultdict indices are:
# [ablate_layer_idx][ablate_dim_idx][cache_dim_idx]
act_diffs: dict[tuple[int, int, int], t.Tensor] = {}
for i in ablated_activations:
    for j in ablated_activations[i]:
        for k in ablated_activations[i][j]:
            ablate_vec = ablated_activations[i][j][k]
            base_vec = base_activations_top_positions[i][j][k]

            assert (
                ablate_vec.shape == base_vec.shape == (1, base_vec.size(1), 1)
            ), dedent(
                f"""
                Shape mismatch between ablated and base vectors for ablate
                layer {i}, ablate dim {j}, and cache dim {k}; ablate shape
                {ablate_vec.shape} and base shape {base_vec.shape}.
                """
            )

            # The truncated seqs were all flattened. Now we just want what
            # would be the last position of each sequence.
            act_diffs[i, j, k] = 0.0
            for x in truncated_seqs_final_indices:
                act_diffs[i, j, k] += ablate_vec[:, x, :] - base_vec[:, x, :]

# There should be any overall effect.
OVERALL_EFFECTS = 0.0
for i, j, k in act_diffs:
    OVERALL_EFFECTS += abs(act_diffs[i, j, k].item())
assert OVERALL_EFFECTS != 0.0, "Ablate hook effects sum to exactly zero."

sorted_diffs = dict(sorted(act_diffs.items()), key=lambda x: abs(x[-1].item()))

if N_EFFECTS is not None:
    select_diffs = dict(list(sorted_diffs.items())[:N_EFFECTS])
else:
    select_diffs = sorted_diffs

save_path: str = save_paths(
    __file__,
    f"{sanitize_model_name(MODEL_DIR)}/feature_web.svg",    
)
# wandb wants a flat dict indexed by strings.
raw_diffs: dict[str, float] = {}
for i, j, k in sorted_diffs:
    raw_diffs[f"{i}.{j}.{k}"] = sorted_diffs[i, j, k].item()
wandb.log(raw_diffs)

graph_causal_effects(
    select_diffs,
    MODEL_DIR,
    TOP_K_INFO_FILE,
    OVERALL_EFFECTS,
    __file__,
).draw(
    save_path,
    format="svg",
    prog="dot",
)

# Read the .svg into a `wandb` artifact.
artifact = wandb.Artifact("feature_web", type="directed_graph")
artifact.add_file(save_path)
wandb.log_artifact(artifact)

# %%
# Wrap up logging.
wandb.finish()
