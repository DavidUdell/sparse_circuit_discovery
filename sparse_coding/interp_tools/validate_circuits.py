# %%
"""Validate circuits with simultaneous ablation studies."""


import gc
import warnings
from collections import defaultdict
from contextlib import ExitStack
from textwrap import dedent

import numpy as np
import torch as t
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
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
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
SEQ_PER_DIM_CAP = config.get("SEQ_PER_DIM_CAP", 10)
# dict[int, list[int]]
VALIDATION_DIMS_PINNED = config.get("VALIDATION_DIMS_PINNED")
LOGIT_TOKENS = config.get("LOGIT_TOKENS")
SEED = config.get("SEED")

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
# Load and prepare the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)
ablate_layer_range: range = layer_range[:-1]

# %%
# Load the `openwetext` validation set.
dataset: list[list[str]] = load_dataset(
    "Elriggs/openwebtext-100k",
    split="train",
)["text"]
dataset_indices: np.ndarray = np.random.choice(
    len(dataset),
    size=len(dataset),
    replace=False,
)
STARTING_META_IDX: int = len(dataset) - NUM_SEQUENCES_INTERPED
eval_indices: np.ndarray = dataset_indices[STARTING_META_IDX:]
eval_set: list[list[str]] = [dataset[i] for i in eval_indices]

# %%
# Prepare all layer autoencoders and layer dim index lists up front.
# layer_encoders: dict[int, tuple[t.Tensor]]
# layer_dim_indices: dict[int, list[int]]
layer_encoders, layer_dim_indices = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)
layer_decoders, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

# %%
# Sanity check the pinned circuit indices.
for k, v in VALIDATION_DIMS_PINNED.items():
    assert (
        k in ablate_layer_range
    ), "Layer range should include one more layer after the last pinned layer."
    for i in v:
        assert i in layer_dim_indices[k]

# %%
# Collect base case data.
base_activations_all_positions = defaultdict(recursive_defaultdict)
for ablate_layer_idx in VALIDATION_DIMS_PINNED:
    # Base run, to determine top activating sequence positions. I'm
    # repurposing the hooks_lifecycle to cache _at_ the would-be ablated
    # layer, by using its interface in a hacky way.
    with hooks_manager(
        ablate_layer_idx - 1,
        None,
        layer_range,
        layer_dim_indices,
        model,
        layer_encoders,
        layer_decoders,
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
# Using collected activation data, select datapoints for the pinned circuit
# dims. `truncated_tok_seqs` is the output of this block, and should contain
# favorite sequences of all pinned dims, assembled in a list.
favorite_sequence_positions: dict[tuple[int, int, int], list[int]] = {}
truncated_tok_seqs = []
for ablate_layer_idx, ablate_dim_indices in VALIDATION_DIMS_PINNED.items():
    for ablate_dim_idx in ablate_dim_indices:
        # The t.argmax here finds the top sequence position for each dict
        # index tuple. # favorite_sequence_position indices are now the
        # tuple (ablate_layer_idx, None, base_cache_dim_index).
        activations_tensor = base_activations_all_positions[
            ablate_layer_idx - 1
        ][None][ablate_dim_idx]

        fave_seq_pos_flat: int = (
            t.argmax(activations_tensor, dim=1).squeeze().item()
        )
        max_val = activations_tensor[:, fave_seq_pos_flat, :].unsqueeze(1)
        min_val = max_val / 2.0
        mask = (activations_tensor >= min_val) & (
            activations_tensor <= max_val
        )

        top_indices: t.Tensor = t.nonzero(mask)[:, 1]

        if top_indices.size(0) <= SEQ_PER_DIM_CAP:
            choices = top_indices.tolist()
        else:
            # Solves the problem of densely activating features taking too many
            # forward passes.
            superset_acts = activations_tensor.squeeze()[top_indices]
            meta_indices = t.topk(superset_acts, SEQ_PER_DIM_CAP).indices
            choices = top_indices[meta_indices].tolist()

        favorite_sequence_positions[ablate_layer_idx, None, ablate_dim_idx] = (
            choices
        )

for ablate_layer_idx, ablate_dim_indices in VALIDATION_DIMS_PINNED.items():
    for ablate_dim_idx in ablate_dim_indices:
        for fav_seq_pos in favorite_sequence_positions[
            ablate_layer_idx, None, ablate_dim_idx
        ]:
            for seq in eval_set:
                # The tokenizer also takes care of MAX_SEQ_INTERPED_LEN.
                tok_seq = tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_INTERPED_LEN,
                )
                # fav_seq_pos is the index for a flattened eval_set.
                if tok_seq["input_ids"].size(-1) < fav_seq_pos:
                    fav_seq_pos = fav_seq_pos - tok_seq["input_ids"].size(-1)
                    continue
                if tok_seq["input_ids"].size(-1) >= fav_seq_pos:
                    tok_seq = tokenizer(
                        seq,
                        return_tensors="pt",
                        truncation=True,
                        max_length=fav_seq_pos + 1,
                    )
                    truncated_tok_seqs.append(tok_seq)
                    break
                raise ValueError("fav_seq_pos out of range.")

        assert len(truncated_tok_seqs) > 0, dedent(
            f"No truncated sequences for {ablate_layer_idx}.{ablate_dim_idx}."
        )

# %%
# Validate the pinned circuit with ablations. Base case first.
BASE_LOGITS = None
for seq in truncated_tok_seqs:
    outputs = model(**seq.to(model.device))
    logit = outputs.logits[:, -1, :]
    if BASE_LOGITS is None:
        BASE_LOGITS = logit
    else:
        BASE_LOGITS = t.cat((BASE_LOGITS, logit), dim=0)

ALTERED_LOGITS = None
with ExitStack() as stack:
    for k, v in VALIDATION_DIMS_PINNED.items():
        stack.enter_context(
            hooks_manager(
                k,
                v,
                layer_range,
                {k + 1: []},
                model,
                layer_encoders,
                layer_decoders,
                defaultdict(list),
            )
        )

    for seq in truncated_tok_seqs:
        outputs = model(**seq.to(model.device))
        logit = outputs.logits[:, -1, :]
        if ALTERED_LOGITS is None:
            ALTERED_LOGITS = logit
        else:
            ALTERED_LOGITS = t.cat((ALTERED_LOGITS, logit), dim=0)

# %%
# Compute and display logit diffs.
prob_diff = t.nn.functional.softmax(
    ALTERED_LOGITS, dim=-1
) - t.nn.functional.softmax(BASE_LOGITS, dim=-1)

prob_diff = prob_diff.mean(dim=0)
positive_tokens = prob_diff.topk(LOGIT_TOKENS).indices
negative_tokens = prob_diff.topk(LOGIT_TOKENS, largest=False).indices
token_ids = t.cat((positive_tokens, negative_tokens), dim=0)

for meta_idx, token_id in enumerate(token_ids):
    if meta_idx == len(positive_tokens):
        print()

    token = tokenizer.convert_ids_to_tokens(token_id.item())
    token = tokenizer.convert_tokens_to_string([token])
    token = token.replace("\n", "\\n")

    print(
        token,
        str(round(prob_diff[token_id.item()].item() * 100, 2)) + "%",
        sep="  ",
    )

# %%
# Wrap up logging.
wandb.finish()
