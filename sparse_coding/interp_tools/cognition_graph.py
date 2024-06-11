# %%
"""
Mess with autoencoder activation dims during `webtext` and graph effects.

`directed_graph_webtext` identifies the sequence positions that most excited
each autoencoder dimension and plots ablation effects at those positions. It
relies on prior cached data from `pipe.py`.

You may need to have logged a HF access token, if applicable.
"""


import warnings
from collections import defaultdict
from textwrap import dedent

import numpy as np
import torch as t
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm
import wandb

from sparse_coding.interp_tools.utils.computations import calc_act_diffs
from sparse_coding.interp_tools.utils.graphs import graph_and_log
from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
    prepare_dim_indices,
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
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
GRAPH_FILE = config.get("GRAPH_FILE")
GRAPH_DOT_FILE = config.get("GRAPH_DOT_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
SEQ_PER_DIM_CAP = config.get("SEQ_PER_DIM_CAP", 10)
INIT_THINNING_FACTOR = config.get("INIT_THINNING_FACTOR", None)
BRANCHING_FACTOR = config.get("BRANCHING_FACTOR")
DIMS_PINNED: dict[int, list[int]] = config.get("DIMS_PINNED", None)
THRESHOLD = config.get("THRESHOLD", 0.0)
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED", 0)
# COEFFICIENT = config.get("COEFFICIENT", 0.0)


if DIMS_PINNED is not None:
    for v in DIMS_PINNED.values():
        assert isinstance(v, list) and len(v) == 1, dedent(
            """
            In this script, DIMS_PINNED for ablations should be a dict of
            singleton index lists.
            """
        )

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
# Fix the validation set.
eval_set: list[str] = [PROMPT]

print("Prompt is as follows:")
for i in eval_set:
    print(i)

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
# For the top sequence positions, run ablations and reduce the ouput.
for ablate_layer_idx in ablate_layer_range:

    # Preprocess layer dimensions, if applicable.
    if ablate_layer_idx == ablate_layer_range[0] or (
        DIMS_PINNED is not None and
        DIMS_PINNED.get(ablate_layer_idx) is not None
    ):
        # list[int]
        layer_dim_indices[ablate_layer_idx] = prepare_dim_indices(
            INIT_THINNING_FACTOR,
            DIMS_PINNED,
            layer_dim_indices[ablate_layer_idx],
            ablate_layer_idx,
            layer_range,
            SEED,
        )

    # Tokenize and truncate prompts.
    token_sequences = [
        tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_INTERPED_LEN,
        )
        for context in eval_set
    ]

    # Collect the layer base case data.
    BASE_LOGITS = None
    base_case_activations = defaultdict(recursive_defaultdict)

    with hooks_manager(
        ablate_layer_idx,
        None,
        layer_range,
        layer_dim_indices,
        model,
        layer_encoders,
        layer_decoders,
        base_case_activations,
        ablate_during_run=False,
    ):
        for sequence in token_sequences:
            sequence = sequence.to(model.device)
            _ = t.manual_seed(SEED)

            output = model(**sequence)
            logit = output.logits[:, -1, :].cpu()

            if BASE_LOGITS is None:
                BASE_LOGITS = logit
            elif isinstance(BASE_LOGITS, t.Tensor):
                BASE_LOGITS = t.cat([BASE_LOGITS, logit], dim=0)

    # At a layer, loop through the ablation dimensions, reducing each.
    assert len(token_sequences) > 0
    for dimension in tqdm(
        layer_dim_indices[ablate_layer_idx],
        desc="Dimensions Progress",
    ):
        ALTERED_LOGITS = None
        altered_activations = defaultdict(recursive_defaultdict)

        with hooks_manager(
            ablate_layer_idx,
            dimension,
            layer_range,
            layer_dim_indices,
            model,
            layer_encoders,
            layer_decoders,
            altered_activations,
            albate_during_run=True,
        ):
            for sequence in token_sequences:
                _ = t.manual_seed(SEED)

                output = model(**sequence)
                logit = output.logits[:, -1, :].cpu()

                if ALTERED_LOGITS is None:
                    ALTERED_LOGITS = logit
                elif isinstance(ALTERED_LOGITS, t.Tensor):
                    ALTERED_LOGITS = t.cat([ALTERED_LOGITS, logit], dim=0)

        # Logits immediately become probability differences.
        log_probability_diff = -t.nn.functional.log_softmax(
            ALTERED_LOGITS,
            dim=-1,
        ) + t.nn.functional.log_softmax(
            BASE_LOGITS,
            dim=-1,
        )

        # Postprocess the altered activations, if applicable.
        MOST_AFFECTED_DIMENSIONS = None

        if BRANCHING_FACTOR is not None:
            assert isinstance(BRANCHING_FACTOR, int)

            WORKING_TENSOR = None
            cache_indices = list(
                altered_activations[ablate_layer_idx][dimension].keys()
            )

            # Build tensor block of effects from the activations dict.
            for cache_dim in altered_activations[ablate_layer_idx][dimension]:
                if WORKING_TENSOR is None:
                    WORKING_TENSOR = t.abs(
                        altered_activations[
                            ablate_layer_idx][dimension][cache_dim]
                        - base_case_activations[
                            ablate_layer_idx][None][cache_dim]
                    ).mean(dim=1)
                else:
                    WORKING_TENSOR = t.cat(
                        [
                            WORKING_TENSOR,
                            t.abs(
                                altered_activations[
                                    ablate_layer_idx][dimension][cache_dim]
                                - base_case_activations[
                                    ablate_layer_idx][None][cache_dim]
                            ).mean(dim=1)
                        ]
                    )

            # Sort effects.
            # ordered_meta_indices: t.LongTensor
            _, ordered_meta_indices = t.sort(
                WORKING_TENSOR.squeeze(),
                descending=True,
            )
            assert len(ordered_meta_indices) == len(cache_indices)

            # Keep just the greatest effects, if applicable.
            MOST_AFFECTED_DIMENSIONS = []
            reference_dimensions = set(layer_dim_indices[ablate_layer_idx + 1])

            for i in ordered_meta_indices:
                assert cache_indices[i.item()] in reference_dimensions

                MOST_AFFECTED_DIMENSIONS.append(cache_indices[i.item()])
                if len(MOST_AFFECTED_DIMENSIONS) == BRANCHING_FACTOR:
                    break

        # Continue on with the affected dimensions, if applicable.
        if MOST_AFFECTED_DIMENSIONS is not None:
            layer_dim_indices[ablate_layer_idx + 1] = list(
                set(MOST_AFFECTED_DIMENSIONS)
            )

        activation_diff: dict[tuple[int, int, int], t.Tensor] = calc_act_diffs(
            altered_activations,
            base_case_activations,
        )







# keepers: dict[tuple[int, int], int] = {}

# %%
# Graph effects.
graph_and_log(
    act_diffs,
    keepers,
    BRANCHING_FACTOR,
    MODEL_DIR,
    GRAPH_FILE,
    GRAPH_DOT_FILE,
    TOP_K_INFO_FILE,
    THRESHOLD,
    LOGIT_TOKENS,
    tokenizer,
    probability_diffs,
    __file__,
)

# %%
# Wrap up logging.
wandb.finish()
