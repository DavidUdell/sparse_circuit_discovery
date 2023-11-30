# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


import csv
from textwrap import dedent

import numpy as np
import torch as t
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm

from sparse_coding.interp_tools.utils.hooks import (
    rasp_ablations_hook_fac,
    ablations_lifecycle,
)
from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_seq,
    load_yaml_constants,
    save_paths,
    sanitize_model_name,
)
from sparse_coding.utils.tasks import multiple_choice_task
from sparse_coding.rasp.rasp_to_transformer_lens import transformer_lens_model
from sparse_coding.rasp.rasp_torch_tokenizer import tokenize
from sparse_coding.interp_tools.utils.graphs import graph_causal_effects


# %%
# Import constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
NUM_QUESTIONS_EVALED = config.get("NUM_QUESTIONS_EVALED", 800)
NUM_SHOT = config.get("NUM_SHOT", 6)
SEED = config.get("SEED")

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# This pathway validates against just the rasp model.
if MODEL_DIR == "rasp":
    print(
        dedent(
            """
            `feature_web.py` will always use RASP layers 0 and 1 when the model
            directory "rasp" is passed to it.
            """
        )
    )

    # Record the differential downstream effects of ablating each dim.
    prompt = ["BOS", "w", "w", "w", "w", "x", "x", "x", "z", "z"]
    token_ids = tokenize(prompt)

    base_activations = {}
    ablated_activations = {}

    # Cache base activations.
    for residual_idx in range(0, 2):
        for neuron_idx in range(transformer_lens_model.cfg.d_model):
            (  # pylint: disable=unpacking-non-sequence
                _,
                base_activations[residual_idx, neuron_idx],
            ) = transformer_lens_model.run_with_cache(token_ids)
    # Cache ablated activations.
    for residual_idx in range(0, 2):
        for neuron_idx in range(transformer_lens_model.cfg.d_model):
            transformer_lens_model.add_perma_hook(
                "blocks.0.hook_resid_pre",
                rasp_ablations_hook_fac(neuron_idx),
            )

            (  # pylint: disable=unpacking-non-sequence
                _,
                ablated_activations[residual_idx, neuron_idx],
            ) = transformer_lens_model.run_with_cache(token_ids)

            transformer_lens_model.reset_hooks(including_permanent=True)

    # Compute effects.
    activation_diffs = {}

    for layer_idx, neuron_idx in ablated_activations:
        activation_diffs[layer_idx, neuron_idx] = (
            base_activations[(layer_idx, neuron_idx)][
                "blocks.1.hook_resid_pre"
            ]
            .sum(axis=1)
            .squeeze()
            - ablated_activations[(layer_idx, neuron_idx)][
                "blocks.1.hook_resid_pre"
            ]
            .sum(axis=1)
            .squeeze()
        )

    # Plot and save effects.
    graph_causal_effects(activation_diffs).draw(
        save_paths(__file__, "feature_web.png"),
        prog="dot",
    )

# %%
# This pathway finds circuits in the full HF models, from the repo's interface.
# else:
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_seq(model, ACTS_LAYERS_SLICE)

# Load the complementary validation dataset subset.
dataset: dict = load_dataset("truthful_qa", "multiple_choice")
all_indices: np.ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=len(dataset["validation"]["question"]),
    replace=False,
)
validation_indices: list = all_indices[NUM_QUESTIONS_EVALED:].tolist()

ablated_activations = {}
ablations_range: range = layer_range[:-1]

for layer_idx in tqdm(ablations_range, desc="Layer Progress"):
    # Load the per-layer data.
    encoder = t.load(
        save_paths(
            __file__,
            sanitize_model_name(MODEL_DIR)
            + "/"
            + str(layer_idx)
            + "/"
            + ENCODER_FILE,
        )
    )
    encoder = accelerator.prepare(encoder)
    biases = t.load(
        save_paths(
            __file__,
            sanitize_model_name(MODEL_DIR)
            + "/"
            + str(layer_idx)
            + "/"
            + BIASES_FILE,
        )
    )
    biases = accelerator.prepare(biases)
    meaningful_dims = []
    with open(
        save_paths(
            __file__,
            sanitize_model_name(MODEL_DIR)
            + "/"
            + str(layer_idx)
            + "/"
            + TOP_K_INFO_FILE,
        ),
        mode="r",
        encoding="utf-8",
    ) as top_k_info_file:
        reader = csv.reader(top_k_info_file)
        next(reader)
        for row in reader:
            meaningful_dims.append(int(row[0]))

    for ablation_idx in tqdm(
        meaningful_dims, desc="Feature Ablations Progress"
    ):
        with ablations_lifecycle(
            ablation_idx,
            meaningful_dims,
            layer_idx,
            layer_range,
            model,
            encoder,
            biases,
            ablated_activations,
        ):
            multiple_choice_task(
                dataset,
                validation_indices,
                model,
                tokenizer,
                accelerator,
                NUM_SHOT,
                ACTS_LAYERS_SLICE,
                streamlined_mode=True,
            )

# Compute diffs. Baseline activations were cached back in `collect_acts`.
print(ablated_activations)
