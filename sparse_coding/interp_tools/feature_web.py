# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from collections import defaultdict
from textwrap import dedent

import numpy as np
import torch as t
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm

from sparse_coding.interp_tools.utils.hooks import (
    rasp_ablations_hook_fac,
    hooks_lifecycle,
)
from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_seq,
    load_yaml_constants,
    save_paths,
    load_layer_tensors,
    load_layer_feature_indices,
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

    for ablate_layer_idx, neuron_idx in ablated_activations:
        activation_diffs[ablate_layer_idx, neuron_idx] = (
            base_activations[(ablate_layer_idx, neuron_idx)][
                "blocks.1.hook_resid_pre"
            ]
            .sum(axis=1)
            .squeeze()
            - ablated_activations[(ablate_layer_idx, neuron_idx)][
                "blocks.1.hook_resid_pre"
            ]
            .sum(axis=1)
            .squeeze()
        )

    # Plot and save effects.
    graph_causal_effects(activation_diffs, rasp=True).draw(
        save_paths(__file__, "feature_web.png"),
        prog="dot",
    )

# %%
# This pathway finds circuits in the full HF models, from the repo's interface.
else:
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()

    layer_range: range = slice_to_seq(model, ACTS_LAYERS_SLICE)
    ablate_range: range = layer_range[:-1]

    # Load the complementary validation dataset subset.
    dataset: dict = load_dataset("truthful_qa", "multiple_choice")
    dataset_indices: np.ndarray = np.random.choice(
        len(dataset["validation"]["question"]),
        size=len(dataset["validation"]["question"]),
        replace=False,
    )
    validation_indices: list = dataset_indices[NUM_QUESTIONS_EVALED:].tolist()

    def recursive_defaultdict():
        """Recursively create a defaultdict."""
        return defaultdict(recursive_defaultdict)

    base_activations = defaultdict(recursive_defaultdict)
    ablated_activations = defaultdict(recursive_defaultdict)

    for meta_index, ablate_layer_idx in enumerate(ablate_range):
        # Ablation layer autoencoder tensors.
        ablate_layer_encoder, ablate_layer_bias = load_layer_tensors(
            MODEL_DIR, ablate_layer_idx, ENCODER_FILE, BIASES_FILE, __file__
        )
        ablate_layer_encoder, ablate_layer_bias = accelerator.prepare(
            ablate_layer_encoder, ablate_layer_bias
        )
        tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]] = {
            ablate_layer_idx: (ablate_layer_encoder, ablate_layer_bias)
        }

        # Ablation layer feature-dim indices.
        ablate_dim_indices = []
        ablate_dim_indices = load_layer_feature_indices(
            MODEL_DIR,
            ablate_layer_idx,
            TOP_K_INFO_FILE,
            __file__,
            ablate_dim_indices,
        )

        cache_dim_indices_per_layer = {}
        cache_layer_range = layer_range[meta_index + 1 :]

        for cache_layer_idx in cache_layer_range:
            # Cache layer autoencoder tensors.
            (cache_layer_encoder, cache_layer_bias) = load_layer_tensors(
                MODEL_DIR,
                cache_layer_idx,
                ENCODER_FILE,
                BIASES_FILE,
                __file__,
            )
            cache_layer_encoder, cache_layer_bias = accelerator.prepare(
                cache_layer_encoder, cache_layer_bias
            )
            tensors_per_layer[cache_layer_idx] = (
                cache_layer_encoder,
                cache_layer_bias,
            )

            # Cache layer feature-dim indices.
            layer_cache_dims = []
            layer_cache_dims = load_layer_feature_indices(
                MODEL_DIR,
                cache_layer_idx,
                TOP_K_INFO_FILE,
                __file__,
                layer_cache_dims,
            )
            cache_dim_indices_per_layer[cache_layer_idx] = layer_cache_dims

        for ablate_dim_idx in tqdm(
            ablate_dim_indices, desc="Feature ablations progress"
        ):
            np.random.seed(SEED)
            # Base run.
            with hooks_lifecycle(
                ablate_layer_idx,
                ablate_dim_idx,
                layer_range,
                cache_dim_indices_per_layer,
                model,
                tensors_per_layer,
                base_activations,
                run_with_ablations=False,
            ):
                multiple_choice_task(
                    dataset,
                    validation_indices,
                    model,
                    tokenizer,
                    accelerator,
                    NUM_SHOT,
                    ACTS_LAYERS_SLICE,
                    return_outputs=False,
                )

            np.random.seed(SEED)
            # Ablated run.
            with hooks_lifecycle(
                ablate_layer_idx,
                ablate_dim_idx,
                layer_range,
                cache_dim_indices_per_layer,
                model,
                tensors_per_layer,
                ablated_activations,
                run_with_ablations=True,
            ):
                multiple_choice_task(
                    dataset,
                    validation_indices,
                    model,
                    tokenizer,
                    accelerator,
                    NUM_SHOT,
                    ACTS_LAYERS_SLICE,
                    return_outputs=False,
                )

    # %%
    # Compute differential downstream ablation effects.
    # dict[ablation_layer_idx][ablated_dim_idx][downstream_dim]
    activation_diffs = {}

    for i in ablate_range:
        for j in base_activations[i].keys():
            for k in base_activations[i][j].keys():
                activation_diffs[(i, j, k)] = (
                    base_activations[i][j][k].sum(axis=1).squeeze()
                    - ablated_activations[i][j][k].sum(axis=1).squeeze()
                )

    # %%
    # Plot and save effects.
    # `rasp=False` is quite ugly; I'll want to factor that out by giving both
    # the rasp and full-scale models common output shapes with some squeezing.
    # Then, `graph_causal_effects` can be a single common call outside the
    # if/else.
    graph_causal_effects(activation_diffs, rasp=False).draw(
        save_paths(__file__, "feature_web.png"),
        prog="dot",
    )
