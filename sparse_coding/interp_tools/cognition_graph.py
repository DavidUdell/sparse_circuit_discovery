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
from pygraphviz import AGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm
import wandb

from sparse_coding.interp_tools.utils.computations import calc_act_diffs
from sparse_coding.interp_tools.utils.graphs import (
    color_range_from_scalars,
    label_highlighting,
)
from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
    prepare_dim_indices,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    load_preexisting_graph,
    parse_slice,
    sanitize_model_name,
    save_paths,
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
DIMS_PINNED: dict[int, list[int]] = config.get("DIMS_PINNED", None)
THRESHOLD_EXP = config.get("THRESHOLD_EXP")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED", 0)
# COEFFICIENT = config.get("COEFFICIENT", 0.0)

if THRESHOLD_EXP is None:
    THRESHOLD = 0.0
else:
    THRESHOLD = 2.0**THRESHOLD_EXP

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

print("Prompt:")
print()
for i in eval_set:
    print(i)
print()

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

# Load preexisting graph, if applicable.
graph = load_preexisting_graph(MODEL_DIR, GRAPH_DOT_FILE, __file__)
if graph is None:
    graph = AGraph(directed=True)

save_graph_path: str = save_paths(
    __file__,
    f"{sanitize_model_name(MODEL_DIR)}/{GRAPH_FILE}",
)
save_dot_path: str = save_paths(
    __file__,
    f"{sanitize_model_name(MODEL_DIR)}/{GRAPH_DOT_FILE}",
)

minor_effects: int = 0
total_effect: float = 0.0
plotted_effect: float = 0.0

# %%
# For the top sequence positions, run ablations and reduce the ouput.
for ablate_layer_idx in ablate_layer_range:

    # Preprocess layer dimensions, if applicable.
    if ablate_layer_idx == ablate_layer_range[0] or (
        DIMS_PINNED is not None
        and DIMS_PINNED.get(ablate_layer_idx) is not None
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
            ablate_during_run=True,
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
        log_probability_diff: dict = {
            (ablate_layer_idx, dimension): log_probability_diff
        }

        activation_diff: dict[tuple[int, int, int], t.Tensor] = calc_act_diffs(
            altered_activations,
            base_case_activations,
        )

        # Reduce effects to graph data.
        for (_, __, downstream_index), effect in activation_diff.items():
            effect: float = effect.item()
            magnitude: float = abs(effect)
            total_effect += magnitude

            if magnitude <= THRESHOLD or 0.0 == effect:
                minor_effects += 1
                continue

            plotted_effect += magnitude

            graph.add_node(
                f"{ablate_layer_idx}.{dimension}",
                label=label_highlighting(
                    ablate_layer_idx,
                    dimension,
                    MODEL_DIR,
                    TOP_K_INFO_FILE,
                    LOGIT_TOKENS,
                    tokenizer,
                    f"{ablate_layer_idx}.{dimension}",
                    log_probability_diff,
                    __file__,
                ),
                shape="box",
            )

            graph.add_node(
                f"{ablate_layer_idx + 1}.{downstream_index}",
                label=label_highlighting(
                    ablate_layer_idx + 1,
                    downstream_index,
                    MODEL_DIR,
                    TOP_K_INFO_FILE,
                    LOGIT_TOKENS,
                    tokenizer,
                    f"{ablate_layer_idx + 1}.{downstream_index}",
                    log_probability_diff,
                    __file__,
                ),
                shape="box",
            )

            color_min, color_max = color_range_from_scalars(activation_diff)
            if effect > 0.0:
                red: int = 0
                blue: int = 255
            elif effect < 0.0:
                red: int = 255
                blue: int = 0
            alpha = int(255 * magnitude / max(abs(color_max), abs(color_min)))
            rgba_str: str = f"#{red:02x}00{blue:02x}{alpha:02x}"

            graph.add_edge(
                f"{ablate_layer_idx}.{dimension}",
                f"{ablate_layer_idx + 1}.{downstream_index}",
                color=rgba_str,
            )

# %%
# Cleanup graph.
edges = graph.edges()
assert len(edges) == len(set(edges)), "Repeat edges in graph."

unlinked_nodes: int = 0
for node in graph.nodes():
    if len(graph.edges(node)) == 0:
        graph.remove_node(node)
        unlinked_nodes += 1

if total_effect == 0.0:
    raise ValueError("Total effect logged was 0.0")
fraction_included = round(plotted_effect / total_effect, 2)
graph.add_node(f"Effects plotted out of collected: ~{fraction_included*100}%.")

print(
    f"{minor_effects} minor effect(s) were ignored."
    f" {unlinked_nodes} unlinked node(s) were dropped.\n"
)

# %%
# Render and save graph.
graph.write(save_dot_path)
graph.draw(
    save_graph_path,
    format="svg",
    prog="dot",
)

artifact = wandb.Artifact(
    "cognition_graph",
    type="directed_graph",
)
artifact.add_file(save_graph_path)
wandb.log_artifact(artifact)

print(f"Graph saved to {save_graph_path}.")

# %%
# Wrap up logging.
wandb.finish()
