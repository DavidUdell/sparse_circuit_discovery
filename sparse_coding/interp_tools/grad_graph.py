# %%
"""
Circuit discovery with gradient methods, with sparse autoencoders.

Implements the unsupervised circuit discovery algorithm in Baulab 2024.
"""


import re

import torch as t
from accelerate import Accelerator
from pygraphviz import AGraph
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from sparse_coding.utils.interface import (
    load_preexisting_graph,
    load_yaml_constants,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
)
from sparse_coding.utils.tasks import recursive_defaultdict
from sparse_coding.interp_tools.utils.hooks import (
    grads_manager,
    prepare_autoencoder_and_indices,
)


# %%
# Load constants.
_, config = load_yaml_constants(__file__)

WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
ATTN_ENCODER_FILE = config.get("ATTN_ENCODER_FILE")
ATTN_ENC_BIASES_FILE = config.get("ATTN_ENC_BIASES_FILE")
ATTN_DECODER_FILE = config.get("ATTN_DECODER_FILE")
ATTN_DEC_BIASES_FILE = config.get("ATTN_DEC_BIASES_FILE")
MLP_ENCODER_FILE = config.get("MLP_ENCODER_FILE")
MLP_ENC_BIASES_FILE = config.get("MLP_ENC_BIASES_FILE")
MLP_DECODER_FILE = config.get("MLP_DECODER_FILE")
MLP_DEC_BIASES_FILE = config.get("MLP_DEC_BIASES_FILE")
RESID_TOKENS_FILE = config.get("TOP_K_INFO_FILE")
ATTN_TOKENS_FILE = config.get("ATTN_TOKEN_FILE")
MLP_TOKENS_FILE = config.get("MLP_TOKEN_FILE")
GRADS_FILE = config.get("GRADS_FILE")
GRADS_DOT_FILE = config.get("GRADS_DOT_FILE")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

# Or some other means of thresholding approximated effects.
NUM_TOP_EFFECTS: int = 10

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Log to wandb.
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)

# %%
# Load and prepare the model.
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    output_hidden_states=True,
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator: Accelerator = Accelerator()

model = accelerator.prepare(model)
layer_range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Prepare all layer range autoencoders.
# Residual autoencoders
res_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    RESID_TOKENS_FILE,
    accelerator,
    __file__,
)
res_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    RESID_TOKENS_FILE,
    accelerator,
    __file__,
)

# Attention autoencoders
attn_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ATTN_ENCODER_FILE,
    ATTN_ENC_BIASES_FILE,
    ATTN_TOKENS_FILE,
    accelerator,
    __file__,
)
attn_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ATTN_DECODER_FILE,
    ATTN_DEC_BIASES_FILE,
    ATTN_TOKENS_FILE,
    accelerator,
    __file__,
)

# MLP autoencoders
mlp_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    MLP_ENCODER_FILE,
    MLP_ENC_BIASES_FILE,
    MLP_TOKENS_FILE,
    accelerator,
    __file__,
)
mlp_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    MLP_DECODER_FILE,
    MLP_DEC_BIASES_FILE,
    MLP_TOKENS_FILE,
    accelerator,
    __file__,
)

# %%
# Load preexisting graph, if available.
graph = load_preexisting_graph(MODEL_DIR, GRADS_DOT_FILE, __file__)
if graph is None:
    graph = AGraph(directed=True)

save_graph_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_FILE}"
)
save_dot_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_DOT_FILE}"
)

# %%
# Define cross-entropy loss.
metric = t.nn.CrossEntropyLoss()

# %%
# Model passes.
print("Prompt:")
print()
print(PROMPT)

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
acts_dict: dict = None
grads_dict: dict = None
marginal_grads_dict: dict = recursive_defaultdict()

with grads_manager(
    model,
    layer_range,
    res_enc_and_biases,
    res_dec_and_biases,
    attn_enc_and_biases,
    attn_dec_and_biases,
    mlp_enc_and_biases,
    mlp_dec_and_biases,
) as acts_and_grads:

    # Forward pass installs all backward hooks.
    output = model(**inputs)

    # Metric backward pass.
    metric(
        output.logits.squeeze(),
        inputs["input_ids"].squeeze(),
    ).backward(retain_graph=True)

    acts_dict, grads_dict = acts_and_grads

    # Add model_dim activations to acts_dict, if needed.
    for grad in grads_dict:
        if "res_error" in grad:
            idx: int = int(grad.split("_")[-1])
            act: t.Tensor = output.hidden_states[idx]
            acts_dict[grad] = act

    # Compute Jacobian-vector products.
    for act in acts_dict:
        assert act in grads_dict

    for loc, grad in grads_dict.items():
        assert loc in acts_dict

        grad = grad.squeeze().unsqueeze(0).detach()
        act = acts_dict[loc].squeeze().unsqueeze(0)

        if re.match("res_", loc) is not None:
            mlp_confound: str = re.sub("(res_error_|res_)", "mlp_", loc)
            mlp_error_confound: str = re.sub(
                "(res_error_|res_)", "mlp_error_", loc
            )
            # The jvp at an activation is a scalar with gradient tracking that
            # represents how well the model would do on the loss metric in that
            # forward pass, if all later modules were replaced with a best-fit
            # first-order Taylor approximation.
            jvp = (
                t.einsum(
                    "...sd,...sd->...s",
                    grads_dict[mlp_confound].detach(),
                    acts_dict[mlp_confound],
                ).squeeze()
                + t.einsum(
                    "...sd,...sd->...s",
                    grads_dict[mlp_error_confound].detach(),
                    acts_dict[mlp_error_confound],
                ).squeeze()
            )
            jvp[-1].backward(retain_graph=True)
            _, jvp_grads = acts_and_grads

        weighted_prod = t.einsum("bsd,bsd->bsd", grad, act)[:, -1, :].squeeze()
        for dim_idx, prod in enumerate(tqdm(weighted_prod, desc=loc)):
            prod.backward(retain_graph=True)
            _, marginal_grads = acts_and_grads

            down_layer_idx = int(loc.split("_")[-1])
            up_layer_idx = down_layer_idx - 1
            if up_layer_idx not in layer_range:
                continue

            # res_x
            # mlp_x
            # attn_x
            # res_error_x
            # mlp_error_x
            # attn_error_x
            if "attn_" in loc:
                # Upstream res_
                marginal_grads_dict[f"res_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = marginal_grads[f"res_{up_layer_idx}"]
                marginal_grads_dict[f"res_error_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = marginal_grads[f"res_error_{up_layer_idx}"]
            elif "mlp_" in loc:
                # Same-layer attn_
                marginal_grads_dict[f"attn_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = marginal_grads[f"attn_{down_layer_idx}"]
                marginal_grads_dict[f"attn_error_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = marginal_grads[f"attn_error_{down_layer_idx}"]
            elif "res_" in loc:
                # Upstream res_; double-counting corrections.
                marginal_grads_dict[f"res_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = (
                    marginal_grads[f"res_{up_layer_idx}"]
                    - jvp_grads[f"res_{up_layer_idx}"]
                )
                marginal_grads_dict[f"res_error_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = (
                    marginal_grads[f"res_error_{up_layer_idx}"]
                    - jvp_grads[f"res_error_{up_layer_idx}"]
                )

                # Same-layer mlp_
                marginal_grads_dict[f"mlp_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = marginal_grads[f"mlp_{down_layer_idx}"]
                marginal_grads_dict[f"mlp_error_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = marginal_grads[f"mlp_error_{down_layer_idx}"]
            else:
                raise ValueError("Module location not recognized.")

# %%
# Render the graph.
for edge_type, values in marginal_grads_dict.items():
    # We'll plot only the contributions of the final forward pass:
    values = values.detach().squeeze(0)[-1, :]

    top_values, top_indices = t.topk(values, NUM_TOP_EFFECTS, largest=True)
    top_values, top_indices = top_values.tolist(), top_indices.tolist()
    bot_values, bot_indices = t.topk(values, NUM_TOP_EFFECTS, largest=False)
    bot_values, bot_indices = bot_values.tolist(), bot_indices.tolist()

    for v, dim in zip(top_values + bot_values, top_indices + bot_indices):
        if round(v) == 0.0:
            continue

        print(edge_type, dim, v)

    print()
