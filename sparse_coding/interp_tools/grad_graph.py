# %%
"""
Circuit discovery with gradient methods, with sparse autoencoders.

Implements the unsupervised circuit discovery algorithm in Baulab 2024.
"""


import torch as t
from accelerate import Accelerator
from pygraphviz import AGraph
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

    # Sanity checks
    for act in acts_dict:
        assert act in grads_dict
    for grad in grads_dict:
        assert grad in acts_dict

    # Compute Jacobian-vector products.
    jvp_dict: dict = {}
    for location, grad in grads_dict.items():
        act = acts_dict[location]

        # Shape regularization
        act = act.squeeze()
        act = act.unsqueeze(0)
        grad = grad.squeeze()
        grad = grad.unsqueeze(0)

        jvp = t.einsum("bsd, bsd -> bs", grad, act)
        jvp_last = jvp.squeeze()[-1]
        jvp_dict[location] = jvp_last

    jvp_sum: t.Tensor = t.tensor(
        0.0,
        device=model.device,
        requires_grad=True,
    )
    # We compute a jvp_sum tensor to combine several gradient calculations.
    # Leverages the fact that the gradient of a sum is the sum of the
    # gradients.
    for jvp in jvp_dict.values():
        jvp_sum = jvp_sum + jvp

    edge_grads: dict = None
    # jvp_sum backward pass.
    jvp_sum.backward()

    _, edge_grads = acts_and_grads

# %%
# Compute the final products.
for grad_loc, grad in edge_grads.items():
    grad_loc_bits: tuple = grad_loc.split("_")
    grad_mod = grad_loc_bits[0]
    # Append the error label when present.
    if len(grad_loc_bits) == 3:
        grad_mod += "_" + grad_loc_bits[1]
    grad_idx = grad_loc_bits[-1]

    # Act idx is one up from grad idx.
    act_idx = int(grad_idx) - 1
    if act_idx not in layer_range:
        continue
    act_loc = f"{grad_mod}_{act_idx}"

    product: t.Tensor = grads_dict[grad_loc] * acts_dict[act_loc]
    # Regularize shape
    product = product.squeeze().unsqueeze(0)
    product = product[:, -1, :].squeeze()

    # Threshold
    top_k_values, top_k_indices = t.topk(t.abs(product), NUM_TOP_EFFECTS)
    for i in top_k_indices:
        print(act_loc, f"#{i.item()}", round(product[i].item(), 2))
    print()
