# %%
"""The RASP model as a `transformer_lens` HookedTransformer."""


import einops
import numpy as np
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from sparse_coding.rasp.haiku_rasp_model import haiku_model


# %%
# Set constants.
DEVICE = "cpu"
ACTIVATION_FUNCTION = "relu"

# %%
# Initialize and pull hyperparams from the Haiku/JAX RASP model.
NUM_HEADS = haiku_model.model_config.num_heads
NUM_LAYERS = haiku_model.model_config.num_layers
HEAD_DIM = haiku_model.model_config.key_size
MLP_DIM = haiku_model.model_config.mlp_hidden_size

if haiku_model.model_config.layer_norm:
    NORMALIZATION_TYPE = "LN"
else:
    NORMALIZATION_TYPE = None
if haiku_model.model_config.causal:
    ATTENTION_TYPE = "causal"
else:
    ATTENTION_TYPE = "bidirectional"

# VOCAB_DIM_TOTAL equals lengh of vocab above plus two, for BOS and PAD.
VOCAB_DIM_TOTAL = haiku_model.params["token_embed"]['embeddings'].shape[0]
DIM_VOCAB_OUT = haiku_model.params["token_embed"]['embeddings'].shape[0] - 2
CONTEXT_LENGTH = haiku_model.params["pos_embed"]['embeddings'].shape[0]
HIDDEN_DIM = haiku_model.params["token_embed"]['embeddings'].shape[1]

# %%
# Pull Haiku model params as a state dict. Reshape everything going into the
# state dict to match NUM_HEADS.
state_dict = {}

state_dict["pos_embed.W_pos"] = haiku_model.params["pos_embed"]['embeddings']
state_dict["embed.W_E"] = haiku_model.params["token_embed"]['embeddings']
# The unembed is just a projection onto the first few elements of the residual
# stream; these store output tokens.
state_dict["unembed.W_U"] = np.eye(HIDDEN_DIM, DIM_VOCAB_OUT)

for layer_idx in range(NUM_LAYERS):
    state_dict[f"blocks.{layer_idx}.attn.W_K"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.b_K"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.W_Q"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.b_Q"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.W_V"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.b_V"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.W_O"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head = HEAD_DIM,
        n_heads = NUM_HEADS
    )
    state_dict[f"blocks.{layer_idx}.attn.b_O"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/attn/linear"
        ]["b"]
    state_dict[f"blocks.{layer_idx}.mlp.W_in"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_1"
        ]["w"]
    state_dict[f"blocks.{layer_idx}.mlp.b_in"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_1"
        ]["b"]
    state_dict[f"blocks.{layer_idx}.mlp.W_out"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_2"
        ]["w"]
    state_dict[f"blocks.{layer_idx}.mlp.b_out"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_2"
        ]["b"]

# %%
# Convert all state_dict weights into torch tensors.
for k, v in state_dict.items():
    state_dict[k] = t.tensor(np.array(v))

# %%
# Initialize the `HookedTransformer` with those hyperparameters and the state
# dict.
config = HookedTransformerConfig(
    n_layers=NUM_LAYERS,
    d_model=HIDDEN_DIM,
    d_head=HEAD_DIM,
    n_ctx=CONTEXT_LENGTH,
    d_vocab=VOCAB_DIM_TOTAL,
    d_vocab_out=DIM_VOCAB_OUT,
    d_mlp=MLP_DIM,
    n_heads=NUM_HEADS,
    act_fn=ACTIVATION_FUNCTION,
    attention_dir=ATTENTION_TYPE,
    normalization_type=NORMALIZATION_TYPE,
    use_attn_result=True,
    use_split_qkv_input=True,
    device=DEVICE,
)

transformer_lens_model = HookedTransformer(config)

if "use_hook_mlp_in" in transformer_lens_model.cfg.to_dict():
    transformer_lens_model.set_use_hook_mlp_in(True)

transformer_lens_model.load_state_dict(state_dict, strict=False)
transformer_lens_model.eval()
