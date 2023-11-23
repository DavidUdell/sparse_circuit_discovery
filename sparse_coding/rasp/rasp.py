"""The RASP model, as a `transformer_lens` HookedTransformer."""


import einops
import numpy as np
import torch as t
from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.lib import make_frac_prevs
from transformer_lens import HookedTransformer, HookedTransformerConfig


bos = "BOS"
device = "cpu"

model = compiler.compiling.compile_rasp_to_model(
        make_frac_prevs(rasp.tokens == "x"),
        vocab={"w", "x", "y", "z"},
        max_seq_len=10,
        compiler_bos="BOS",
)

n_heads = model.model_config.num_heads
n_layers = model.model_config.num_layers
d_head = model.model_config.key_size
d_mlp = model.model_config.mlp_hidden_size
act_fn = "relu"
normalization_type = "LN"  if model.model_config.layer_norm else None
attention_type = "causal"  if model.model_config.causal else "bidirectional"


n_ctx = model.params["pos_embed"]['embeddings'].shape[0]
# Equivalent to length of vocab, with BOS and PAD at the end
d_vocab = model.params["token_embed"]['embeddings'].shape[0]
# Residual stream width, I don't know of an easy way to infer it from the above config.
d_model = model.params["token_embed"]['embeddings'].shape[1]

# Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
d_vocab_out = model.params["token_embed"]['embeddings'].shape[0] - 2

cfg = HookedTransformerConfig(
    n_layers=n_layers,
    d_model=d_model,
    d_head=d_head,
    n_ctx=n_ctx,
    d_vocab=d_vocab,
    d_vocab_out=d_vocab_out,
    d_mlp=d_mlp,
    n_heads=n_heads,
    act_fn=act_fn,
    attention_dir=attention_type,
    normalization_type=normalization_type,
    use_attn_result=True,
    use_split_qkv_input=True,
    device=device,
)
tl_model = HookedTransformer(cfg)
if "use_hook_mlp_in" in tl_model.cfg.to_dict(): # both tracr models include MLPs
    tl_model.set_use_hook_mlp_in(True)

# Extract the state dict, and do some reshaping so that everything has a n_heads dimension
sd = {}
sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
sd["embed.W_E"] = model.params["token_embed"]['embeddings']
# Equivalent to max_seq_len plus one, for the BOS

# The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
# This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

for l in range(n_layers):
    sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

    sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
    sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
    sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
    sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]
print(sd.keys())

# Convert weights to tensors and load into the tl_model

for k, v in sd.items():
    # I cannot figure out a neater way to go from a Jax array to a numpy array lol
    sd[k] = t.tensor(np.array(v))

tl_model.load_state_dict(sd, strict=False)


# Create helper functions to do the tokenization and de-tokenization

INPUT_ENCODER = model.input_encoder
OUTPUT_ENCODER = model.output_encoder

def create_model_input(input, input_encoder=INPUT_ENCODER, device=device):
    encoding = input_encoder.encode(input)
    return t.tensor(encoding).unsqueeze(dim=0).to(device)

input = [bos, "x", "w", "w", "x"]
out = model.apply(input)
print("Original Decoding:", out.decoded)

input_tokens_tensor = create_model_input(input)
logits = tl_model(input_tokens_tensor)

logits, cache = tl_model.run_with_cache(input_tokens_tensor)

for layer in range(tl_model.cfg.n_layers):
    print(f"Layer {layer} Attn Out Equality Check:", np.isclose(cache["attn_out", layer].detach().cpu().numpy(), np.array(out.layer_outputs[2*layer])).all())
    print(f"Layer {layer} MLP Out Equality Check:", np.isclose(cache["mlp_out", layer].detach().cpu().numpy(), np.array(out.layer_outputs[2*layer+1])).all())
