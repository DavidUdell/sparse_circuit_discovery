"""Explore the structure of the compiled Haiku RASP model."""


import numpy as np
import torch as t
from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.lib import make_frac_prevs


haiku_model = compiler.compiling.compile_rasp_to_model(
    make_frac_prevs(rasp.tokens == "x"),
    vocab={"w", "x", "y", "z"},
    max_seq_len=5,
    compiler_bos="BOS",
)

torch_tensors = {}
for layer in haiku_model.params:
    for matrix in haiku_model.params[layer]:
        tensor_name = f"{layer}_{matrix}"
        torch_tensors[tensor_name] = t.tensor(
            np.array(haiku_model.params[layer][matrix])
        )

for key in torch_tensors:
    print(key)

# haiku_model
#   apply
#   forward
#   get_compiled_model
#   model_config
#       activation_function
#       causal
#       dropout_rate
#       key_size
#       layer_norm
#       mlp_hidden_size
#       num_heads
#       num_layers
#   output_encoder
#       bos_encoding
#       bos_token
#       decode
#       encode
#       pad_encoding
#       pad_token
#   params
#   residual_labels
#   input_encoder
#       bos_encoding
#       bos_token
#       decode
#       encode
#       encoding_map
#       enforce_bos
#       pad_encoding
#       pad_token
#       vocab_size


print(haiku_model.apply(["BOS", "w", "x", "y", "z"]).decoded)
