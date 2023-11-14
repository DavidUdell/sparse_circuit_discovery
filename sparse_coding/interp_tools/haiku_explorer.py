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

for key in torch_tensors.keys():
    print(key)
