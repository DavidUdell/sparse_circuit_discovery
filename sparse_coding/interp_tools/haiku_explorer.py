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

model_len: int = len(haiku_model.params)
print(model_len)
for layer in haiku_model.params:
    print(layer)

# matrix = haiku_model.params.popitem()
# print(matrix)
# print(type(matrix))
# print(len(matrix))
# print(matrix[0])
# print(type(matrix[0]))
# print(matrix[1])
# print(type(matrix[1]))
# print(len(matrix[1]))
# print(matrix[1]["w"])
# print(type(matrix[1]["w"]))
# print(type(t.tensor(np.array(matrix[1]["w"]))))
