"""Looking over the rasp_to_torch model."""


import numpy as np
import torch as t
import jax.numpy as jnp

from sparse_coding.rasp.rasp_to_torch import RaspModel
from sparse_coding.rasp.rasp_tokenizer import encode, decode


model = RaspModel()
model.eval()
input_tokens = model.haiku_model.input_encoder.encode(
    ["BOS", "w", "x", "y", "z"]
    )
outputs = []

for input_token in input_tokens:
    # call encode
    output = model(input_token).detach()
    # call decode
    outputs.append()

print(output)
