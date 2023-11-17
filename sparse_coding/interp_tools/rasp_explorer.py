"""Looking over the rasp_to_torch model."""


import torch as t

from sparse_coding.rasp.rasp_to_torch import RaspModel


model = RaspModel()
model.eval()
input_tokens = model.haiku_model.input_encoder.encode(
    ["BOS", "w", "x", "y", "z"]
    )
output_tokens = []

for input_token in input_tokens:
    input_token = t.tensor(input_token).unsqueeze(0)
    output = model(input_token)
    output_tokens.append(output.sum().item())

print(output_tokens)
