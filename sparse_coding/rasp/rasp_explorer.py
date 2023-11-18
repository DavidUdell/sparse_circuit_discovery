"""Looking over the rasp_to_torch model."""


from sparse_coding.rasp.rasp_to_torch import RaspModel
from sparse_coding.rasp.rasp_tokenizer import rasp_encode, rasp_decode


model = RaspModel()
model.eval()

input_tokens = ["BOS", "w", "x", "y", "z"]

input_ids = rasp_encode(model, input_tokens)
output = model(input_ids).detach()
print(output)
output_tokens = rasp_decode(model, output)

print(output_tokens)
