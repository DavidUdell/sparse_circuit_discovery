"""Looking over the rasp_to_torch model."""


from sparse_coding.rasp.rasp_to_torch import RaspModel
from sparse_coding.rasp.rasp_tokenizer import rasp_encode


model = RaspModel()
model.eval()

input_tokens = ["BOS", "x", "x", "y", "y", "y", "z", "z", "z"]

input_ids = rasp_encode(model, input_tokens)
output = model(input_ids).detach()
print(output.shape)

for i in range(9):
    print([output[i, :].sum(-1).item()])
