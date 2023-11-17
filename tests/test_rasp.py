"""Tests for the rasp model functionality."""


import torch as t

from sparse_coding.rasp.rasp_to_torch import RaspModel


# Test determinism.
t.manual_seed(0)


def test_rasp_model_inference():
    """Test forward passes on the torchified rasp model."""

    model = RaspModel()
    model.eval()
    input_tokens = model.haiku_model.input_encoder.encode(
        ["BOS", "w", "x", "y", "z"]
        )
    output_tokens = []

    for idx, input_token, ground_truth in zip(range(5), input_tokens, (0, 0, 1, 0, 0)):
        input_token = t.tensor(input_token, dtype=t.int).unsqueeze(0)
        output = model(input_token)
        output_tokens.append(output.sum().item())

        assert isinstance(output_tokens[-1], float), (
            f"Model output {output_tokens[-1]} must be a float."
        )
        assert output_tokens[-1] == ground_truth, (
            f"Model output (index {idx}) {output_tokens[-1]} should be {ground_truth}."
        )

    output_tokens = model.haiku_model.input_encoder.decode(output_tokens)
    print(output_tokens)
