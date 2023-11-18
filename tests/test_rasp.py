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

    for idx, input_token, ground_truth in zip(
        range(5),
        input_tokens,
        ("BOS", 0.0, 0.5, 0.33, 0.25)
        ):

        input_token = t.tensor(input_token, dtype=t.int).unsqueeze(0)
        output = model(input_token)
        output_tokens.append(output.sum().item())

        assert isinstance(output_tokens[-1], float), (
            f"Model output {output_tokens[-1]} must be a float."
        )
        if isinstance(ground_truth, float):
            assert t.isclose(
                t.tensor(output_tokens[-1]),
                t.tensor(ground_truth),
                atol=0.00001), (
                f"Model output (sequence index {idx}) {output_tokens[-1]} " \
                f"should be {ground_truth}."
                )
    output_tokens = model.haiku_model.input_encoder.decode(output_tokens)
    print(output_tokens)
