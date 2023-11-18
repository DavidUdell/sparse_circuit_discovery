"""Tokenization functionality for the rasp_to_torch model."""


import numpy as np
import torch as t

from sparse_coding.rasp.rasp_to_torch import RaspModel


def encode(model: RaspModel, tokens: list[str]):
    """Encodings from string tokens."""
    input_encodings = model.haiku_model.input_encoder.encode(tokens)
    input_encodings = t.tensor(input_encodings).unsqueeze(0)
    return input_encodings


def decode(model: RaspModel, logits: t.Tensor):
    """String tokens from logits."""
    normed_logits = logits.squeeze(0).argmax(-1)
    output_tokens = model.haiku_model.output_encoder.decode(
        normed_logits.tolist()
    )
    # Prepend BOS token
    output_tokens = ["BOS"] + output_tokens[1:]

    return output_tokens
