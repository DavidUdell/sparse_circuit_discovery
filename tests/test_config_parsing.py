"""Functions to test config parsing."""


from textwrap import dedent

from pytest import fixture
from transformers import AutoModelForCausalLM, PreTrainedModel

from sparse_coding.utils.interface import parse_slice, slice_to_seq


@fixture
def mock_slices():
    """Slice strings, their ground truths, and a model to test with."""

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    slice_strings: list[str] = [
        "0:1",
        "1:2",
        "0:2:1",
        "0::2",
        ":",
        "::",
        "0::",
        ":1",
        "2:",
        "::5",
        ":3:",
        "-1:",
        "0:9",
        "2::2",
        "0:5:-1",
        "::-1",
    ]
    slices: list[slice] = [
        slice(0, 1),
        slice(1, 2),
        slice(0, 2, 1),
        slice(0, None, 2),
        slice(None, None),
        slice(None, None),
        slice(0, None),
        slice(None, 1),
        slice(2, None),
        slice(None, None, 5),
        slice(None, 3),
        slice(-1, None),
        slice(0, 9),
        slice(2, None, 2),
        slice(0, 5, -1),
        slice(None, None, -1),
    ]
    # Pythia 70M has 6 layers.
    ranges: list[range] = [
        range(0, 1),
        range(1, 2),
        range(0, 2, 1),
        range(0, 6, 2),
        range(0, 6),
        range(0, 6),
        range(0, 6),
        range(0, 1),
        range(2, 6),
        range(0, 6, 5),
        range(0, 3),
        range(5, 6),
        range(0, 6),
        range(2, 6, 2),
        range(0, 5, -1),
        range(0, 6, -1),
    ]

    return model, slice_strings, slices, ranges


def test_parse_slice(mock_slices):  # pylint: disable=redefined-outer-name
    """Test parse_slice, especially for the weirder slice syntaxes."""

    _, slice_strings, slices, _ = mock_slices

    for slice_string, ground_truth in zip(slice_strings, slices):
        assert parse_slice(slice_string) == ground_truth, dedent(
            f"""
                Parsed slice {slice_string} should have been interpreted as
                {ground_truth}, not {parse_slice(slice_string)}
                """
        )


def test_slice_to_seq(mock_slices):  # pylint: disable=redefined-outer-name
    """Test slice_to_seq, especially for the weirder slice syntaxes."""

    model, _, slices, ranges = mock_slices

    for parsed_slice, model_range in zip(slices, ranges):
        assert slice_to_seq(model, parsed_slice) == model_range, dedent(
            f"""
                Parsed slice {parsed_slice} should have been interpreted as
                {model_range}, not {slice_to_seq(model, parsed_slice)}
                """
        )
