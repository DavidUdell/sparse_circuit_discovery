"""Help cache data and layer tensors from many model layers."""


from textwrap import dedent

from transformers import PreTrainedModel


def parse_slice(slice_string: str) -> slice:
    """Parse a well-formed slice string into a proper slice."""

    start = stop = step = None

    slice_parts = slice_string.split(":")

    if not 0 <= len(slice_parts) <= 3:
        raise ValueError(
            dedent(
                f"""
                Slice string {slice_string} is not well-formed.
                """
            )
        )

    # Remember that Python evaluates empty strings as falsy.
    if slice_parts[0]:
        start = int(slice_parts[0])
    if len(slice_parts) > 1 and slice_parts[1]:
        stop = int(slice_parts[1])
    if len(slice_parts) == 3 and slice_parts[2]:
        step = int(slice_parts[2])

    layers_slice = slice(start, stop, step)
    assert start < stop, dedent(
        f"""
        Slice start ({layers_slice.start}) must be less than stop
        ({layers_slice.stop})
        """
    )

    return layers_slice


def validate_slice(model: PreTrainedModel, layers_slice: slice) -> None:
    """See whether the layers slice fits in the model's layers."""

    if layers_slice.stop is None:
        return

    # num_hidden_layers is not inclusive.
    last_layer: int = model.config.num_hidden_layers - 1

    # slice.stop is not inclusive.
    if last_layer < layers_slice.stop - 1:
        raise ValueError(
            dedent(
                f"""
                The layers slice {layers_slice} is out of bounds for the
                model's layer count.
                """
            )
        )

    return
