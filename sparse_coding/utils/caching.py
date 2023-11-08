"""Help cache data and layer tensors from many model layers."""


from textwrap import dedent


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

    return slice(start, stop, step)
