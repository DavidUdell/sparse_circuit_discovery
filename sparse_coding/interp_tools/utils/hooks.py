"""Ablation and caching hooks."""


from contextlib import contextmanager

import torch as t


def rasp_ablations_hook_fac(neuron_index: int):
    """Factory for rasp ablations hooks, working at a neuron idx."""

    # All `transformer_lens` hook functions must have this interface.
    def ablations_hook(  # pylint: disable=unused-argument
        acts_tensor: t.Tensor, hook
    ) -> t.Tensor:
        """Zero out a particular neuron's activations."""

        acts_tensor[:, :, neuron_index] = 0.0

        return acts_tensor

    return ablations_hook


@contextmanager
def ablations_lifecycle(
    dim_idx: int,
    layer_idx: int,
    model,
    encoder: t.Tensor,
    biases: t.Tensor,
    cache: dict,
):
    """
    Context manager for the full-scale ablations and caching.

    Ablates the specified feature at `layer_idx`, and caches the downstream
    value at `layer_idx + 1` in `cache`. Be sure `layer_idx + 1` exists.
    """

    def encoder_hook_fac(dim_idx: int, encoder: t.Tensor, biases: t.Tensor):
        """Create hooks that zero a projected neuron and project it back."""

        def ablations_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Project activation vectors; ablate them; project them back."""

            # Project activations through the encoder/bias.
            projected_acts_unrec = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    input, encoder, bias=biases
                )
            )
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=True
            )
            # Zero out the activation at dim_idx.
            projected_acts[:, dim_idx] = 0.0
            # Project back to activation space.
            output = t.nn.functional.linear(  # pylint: disable=not-callable
                projected_acts,
                encoder.t(),
                bias=biases,
            )

        return ablations_hook

    def caching_hook_fac(
        dim_idx: int,
        layer_idx: int,
        encoder: t.Tensor,
        biases: t.Tensor,
        cache: dict,
    ):
        """Create hooks that cache the projected activations."""

        def caching_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Cache the projected activations at dim_idx."""

            # Project activations through the encoder/bias.
            projected_acts_unrec = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    input, encoder, bias=biases
                )
            )
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=True
            )
            # Cache the activations at dim_idx.
            cache[layer_idx][dim_idx] = projected_acts[:, dim_idx]

        return caching_hook

    encoder_hook_handle = model[layer_idx].register_forward_hook(
        encoder_hook_fac(dim_idx, encoder, biases)
    )
    caching_hook_handle = model[layer_idx + 1].register_forward_hook(
        caching_hook_fac(dim_idx, layer_idx + 1, encoder, biases, cache)
    )

    try:
        yield
    finally:
        encoder_hook_handle.remove()
        caching_hook_handle.remove()
