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
    meaningful_dims: list[int],
    layer_idx: int,
    layer_range: range,
    model,
    encoder: t.Tensor,
    biases: t.Tensor,
    cache: dict,
):
    """
    Context manager for the full-scale ablations and caching.

    Ablates the specified feature at `layer_idx` and caches the downstream
    effects.
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
                    input[0],
                    encoder.to(model.device),
                    bias=biases.to(model.device),
                )
            )
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=True
            )
            # Zero out the activation at dim_idx.
            projected_acts[:, :, dim_idx] = 0.0
            # Project back to activation space.
            output = projected_acts.to(model.device) - biases.to(model.device)
            output = t.einsum(
                "bij, jk -> bik",
                projected_acts.to(model.device),
                encoder.to(model.device),
            )

        return ablations_hook

    def caching_hook_fac(
        ablated_dim_idx: int,
        meaningful_dims: list[int],
        layer_idx: int,
        encoder: t.Tensor,
        biases: t.Tensor,
        cache: dict,
    ):
        """Create hooks that cache the projected activations."""

        def caching_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Cache projected activations."""

            # Project activations through the encoder/bias.
            projected_acts_unrec = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    input, encoder, bias=biases
                )
            )
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=True
            )
            # Cache the activations.
            for downstream_dim in meaningful_dims:
                cache[
                    layer_idx][ablated_dim_idx][downstream_dim
                ] = projected_acts[:, :, downstream_dim]

        return caching_hook

    if layer_idx == layer_range[-1]:
        raise ValueError("Cannot ablate and cache from the last layer.")

    downstream_range: range = range(layer_idx + 1, layer_range[-1])
    # Pythia layer syntax, for now.
    encoder_hook_handle = model.gpt_neox.layers[
        layer_idx
    ].register_forward_hook(encoder_hook_fac(dim_idx, encoder, biases))

    caching_hook_handles = {}
    for index, layer in enumerate(downstream_range):
        caching_hook_handles[index] = model.gpt_neox.layers[
            layer
        ].register_forward_hook(
            caching_hook_fac(
                dim_idx,
                meaningful_dims,
                layer,
                encoder,
                biases,
                cache
            )
        )

    try:
        yield

    finally:
        encoder_hook_handle.remove()
        for handle in caching_hook_handles.items():
            handle.remove()
