"""Ablation and caching hooks."""


from collections import defaultdict
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
def hooks_lifecycle(
    ablation_dim_idx: int,
    to_cache_dims: dict[int, list[int]],
    ablation_layer_idx: int,
    full_layer_range: range,
    model,
    encoder: t.Tensor,
    biases: t.Tensor,
    cache: defaultdict,
    ablations_mode: bool = True,
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
                projected_acts_unrec, inplace=False
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
            output = (output,)

            assert output[0].shape == input[0].shape, (
                f"Ablations hook output shape {output[0].shape} does not match"
                f"input[0] shape {input[0].shape}."
            )
            assert output[0].sum() != input[0].sum(), (
                f"Ablation had no effect; output {output[0]} == {input[0]}"
            )

        return ablations_hook

    def caching_hook_fac(
        ablated_dim_idx: int,
        cache_dims: list[int],
        ablation_layer_idx: int,
        encoder: t.Tensor,
        biases: t.Tensor,
        cache: defaultdict,
    ):
        """Create hooks that cache the projected activations."""

        def caching_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Cache projected activations."""

            # Project activations through the encoder/bias.
            projected_acts_unrec = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    input[0],
                    encoder.to(model.device),
                    bias=biases.to(model.device),
                )
            )
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=False
            )
            # Cache the activations.
            for downstream_dim in cache_dims:
                cache[
                    ablation_layer_idx][ablated_dim_idx][downstream_dim
                ] = projected_acts[:, :, downstream_dim].detach().cpu()

                assert output[0].shape == input[0].shape, (
                f"Output shape {output.shape} does not match input[0] shape "
                f"{input[0].shape}."
            )


        return caching_hook

    if ablation_layer_idx == full_layer_range[-1]:
        raise ValueError("Cannot ablate and cache from the last layer.")

    downstream_range: range = range(
        ablation_layer_idx + 1, full_layer_range[-1] + 1
    )
    # Just Pythia layer syntax, for now.
    if ablations_mode:
        encoder_hook_handle = model.gpt_neox.layers[
            ablation_layer_idx
        ].register_forward_hook(
            encoder_hook_fac(ablation_dim_idx, encoder, biases)
        )

    caching_hook_handles = {}
    for index, layer in enumerate(downstream_range):
        caching_hook_handles[index] = model.gpt_neox.layers[
            layer
        ].register_forward_hook(
            caching_hook_fac(
                ablation_dim_idx,
                to_cache_dims[layer],
                ablation_layer_idx,
                encoder,
                biases,
                cache
            )
        )

    try:
        yield
    finally:
        if ablations_mode:
            encoder_hook_handle.remove()
        for handle in caching_hook_handles.values():
            handle.remove()
