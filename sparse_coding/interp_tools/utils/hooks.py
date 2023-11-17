"""Ablation and caching hooks, hook factories, and context managers."""


from contextlib import contextmanager

import torch as t


def ablations_hook_fac(neuron_index: int):
    """Factory for ablations hooks, working at a neuron idx."""

    # All `torch` hooks _must_ have this interface.
    def ablations_hook(  # pylint: disable=unused-argument, redefined-builtin
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Zero out a particular neuron's activations."""
        output[:, neuron_index] = 0.0

    return ablations_hook


def caching_hook_fac(
    neuron_index: int,
    layer_index: int,
    token: t.Tensor,
    activations_dict: dict,
):
    """Factory for caching_hooks that use an activations dict."""

    # Again, this hook interface is mandatory in `torch`.
    def caching_hook(  # pylint: disable=unused-argument, redefined-builtin
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Cache downstream layer activations."""
        activations_dict[
            (
                f"layer_{layer_index}",
                f"dim_{neuron_index}",
                f"token_id_{token}",
            )
        ] = (
            output.detach().sum().item()
        )

    return caching_hook


# %%
# Ablations context managers and factories.
@contextmanager
def ablations_lifecycle(
    torch_model: t.nn.Module,
    neuron_index: int,
    layer_index: int,
    token: t.Tensor,
    activations_dict: dict,
) -> None:
    """Define, register, and unregister ablation run hooks."""

    # Register the hooks with `torch`. Note that `attn_1` and `attn_2` are
    # hardcoded for the rasp model for now.
    ablations_hook_handle = torch_model.attn_1.register_forward_hook(
        ablations_hook_fac(neuron_index)
    )
    caching_hook_handle = torch_model.attn_2.register_forward_hook(
        caching_hook_fac(neuron_index, layer_index, token, activations_dict)
    )

    # Yield control to caller function.
    try:
        yield
    # Unregister the hooks.
    finally:
        ablations_hook_handle.remove()
        caching_hook_handle.remove()


@contextmanager
def base_caching_lifecycle(
    torch_model: t.nn.Module,
    neuron_index: int,
    layer_index: int,
    token: t.Tensor,
    activations_dict: dict,
) -> None:
    """Define, register, and unregister just the caching hook."""

    caching_hook_handle = torch_model.attn_2.register_forward_hook(
        caching_hook_fac(neuron_index, layer_index, token, activations_dict)
    )

    try:
        yield
    finally:
        caching_hook_handle.remove()
