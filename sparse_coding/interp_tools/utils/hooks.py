"""Ablation and caching hooks."""


import torch as t


def ablations_hook_fac(neuron_index: int):
    """Factory for ablations hooks, working at a neuron idx."""

    # All `transformer_lens` hook functions must have this interface.
    def ablations_hook(acts_tensor: t.Tensor, hook) -> t.Tensor:  # pylint: disable=unused-argument
        """Zero out a particular neuron's activations."""

        print(f"acts tensor shape: {acts_tensor.shape}")
        acts_tensor[:, :, neuron_index] = 0.0

        return acts_tensor

    return ablations_hook
