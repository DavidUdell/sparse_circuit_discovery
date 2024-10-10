"""Computational utilities for plotting hook effects."""

from collections import defaultdict

import torch as t


def calc_act_diffs(
    ablated_activations: defaultdict,
    base_activations: defaultdict,
) -> dict[tuple[int, int, int], t.Tensor]:
    """
    Compute ablated effects minus base effects.

    The recursive defaultdict indices are:
    [ablation_layer_idx][ablated_dim_idx][downstream_dim].
    """

    act_diffs: dict[tuple[int, int, int], t.Tensor] = {}
    keys_dict = {}
    for i in ablated_activations:
        for j in ablated_activations[i]:
            for k in ablated_activations[i][j]:
                keys_dict[i, j, k] = None

    for i, j, k in keys_dict:
        act_diffs[i, j, k] = (
            ablated_activations[i][j][k][:, -1, :]
            - base_activations[i][None][k][:, -1, :]
        )
        assert act_diffs[i, j, k].shape == (1, 1)
        del ablated_activations[i][j][k]

    return act_diffs


def calc_overall_effects(
    act_diffs: dict[tuple[int, int, int], t.Tensor]
) -> float:
    """Compute the overall absolute effects in `act_diffs`."""

    overall_effects = 0.0
    for i, j, k in act_diffs:
        overall_effects += abs(act_diffs[i, j, k].item())

    assert overall_effects != 0.0, "Hook absolute effects sum to exactly 0.0."

    return overall_effects


class ExactlyZeroEffectError(ValueError):
    """Raised when logged absolute effects sum to exactly 0.0"""

    def __init__(self):
        message: str = "Total effect logged was exactly 0.0; exiting."
        super().__init__(message)


def deduplicate_sequences(
    contexts_and_acts: defaultdict[int, list[tuple[list[str], list[float]]]]
) -> defaultdict[int, list[tuple[list[str], list[float]]]]:
    """Deduplicate sequences when necessary."""

    deduplicated_contexts_and_acts = defaultdict(list)

    for dim_idx, contexts_acts in contexts_and_acts.items():
        for context, acts in contexts_acts:
            if (context, acts) not in deduplicated_contexts_and_acts[dim_idx]:
                deduplicated_contexts_and_acts[dim_idx].append((context, acts))

    return deduplicated_contexts_and_acts
