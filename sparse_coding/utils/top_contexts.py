"""Functions for processing autoencoders into top-k contexts."""


from collections import defaultdict

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer


def context_activations(
    context_token_ids: list[list[int]],
    context_acts: list[t.Tensor],
    encoder,
    tokenizer: AutoTokenizer,
) -> defaultdict[int, defaultdict[str, float]]:
    """Return the autoencoder's summed activations, at each feature dimension,
    at each input token."""

    contexts_and_activations = defaultdict(defaultdict_factory)
    assert len(context_token_ids) == len(context_acts)
    for context, activation in zip(context_token_ids, context_acts):
        context = tokenizer.convert_ids_to_tokens(context)
        context = "".join(context)
        context = context.replace("Ġ", " ")
        context = context.replace("\n", "\\n")

        for dim_idx in range(encoder.encoder_layer.weight.shape[0]):
            contexts_and_activations[dim_idx][context] = activation[:, dim_idx]

    return contexts_and_activations


def defaultdict_factory():
    """Factory for string defaultdicts."""

    return defaultdict(str)


def project_activations(
    acts_list: list[t.Tensor],
    projector,
    accelerator: Accelerator,
) -> list[t.Tensor]:
    """Projects the activations block over to the sparse latent space."""

    # Remember the original question lengths.
    lengths: list[int] = [len(question) for question in acts_list]

    flat_acts: t.Tensor = t.cat(acts_list, dim=0)
    flat_acts: t.Tensor = accelerator.prepare(flat_acts)
    projected_flat_acts: t.Tensor = projector(flat_acts).detach()

    # Reconstruct the original question lengths.
    projected_activations: list[t.Tensor] = []
    current_idx: int = 0
    for length in lengths:
        projected_activations.append(
            projected_flat_acts[current_idx : current_idx + length, :]
        )
        current_idx += length

    return projected_activations


def top_k_contexts(
    contexts_and_activations: defaultdict[int, defaultdict[str, t.Tensor]],
    top_k: int,
) -> defaultdict[int, list[tuple[str, t.Tensor]]]:
    """Select the top-k tokens for each feature."""
    top_k_contexts_acts = defaultdict(list)

    for dim_idx, contexts_acts in contexts_and_activations.items():
        ordered_contexts_acts: list[tuple[str, t.Tensor]] = sorted(
            contexts_acts.items(),
            key=lambda x: x[-1].sum().item(),
            reverse=True,
        )
        top_k_contexts_acts[dim_idx] = ordered_contexts_acts[:top_k]

    return top_k_contexts_acts


def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: list[list[int]]
) -> list[t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        try:
            original_length: int = len(unpadded_prompt)
            # From here on out, activations are unpadded, and so must be
            # packaged as a _list of tensors_ instead of as just a tensor
            # block.
            unpadded_activations.append(
                activations_block[k, :original_length, :]
            )
        except IndexError:
            print(f"IndexError at {k}")
            # This should only occur when the data collection was interrupted.
            # In that case, we just break when the data runs short.
            break

    return unpadded_activations
