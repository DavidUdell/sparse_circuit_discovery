# %%
"""A constant-time approximation of the causal graphing algorithm."""

from collections import namedtuple

import torch as t


# %%
# Function and classes.
EffectOut = namedtuple(
    "EffectOut", ["effects", "deltas", "grads", "total_effect"]
)


def patch_act(
    clean,
    model,
    submodules,
    dictionaries,
    metric_fn,
    metric_kwargs,
):
    """Patch the activations of a model using its gradient."""

    hidden_states_clean = {}
    grads = {}

    # Trace the submodule activations and gradients.
    with model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            x_hat, f = dictionary(x, output_features=True)
            residual = x - x_hat

            hidden_states_clean[submodule] = {"act": f, "res": residual}
            grads[submodule] = hidden_states_clean[submodule].grad.save()

            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            submodule.output = x_recon
            x.grad = x_recon.grad

        metric_clean = metric_fn(model, **metric_kwargs).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}
    grads = {k: v.value for k, v in grads.items()}

    hidden_states_patch = {
        k: {"act": t.zeros_like(v.act), "res": t.zeros_like(v.res)}
        for k, v in hidden_states_clean.items()
    }
    total_effect = None

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = (
            hidden_states_patch[submodule],
            hidden_states_clean[submodule],
            grads[submodule],
        )
        delta = (
            patch_state - clean_state.detach()
            if patch_state is not None
            else -clean_state.detach()
        )
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, grads, total_effect)
