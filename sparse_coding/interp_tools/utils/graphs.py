"""Graph the causal effects of ablations."""


from pygraphviz import AGraph


def graph_causal_effects(activations: dict) -> AGraph:
    """Graph the causal effects of ablations."""

    graph = AGraph(directed=True)

    # Plot neuron nodes.
    for layer_idx, neuron_idx in activations.keys():
        graph.add_node(f"neuron_{layer_idx}.{neuron_idx}")

    # Plot effect edges.
    for (layer_idx, neuron_idx), effects_vector in activations.items():
        if layer_idx == 1:  # Effects are all downstream.
            for downstream_neuron_idx, effect in enumerate(effects_vector):
                if effect.item() == 0:
                    continue
                graph.add_edge(
                    f"neuron_0.{neuron_idx}",
                    f"neuron_1.{downstream_neuron_idx}",
                    label=str(effect.item())
                )

    return graph
