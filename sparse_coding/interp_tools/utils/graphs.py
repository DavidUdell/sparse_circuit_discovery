"""Graph the causal effects of ablations."""


from pygraphviz import AGraph


def graph_causal_effects(activations: dict) -> AGraph:
    """Graph the causal effects of ablations."""

    graph = AGraph(directed=True)

    # Plot nodes.
    for layer_index, neuron_idx in activations.keys():
        graph.add_node(f"{layer_index}_{neuron_idx}")

    # Plot edges.
    graph.add_edges_from(
        [
            (f"{layer_index}_{neuron_idx}", f"{int(layer_index.split('_')[-1]) + 1}_{neuron_idx}")
            for layer_index, neuron_idx in activations.keys()
            if int(layer_index.split("_")[-1]) + 1 < 2
        ],
        label=str(activations[layer_index, neuron_idx])
    )

    return graph
