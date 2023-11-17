"""Graph the causal effects of ablations."""


from pygraphviz import AGraph


def graph_causal_effects(activations: dict) -> AGraph:
    """Graph the causal effects of ablations."""

    graph = AGraph(directed=True)

    # Plot nodes.
    for layer_index, neuron_idx, token in activations.keys():
        graph.add_node(f"{layer_index}_{neuron_idx}_{token}")

    # Plot edges.
    # TODO

    return graph
