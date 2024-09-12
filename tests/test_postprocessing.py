"""Test directed graph postprocessing."""

import pygraphviz

from sparse_coding.interp_tools.utils.graphs import prune_graph


def test_prune_graph():
    """Test graph pruning."""

    # Create a simple graph.
    graph = pygraphviz.AGraph(strict=True, directed=True)
    graph.add_node("res_0.5555")
    graph.add_node("res_1.5555")
    graph.add_node("res_error_1.5555")
    graph.add_node("res_2.5555")
    graph.add_node("res_error_2.5555")

    graph.add_edge("res_0.5555", "res_1.5555")
    graph.add_edge("res_0.5555", "res_error_1.5555")
    graph.add_edge("res_error_1.5555", "res_2.5555")
    graph.add_edge("res_error_1.5555", "res_error_2.5555")

    # Prune the graph.
    pruned_graph = prune_graph(graph, final_layer_idx=2)

    # Check the pruned graph.
    assert pruned_graph.number_of_nodes() == 4
    assert pruned_graph.number_of_edges() == 3
    assert set(pruned_graph.nodes()) == {
        "res_0.5555",
        "res_error_1.5555",
        "res_2.5555",
        "res_error_2.5555",
    }
    assert set(pruned_graph.edges()) == {
        ("res_0.5555", "res_error_1.5555"),
        ("res_error_1.5555", "res_2.5555"),
        ("res_error_1.5555", "res_error_2.5555"),
    }
