"""Test directed graph postprocessing."""

import pygraphviz

from sparse_coding.interp_tools.utils.graphs import prune_graph


def test_prune_graph():
    """Test graph pruning."""

    # Create a simple graph.
    graph = pygraphviz.AGraph(strict=True, directed=True)
    graph.add_node("A.0")
    graph.add_node("B.1")
    graph.add_node("C.1")
    graph.add_node("D.2")
    graph.add_node("E.2")

    graph.add_edge("A.0", "B.1")
    graph.add_edge("A.0", "C.1")
    graph.add_edge("C.1", "D.2")
    graph.add_edge("C.1", "E.2")

    # Prune the graph.
    pruned_graph = prune_graph(graph, final_layer_idx=2)

    # Check the pruned graph.
    assert pruned_graph.number_of_nodes() == 4
    assert pruned_graph.number_of_edges() == 3
    assert set(pruned_graph.nodes()) == {"A.0", "C.1", "D.2", "E.2"}
    assert set(pruned_graph.edges()) == {
        ("A.0", "C.1"),
        ("C.1", "D.2"),
        ("C.1", "E.2"),
    }
