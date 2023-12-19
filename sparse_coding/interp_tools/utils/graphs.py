"""Graph the causal effects of ablations."""


from textwrap import dedent

from pygraphviz import AGraph

from sparse_coding.utils.interface import load_layer_feature_labels


def color_range_from_scalars(activations: dict) -> tuple[float, float]:
    """Get the range of values in the activations dict."""

    min_scalar = min([value.item() for value in activations.values()])
    max_scalar = max([value.item() for value in activations.values()])

    return min_scalar, max_scalar


def graph_causal_effects(
    activations: dict,
    model_dir: str,
    top_k_info_file: str,
    base_file: str,
    rasp=False,
) -> AGraph:
    """Graph the causal effects of ablations."""

    graph = AGraph(directed=True)

    if rasp:
        # Plot neuron nodes.
        for layer_idx, neuron_idx in activations.keys():
            graph.add_node(f"({layer_idx}.{neuron_idx})")

        # Plot effect edges.
        for (layer_idx, neuron_idx), effects_vector in activations.items():
            if layer_idx == 1:  # Effects are all downstream.
                for downstream_neuron_idx, effect in enumerate(effects_vector):
                    if effect.item() == 0:
                        continue
                    graph.add_edge(
                        f"0.{neuron_idx}",
                        f"1.{downstream_neuron_idx}",
                        label=str(effect.item()),
                    )

    else:

        def label_appendable(layer_idx, neuron_idx):
            return load_layer_feature_labels(
                model_dir,
                layer_idx,
                neuron_idx,
                top_k_info_file,
                base_file,
            )

        # Plot neuron nodes.
        for (
            ablation_layer_idx,
            ablated_dim,
            downstream_dim,
        ) in activations.keys():
            # I need to be saving the downstream layer index too. But this
            # works for now.
            graph.add_node(
                dedent(
                    f"""
                    {ablation_layer_idx}.{ablated_dim}:
                    {label_appendable(ablation_layer_idx, ablated_dim)}
                    """
                )
            )
            graph.add_node(
                dedent(
                    f"""
                    {ablation_layer_idx + 1}.{downstream_dim}:
                    {label_appendable(ablation_layer_idx + 1, downstream_dim)}
                    """
                )
            )

        min_scalar, max_scalar = color_range_from_scalars(activations)
        # Plot effect edges.
        for (
            ablation_layer_idx,
            ablated_dim,
            downstream_dim,
        ), effect in activations.items():
            if effect.item() == 0:
                continue
            # Blue means the intervention increased downstream firing, while
            # red means it decreased firing. Alpha indicates distance from 0.0
            # effect size.
            if effect.item() > 0.0:
                red = 0
                blue = 255
            elif effect.item() < 0.0:
                red = 255
                blue = 0
            alpha = int(255 *
                abs(effect.item()) / (
                max(abs(max_scalar), abs(min_scalar))
                )
            )
            rgba_color = f"#{red:02x}00{blue:02x}{alpha:02x}"

            graph.add_edge(
                dedent(
                    f"""
                    {ablation_layer_idx}.{ablated_dim}:
                    {label_appendable(ablation_layer_idx, ablated_dim)}
                    """
                ),
                dedent(
                    f"""
                    {ablation_layer_idx + 1}.{downstream_dim}:
                    {label_appendable(ablation_layer_idx + 1, downstream_dim)}
                    """
                ),
                color=rgba_color,
            )

        # Assert no repeat edges.
        edges = graph.edges()
        assert len(edges) == len(set(edges)), "Repeat edges in graph."

        # Remove unlinked nodes.
        for node in graph.nodes():
            if len(graph.edges(node)) == 0:
                graph.remove_node(node)
                print(f"Removed isolated neuron {node} from causal graph.\n")

    return graph
