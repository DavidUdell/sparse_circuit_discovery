"""Graph the causal effects of ablations."""


from textwrap import dedent

import torch as t
import wandb
from pygraphviz import AGraph

from sparse_coding.utils.interface import (
    load_layer_feature_labels,
    load_preexisting_graph,
    sanitize_model_name,
    save_paths,
)


def graph_and_log(
        act_diffs: dict[tuple[int, int, int], t.Tensor],
        layer_range: range,
        layer_dim_indices: dict[int, list[int]],
        branching_factor: float,
        model_dir: str,
        graph_file: str,
        graph_dot_file: str,
        top_k_info_file: str,
        overall_effects: float,
        base_file: str,
):
    """Graph and log the causal effects of ablations."""

    # All other effect items are t.Tensors, but wandb plays nicer with floats.
    diffs_table = wandb.Table(columns=["Ablated Dim->Cached Dim", "Effect"])
    for i, j, k in act_diffs:
        key: str = f"{i}.{j}->{i+1}.{k}"
        value: float = act_diffs[i, j, k].item()
        diffs_table.add_data(key, value)
    wandb.log({"Effects": diffs_table})

    plotted_diffs = {}
    if branching_factor is not None:
        # Keep only the top effects per ablation site i, j across all
        # downstream indices k.
        working_dict = {}

        for address, effect in act_diffs.items():
            ablation_site = address[:2]

            # Avoids a defaultdict.
            if ablation_site not in working_dict:
                working_dict[ablation_site] = []

            working_dict[ablation_site].append((address, effect))

        for ablation_site, items in working_dict.items():
            sorted_items = sorted(
                items,
                key=lambda x: abs(x[-1].item()),
                reverse=True,
            )
            for (i, j, k), v in sorted_items[:branching_factor]:
                if i == layer_range[0]:
                    plotted_diffs[i, j, k] = v
                # Only plot effects that are downstream of immediately prior
                # ablation sites.
                elif j in layer_dim_indices[i-1]:
                    plotted_diffs[i, j, k] = v

    else:
        plotted_diffs = act_diffs

    save_plot_path: str = save_paths(
        base_file,
        f"{sanitize_model_name(model_dir)}/{graph_file}",
    )
    save_dot_path: str = save_paths(
        base_file,
        f"{sanitize_model_name(model_dir)}/{graph_dot_file}",
    )

    graph = graph_causal_effects(
        plotted_diffs,
        model_dir,
        top_k_info_file,
        graph_dot_file,
        overall_effects,
        base_file,
    )

    # Save the graph .svg.
    graph.draw(
        save_plot_path,
        format="svg",
        prog="dot",
    )

    # Read the .svg into a `wandb` artifact.
    artifact = wandb.Artifact(
        "feature_graph",
        type="directed_graph"
    )
    artifact.add_file(save_plot_path)
    wandb.log_artifact(artifact)

    # Save the AGraph object as a DOT file.
    graph.write(save_dot_path)


def color_range_from_scalars(activations: dict) -> tuple[float, float]:
    """Get the range of values in the activations dict."""

    min_scalar = min([value.item() for value in activations.values()])
    max_scalar = max([value.item() for value in activations.values()])

    return min_scalar, max_scalar


def graph_causal_effects(
    activations: dict[tuple, t.Tensor],
    model_dir: str,
    top_k_info_file: str,
    graph_dot_file: str,
    overall_effects: float,
    base_file: str,
    rasp=False,
) -> AGraph:
    """Graph the causal effects of ablations."""

    # Load preexistin graph, if applicable.
    graph = load_preexisting_graph(
        model_dir,
        graph_dot_file,
        base_file
    )
    if graph is None:
        graph = AGraph(directed=True)
    assert graph is not None, "Graph is None."

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

        return graph

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
    plotted_effects: float = 0.0
    for (
        ablation_layer_idx,
        ablated_dim,
        downstream_dim,
    ), effect in activations.items():
        if effect.item() == 0:
            continue
        plotted_effects += abs(effect.item())
        # Blue means the intervention increased downstream firing, while
        # red means it decreased firing. Alpha indicates distance from 0.0
        # effect size.
        if effect.item() > 0.0:
            red = 0
            blue = 255
        elif effect.item() < 0.0:
            red = 255
            blue = 0
        alpha = int(
            255
            * abs(effect.item())
            / (max(abs(max_scalar), abs(min_scalar)))
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

    # Add an effects fraction excluded node.
    excluded_fraction = round(
        (overall_effects - plotted_effects) / overall_effects, 2
    )
    overall_effects = round(overall_effects, 2)
    graph.add_node(
        f"Fraction of effects not plotted: {excluded_fraction}%."
    )
    graph.add_edge(
        dedent(
            f"""
            {ablation_layer_idx}.{ablated_dim}:
            {label_appendable(ablation_layer_idx, ablated_dim)}
            """
        ),
        f"Fraction of all effects not plotted: {excluded_fraction*100}%.",
    )

    # Assert no repeat edges.
    edges = graph.edges()
    assert len(edges) == len(set(edges)), "Repeat edges in graph."

    # Remove unlinked nodes.
    unlinked_nodes = 0
    for node in graph.nodes():
        if len(graph.edges(node)) == 0:
            graph.remove_node(node)
            unlinked_nodes += 1
    print(
        dedent(
            f"""
            Dropped {unlinked_nodes} unlinked neuron(s) from directed
            graph.\n
            """
        )
    )

    return graph
