"""Graph the causal effects of ablations."""

import html
from collections import defaultdict
from copy import copy
from textwrap import dedent

import requests
from tqdm.auto import tqdm
import torch as t
from pygraphviz import AGraph
import wandb

from sparse_coding.utils.top_contexts import top_k_contexts
from sparse_coding.interp_tools.utils.computations import (
    calc_overall_effects,
    deduplicate_sequences,
)
from sparse_coding.utils.interface import (
    load_layer_feature_labels,
    load_preexisting_graph,
    sanitize_model_name,
    save_paths,
)


def graph_and_log(
    act_diffs: dict[tuple[int, int, int], t.Tensor],
    keepers: dict[tuple[int, int], int],
    branching_factor: int,
    model_dir: str,
    graph_file: str,
    graph_dot_file: str,
    top_k_info_file: str,
    threshold: float,
    logit_tokens: int,
    tokenizer,
    prob_diffs,
    base_file: str,
):
    """Graph and log the causal effects of ablations."""

    # Asserts that there was any overall effect before proceeding.
    overall_effects: float = calc_overall_effects(act_diffs)

    plotted_diffs = {}
    if branching_factor is not None:
        for i, j in keepers:
            for k in keepers[i, j]:
                plotted_diffs[i, j, k] = act_diffs[i, j, k]

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
        threshold,
        logit_tokens,
        tokenizer,
        prob_diffs,
        base_file,
    )

    # Save the graph .svg.
    graph.draw(
        save_plot_path,
        format="svg",
        prog="dot",
    )

    # Read the .svg into a `wandb` artifact.
    artifact = wandb.Artifact("cognition_graph", type="directed_graph")
    artifact.add_file(save_plot_path)
    wandb.log_artifact(artifact)

    # Save the AGraph object as a DOT file.
    graph.write(save_dot_path)

    print(f"Graph saved to {save_plot_path}")


def color_range_from_scalars(activations: dict) -> tuple[float, float]:
    """Get the range of values in the activations dict."""

    min_scalar = min([value.item() for value in activations.values()])
    max_scalar = max([value.item() for value in activations.values()])

    return min_scalar, max_scalar


def label_highlighting(
    layer_idx,
    neuron_idx,
    model_dir,
    top_k_info_file,
    logit_tokens: int,
    tokenizer,
    address: str,
    prob_diffs,
    base_file,
    neuronpedia: bool = False,
    sublayer_type: str = None,
    top_k: int = None,
    view: int = None,
    neuronpedia_key: str = None,
) -> str:
    """Highlight contexts using cached activation data."""

    contexts, acts = load_layer_feature_labels(
        model_dir,
        layer_idx,
        neuron_idx,
        top_k_info_file,
        base_file,
    )
    label = '<<table border="0" cellborder="0" cellspacing="0">'
    label += f'<tr><td><font point-size="16"><b>{address}</b></font></td></tr>'
    for context, act in zip(contexts, acts):
        label += "<tr>"

        max_a = max(act)
        context = tokenizer.convert_ids_to_tokens(
            context,
        )

        for token, act in zip(context, act):
            token = tokenizer.convert_tokens_to_string([token])
            token = html.escape(token)
            # Explicitly handle newlines/control characters.
            token = token.encode("unicode_escape").decode("utf-8")

            if act <= 0.0:
                label += f'<td bgcolor="#ffffff">{token}</td>'

            else:
                blue_prop = act / max_a
                rg_prop = 1.0 - blue_prop

                rg_shade = f"{int(96 + (159*rg_prop)):02x}"
                b_shade = f"{255:02x}"
                shade = f"#{rg_shade}{rg_shade}{b_shade}"
                cell_tag = f'<td bgcolor="{shade}">'
                label += f"{cell_tag}{token}</td>"

        label += "</tr>"

    # Add logit diffs.
    if (layer_idx, neuron_idx) in prob_diffs:
        label += "<tr>"
        pos_tokens_affected = (
            prob_diffs[layer_idx, neuron_idx]
            .sum(dim=0)
            .squeeze()
            .topk(logit_tokens)
            .indices
        )
        # Negative prob_diffs here to get top tokens negatively affected.
        neg_tokens_affected = (
            (-prob_diffs[layer_idx, neuron_idx])
            .sum(dim=0)
            .squeeze()
            .topk(logit_tokens)
            .indices
        )
        for meta_idx, token in enumerate(
            t.cat((pos_tokens_affected, neg_tokens_affected))
        ):
            # Break rows between positive and negative logits.
            if meta_idx == len(pos_tokens_affected):
                label += "</tr><tr>"
            if (
                prob_diffs[layer_idx, neuron_idx][:, token].sum(dim=0).item()
                > 0.0
            ):
                shade = "#6060ff"
                cell_tag = f'<td border="1" bgcolor="{shade}">'
            elif (
                prob_diffs[layer_idx, neuron_idx][:, token].sum(dim=0).item()
                < 0.0
            ):
                shade = "#ff6060"
                cell_tag = f'<td border="1" bgcolor="{shade}">'
            else:
                # Grey for no effect, to disabmiguate from any errors.
                cell_tag = '<td border="1" bgcolor="#cccccc">'

            token = tokenizer.convert_ids_to_tokens(token.item())
            token = tokenizer.convert_tokens_to_string([token])
            token = html.escape(token)
            # Explicitly handle newlines/control characters.
            token = token.encode("unicode_escape").decode("utf-8")

            label += f"{cell_tag}{token}</td>"
        label += "</tr>"

    if neuronpedia:
        assert (
            sublayer_type is not None
            and top_k is not None
            and view is not None
        )

        label += neuronpedia_api(
            layer_idx,
            neuron_idx,
            neuronpedia_key,
            sublayer_type,
            top_k,
            view,
        )

    label += "</table>>"

    return label


def graph_causal_effects(
    activations: dict[tuple, t.Tensor],
    model_dir: str,
    top_k_info_file: str,
    graph_dot_file: str,
    overall_effects: float,
    threshold: float,
    logit_tokens: int,
    tokenizer,
    prob_diffs,
    base_file: str,
    rasp=False,
) -> AGraph:
    """Graph the causal effects of ablations."""

    # Load preexisting graph, if applicable.
    graph = load_preexisting_graph(model_dir, graph_dot_file, base_file)
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

    # Plot neuron nodes.
    for (
        ablation_layer_idx,
        ablated_dim,
        downstream_dim,
    ) in tqdm(activations.keys(), desc="Edges Plotted Progress"):
        graph.add_node(
            f"{ablation_layer_idx}.{ablated_dim}",
            label=label_highlighting(
                ablation_layer_idx,
                ablated_dim,
                model_dir,
                top_k_info_file,
                logit_tokens,
                tokenizer,
                f"{ablation_layer_idx}.{ablated_dim}",
                prob_diffs,
                base_file,
            ),
            shape="box",
        )
        graph.add_node(
            f"{ablation_layer_idx + 1}.{downstream_dim}",
            label=label_highlighting(
                ablation_layer_idx + 1,
                downstream_dim,
                model_dir,
                top_k_info_file,
                logit_tokens,
                tokenizer,
                f"{ablation_layer_idx + 1}.{downstream_dim}",
                prob_diffs,
                base_file,
            ),
            shape="box",
        )

    min_scalar, max_scalar = color_range_from_scalars(activations)

    # Plot effect edges.
    plotted_effects: float = 0.0
    minor_effects: int = 0
    for (
        ablation_layer_idx,
        ablated_dim,
        downstream_dim,
    ), effect in activations.items():
        if abs(effect.item()) <= threshold or 0.0 == effect.item():
            minor_effects += 1
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
        else:
            raise ValueError("Should be unreachable.")

        alpha = int(
            255 * abs(effect.item()) / (max(abs(max_scalar), abs(min_scalar)))
        )
        rgba_color = f"#{red:02x}00{blue:02x}{alpha:02x}"

        graph.add_edge(
            f"{ablation_layer_idx}.{ablated_dim}",
            f"{ablation_layer_idx + 1}.{downstream_dim}",
            color=rgba_color,
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

    included_fraction = round(plotted_effects / overall_effects, 2)
    # This must come after dropping nodes.
    graph.add_node(
        f"Effects plotted out of collected: ~{included_fraction*100}%."
    )

    print(
        dedent(
            f"""
            Dropped {unlinked_nodes} unlinked neuron(s) from graph.
            {minor_effects} minor effect(s) ignored.\n
            """
        )
    )

    return graph


def prune_graph(
    graph: AGraph,
    leaf_nodes: list | None = None,
    final_layer_idx: int = 11,
):
    """
    Filter a directed graph down to its source-to-sink subgraph.

    The current default `final_layer_idx` presumes GPT-2-small.
    """

    # `leaf_nodes` starts off with all nodes without outgoing edges.
    if leaf_nodes is None:
        leaf_nodes: list = []
        for node in graph.nodes():
            if not graph.out_degree(node):
                leaf_nodes.append(node)

    # Prune out all the non-final-layer leaf nodes and append upstream relevant
    # nodes.
    for node in copy(leaf_nodes):
        layer_idx: int = int((node.split(".")[0]).split("_")[-1])
        if layer_idx != final_layer_idx:
            upstream_nodes = graph.predecessors(node)
            graph.remove_node(node)
            upstream_childless_nodes = [
                node for node in upstream_nodes if not graph.out_degree(node)
            ]

            leaf_nodes.extend(upstream_childless_nodes)

        leaf_nodes.remove(node)

    # Recurse if necessary.
    if leaf_nodes:
        leaf_nodes: list = list(set(leaf_nodes))
        graph = prune_graph(graph, leaf_nodes)

    return graph


def neuronpedia_api(
    layer_idx: int,
    dim_idx: int,
    neuronpedia_key: str,
    sublayer_type: str,
    top_k: int,
    view: int,
) -> str:
    """
    Pulls down Neuronpedia API annotations for given graph nodes.
    """

    url_prefix: str = "https://www.neuronpedia.org/api/feature/gpt2-small/"
    url_post_res: str = "-res-jb/"
    url_post_attn: str = "-att_128k-oai/"
    url_post_mlp: str = "-mlp_128k-oai/"

    # sublayer_type: str = "res" | "attn" | "mlp"
    if sublayer_type == "res":
        url_post: str = url_post_res
    elif sublayer_type == "attn":
        url_post: str = url_post_attn
    elif sublayer_type == "mlp":
        url_post: str = url_post_mlp
    else:
        raise ValueError("Sublayer type not recognized:", sublayer_type)

    url: str = url_prefix + str(layer_idx) + url_post + str(dim_idx)

    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "max-age=0",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "sparse_circuit_discovery",
            "X-Api-Key": neuronpedia_key,
        },
        timeout=300,
    )

    assert (
        response.status_code != 404
    ), "Neuronpedia API connection failed: 404"

    neuronpedia_dict: dict = response.json()
    data: list[dict] = neuronpedia_dict["activations"]

    label: str = ""

    # defaultdict[int, list[tuple[list[str], list[float]]]]
    contexts_and_activations = defaultdict(list)
    for seq_dict in data:
        tokens: list[str] = seq_dict["tokens"]
        values: list[float | int] = seq_dict["values"]

        contexts_and_activations[dim_idx].append((tokens, values))

    top_contexts = top_k_contexts(contexts_and_activations, view, top_k)
    top_contexts = deduplicate_sequences(top_contexts)

    for context, acts in top_contexts[dim_idx]:
        if not context:
            continue

        max_a: int | float = max(acts)
        label += "<tr>"
        # It is known that the context is not empty by here.
        for token, act in zip(context, acts):
            token = html.escape(token)
            token = token.encode("unicode_escape").decode("utf-8")

            if act <= 0.0:
                label += f'<td bgcolor="#ffffff">{token}</td>'
            else:
                blue_prop = act / max_a
                rg_prop = 1.0 - blue_prop

                rg_shade = f"{int(96 + (159*rg_prop)):02x}"
                b_shade = f"{255:02x}"
                shade = f"#{rg_shade}{rg_shade}{b_shade}"
                cell_tag = f'<td bgcolor="{shade}">'
                label += f"{cell_tag}{token}</td>"
        label += "</tr>"

    return label
