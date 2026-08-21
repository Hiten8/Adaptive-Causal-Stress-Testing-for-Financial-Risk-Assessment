"""
Graph Change Detector

Purpose
-------
Detect PRIMARY structural changes between two consecutive
NetworkX causal graphs.

This module DOES NOT decide whether a graph should be updated.

It only identifies what changed.
"""

import networkx as nx

# Main Function
def detect_graph_changes(
    previous_graph: nx.DiGraph,
    current_graph: nx.DiGraph,
    weight_threshold: float = 0.05
):
    """
    Compare two NetworkX DiGraphs.

    Returns
    -------
    dict
    """

    # Edge Sets
    previous_edges = set(previous_graph.edges())
    current_edges = set(current_graph.edges())

    # Added / Removed
    added_edges = list(current_edges - previous_edges)
    removed_edges = list(previous_edges - current_edges)

    # Weight Changes
    weight_changed_edges = []
    common_edges = previous_edges.intersection(current_edges)

    for edge in common_edges:
        u, v = edge
        old_weight = previous_graph[u][v].get("weight", 1.0)
        new_weight = current_graph[u][v].get("weight", 1.0)

        if abs(new_weight - old_weight) >= weight_threshold:
            weight_changed_edges.append({
                "edge": edge,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "difference": new_weight - old_weight
            })

    # Isolated Nodes
    isolated_nodes = []
    for node in current_graph.nodes():
        if current_graph.degree(node) == 0:
            isolated_nodes.append(node)

    # Summary
    primary_change_count = (
        len(added_edges)
        + len(removed_edges)
        + len(weight_changed_edges)
    )

    return {
        "AddedEdges": added_edges,
        "RemovedEdges": removed_edges,
        "WeightChangedEdges": weight_changed_edges,
        "IsolatedNodes": isolated_nodes,
        "PrimaryChangeCount": primary_change_count
    }

# Utility
def has_structural_change(change_dict):
    return change_dict["PrimaryChangeCount"] > 0

def print_change_summary(change_dict):
    print("=" * 70)
    print("Primary Structural Changes")
    print("=" * 70)
    print()

    print(f"Added Edges        : {len(change_dict['AddedEdges'])}")
    print(f"Removed Edges      : {len(change_dict['RemovedEdges'])}")
    print(f"Weight Changes     : {len(change_dict['WeightChangedEdges'])}")
    print(f"Isolated Nodes     : {len(change_dict['IsolatedNodes'])}")
    print(f"Total Changes      : {change_dict['PrimaryChangeCount']}")
    print()

    if change_dict["AddedEdges"]:
        print("Added Edges")
        for u, v in change_dict["AddedEdges"]:
            print(f"   + {u} -> {v}")

        print()

    if change_dict["RemovedEdges"]:
        print("Removed Edges")
        for u, v in change_dict["RemovedEdges"]:
            print(f"   - {u} -> {v}")

        print()

    if change_dict["WeightChangedEdges"]:
        print("Weight Changes")
        for edge in change_dict["WeightChangedEdges"]:
            u, v = edge["edge"]
            print(
                f"   * {u} -> {v}"
                f" : "
                f"{edge['old_weight']:.3f}"
                f" -> "
                f"{edge['new_weight']:.3f}"
            )

        print()

    if change_dict["IsolatedNodes"]:
        print("Isolated Nodes")
        for node in change_dict["IsolatedNodes"]:
            print(f"   • {node}")

        print()
    print("=" * 70)