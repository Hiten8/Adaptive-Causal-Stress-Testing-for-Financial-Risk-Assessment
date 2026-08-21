"""
Graph Fingerprint

Computes structural and statistical descriptors for a causal graph.
"""

import networkx as nx
import numpy as np

def compute_graph_fingerprint(
    graph,
    adjacency_matrix,
    edge_strength_matrix,
    regime,
    confidence,
    entropy
):
    """
    Compute a compact fingerprint describing the graph.
    """
    fingerprint = {}

    # Basic Graph Statistics
    fingerprint["num_nodes"] = graph.number_of_nodes()
    fingerprint["num_edges"] = graph.number_of_edges()
    fingerprint["density"] = nx.density(graph)

    # Degree Statistics
    in_degrees = [d for _, d in graph.in_degree()]
    out_degrees = [d for _, d in graph.out_degree()]

    fingerprint["avg_in_degree"] = float(np.mean(in_degrees))
    fingerprint["avg_out_degree"] = float(np.mean(out_degrees))
    fingerprint["max_in_degree"] = int(np.max(in_degrees))
    fingerprint["max_out_degree"] = int(np.max(out_degrees))

    # Connectivity
    undirected = graph.to_undirected()
    fingerprint["connected_components"] = (
        nx.number_connected_components(undirected)
    )
    fingerprint["clustering"] = (
        nx.average_clustering(undirected)
    )

    # Edge Statistics
    weights = []
    for _, _, data in graph.edges(data=True):
        weights.append(abs(data["weight"]))
    if len(weights) > 0:
        fingerprint["avg_edge_weight"] = float(np.mean(weights))
        fingerprint["max_edge_weight"] = float(np.max(weights))
        fingerprint["edge_weight_std"] = float(np.std(weights))
    else:
        fingerprint["avg_edge_weight"] = 0.0
        fingerprint["max_edge_weight"] = 0.0
        fingerprint["edge_weight_std"] = 0.0

    # Matrix Statistics
    fingerprint["adjacency_sparsity"] = float(
        np.mean(adjacency_matrix == 0)
    )
    fingerprint["mean_edge_strength"] = float(
        np.mean(np.abs(edge_strength_matrix))
    )

    # Regime Context
    fingerprint["dominant_regime"] = regime
    fingerprint["average_confidence"] = confidence
    fingerprint["average_entropy"] = entropy

    return fingerprint