"""
Graph Similarity Metrics

Computes similarity between two CausalState objects.
"""

import numpy as np

# Edge Sets
def edge_set(graph):
    return set(graph.edges())

# Structural Hamming Distance
def structural_hamming_distance(state1, state2):
    e1 = edge_set(state1.graph)
    e2 = edge_set(state2.graph)

    return len(e1.symmetric_difference(e2))

# Jaccard Similarity
def jaccard_similarity(state1, state2):
    e1 = edge_set(state1.graph)
    e2 = edge_set(state2.graph)
    union = e1.union(e2)

    if len(union) == 0:
        return 1.0

    intersection = e1.intersection(e2)

    return len(intersection) / len(union)

# Added / Removed Edges
def edge_changes(state1, state2):
    e1 = edge_set(state1.graph)
    e2 = edge_set(state2.graph)
    added = e2 - e1
    removed = e1 - e2

    return len(added), len(removed)

# Edge Weight Difference
def edge_weight_difference(state1, state2):
    common = edge_set(state1.graph).intersection(
        edge_set(state2.graph)
    )

    if len(common) == 0:
        return 0.0, 0.0

    differences = []

    for edge in common:
        w1 = state1.graph.edges[edge]["weight"]
        w2 = state2.graph.edges[edge]["weight"]
        differences.append(abs(w1 - w2))

    return (
        float(np.mean(differences)),
        float(np.max(differences))
    )

# Fingerprint Difference
def fingerprint_difference(state1, state2):
    fp1 = state1.graph_fingerprint
    fp2 = state2.graph_fingerprint

    return {
        "density_difference":
            abs(
                fp1["density"]
                -
                fp2["density"]
            ),

        "avg_in_degree_difference":
            abs(
                fp1["avg_in_degree"]
                -
                fp2["avg_in_degree"]
            ),

        "avg_out_degree_difference":
            abs(
                fp1["avg_out_degree"]
                -
                fp2["avg_out_degree"]
            ),

        "clustering_difference":
            abs(
                fp1["clustering"]
                -
                fp2["clustering"]
            ),

        "edge_weight_difference":
            abs(
                fp1["avg_edge_weight"]
                -
                fp2["avg_edge_weight"]
            ),

        "sparsity_difference":
            abs(
                fp1["adjacency_sparsity"]
                -
                fp2["adjacency_sparsity"]
            )
    }

# Main Similarity Function
def compare_states(stored_state, current_state):
    shd = structural_hamming_distance(
        stored_state,
        current_state
    )

    jaccard = jaccard_similarity(
        stored_state,
        current_state
    )

    added, removed = edge_changes(
        stored_state,
        current_state
    )

    mean_edge_change, max_edge_change = (
        edge_weight_difference(
            stored_state,
            current_state
        )
    )

    fingerprint = fingerprint_difference(
        stored_state,
        current_state
    )

    return {
        "SHD": shd,

        "Jaccard": jaccard,

        "AddedEdges": added,

        "RemovedEdges": removed,

        "MeanEdgeWeightDifference":
            mean_edge_change,

        "MaxEdgeWeightDifference":
            max_edge_change,

        **fingerprint
    }