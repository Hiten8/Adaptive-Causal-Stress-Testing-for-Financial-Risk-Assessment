"""
Feature Extraction for Change Gate

Creates a feature vector from two consecutive CausalStates.
"""

import numpy as np
from src.change_gate.graph_similarity import compare_states

# Pseudo Label Thresholds
SHD_THRESHOLD = 5
CONFIDENCE_DROP = 0.15
ENTROPY_INCREASE = 0.20

# Feature Extraction
def extract_features(stored_state, current_state):
    similarity = compare_states(
        stored_state,
        current_state
    )

    # Regime Features
    regime_changed = int(
        stored_state.dominant_regime !=
        current_state.dominant_regime
    )

    confidence_difference = (
        current_state.average_confidence -
        stored_state.average_confidence
    )

    entropy_difference = (
        current_state.average_entropy -
        stored_state.average_entropy
    )

    # Fingerprint Features
    fp_prev = stored_state.graph_fingerprint
    fp_curr = current_state.graph_fingerprint

    feature_dict = {
        # Graph Similarity
        "SHD":
            similarity["SHD"],

        "Jaccard":
            similarity["Jaccard"],

        "AddedEdges":
            similarity["AddedEdges"],

        "RemovedEdges":
            similarity["RemovedEdges"],

        "MeanEdgeWeightDifference":
            similarity["MeanEdgeWeightDifference"],

        "MaxEdgeWeightDifference":
            similarity["MaxEdgeWeightDifference"],

        # Fingerprint Differences
        "DensityDifference":
            similarity["density_difference"],

        "AverageInDegreeDifference":
            similarity["avg_in_degree_difference"],

        "AverageOutDegreeDifference":
            similarity["avg_out_degree_difference"],

        "ClusteringDifference":
            similarity["clustering_difference"],

        "AverageEdgeWeightDifference":
            similarity["edge_weight_difference"],

        "AdjacencySparsityDifference":
            similarity["sparsity_difference"],

        # Regime Features
        "PreviousRegime":
            stored_state.dominant_regime,

        "CurrentRegime":
            current_state.dominant_regime,

        "RegimeChanged":
            regime_changed,

        "PreviousConfidence":
            stored_state.average_confidence,

        "CurrentConfidence":
            current_state.average_confidence,

        "ConfidenceDifference":
            confidence_difference,

        "PreviousEntropy":
            stored_state.average_entropy,

        "CurrentEntropy":
            current_state.average_entropy,

        "EntropyDifference":
            entropy_difference,

        # Graph Fingerprints
        "PreviousDensity":
            fp_prev["density"],

        "CurrentDensity":
            fp_curr["density"],

        "PreviousAverageDegree":
            (
                fp_prev["avg_in_degree"] +
                fp_prev["avg_out_degree"]
            ) / 2,

        "CurrentAverageDegree":
            (
                fp_curr["avg_in_degree"] +
                fp_curr["avg_out_degree"]
            ) / 2
    }

    return feature_dict

# Pseudo Label Generation
def generate_pseudo_label(features):
    """
    Generate pseudo-labels for supervised training.

    Label = 1
        Significant graph update required

    Label = 0
        Existing graph can be retained
    """

    update = False

    if features["SHD"] >= SHD_THRESHOLD:
        update = True

    if features["RegimeChanged"]:
        update = True

    if features["ConfidenceDifference"] <= -CONFIDENCE_DROP:
        update = True

    if features["EntropyDifference"] >= ENTROPY_INCREASE:
        update = True

    return int(update)

# Final Feature Vector
def build_training_sample(stored_state, current_state):
    """
    Creates one ML training sample.

    Returns
    -------
    feature_dict
    """

    features = extract_features(
        stored_state,
        current_state
    )

    features["UpdateLabel"] = generate_pseudo_label(
        features
    )

    return features