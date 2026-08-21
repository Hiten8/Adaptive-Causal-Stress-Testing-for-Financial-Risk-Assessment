"""
Confidence-Guided Adaptive Change Gate

Purpose
-------
Determines whether the current causal graph represents a
meaningful structural change relative to the stored causal state.

The Change Gate performs:

    1. Feature extraction
    2. Structural change detection
    3. ML-based structural change scoring
    4. Adaptive threshold calculation
    5. Decision confidence estimation
    6. Primary structural change normalization
    7. Human-readable explanation

Important
---------
PrimaryChangeCount is defined consistently with the downstream
graph-evolution architecture.

A primary change is any structural change belonging to one of:

    EDGE_ADDITION
    EDGE_REMOVAL
    WEIGHT_CHANGE
    ISOLATED_NODE

Therefore:

    PrimaryChangeCount
        =
        AddedEdges
        + RemovedEdges
        + WeightChangedEdges
        + IsolatedNodes

The graph-change detector may provide its own count, but the
Change Gate recalculates the authoritative count so that all
downstream modules use the same definition.
"""

from pathlib import Path
import joblib
from src.change_gate.feature_extractor import (
    extract_features
)
from src.change_gate.graph_change_detector import (
    detect_graph_changes
)

MODEL_PATH = Path(
    "models/change_gate/change_gate_xgboost.pkl"
)

BASE_THRESHOLD = 0.70
MIN_THRESHOLD = 0.40
MAX_THRESHOLD = 0.90

class ChangeGate:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Cannot find trained model:\n"
                f"{MODEL_PATH}\n\n"
                "Run train_change_gate.py first."
            )

        self.model = joblib.load(
            MODEL_PATH
        )

    def _calculate_primary_change_count(
        self,
        detected_changes
    ):
        """
        Calculate the authoritative number of primary structural changes.

        A primary change is one item in any of:

            AddedEdges
            RemovedEdges
            WeightChangedEdges
            IsolatedNodes

        This definition is intentionally aligned with the
        GraphEvolutionEngine and PropagationRuleEngine.

        Returns
        -------
        int
            Total number of primary structural changes.
        """

        added_edges = detected_changes.get(
            "AddedEdges",
            []
        )

        removed_edges = detected_changes.get(
            "RemovedEdges",
            []
        )

        weight_changed_edges = detected_changes.get(
            "WeightChangedEdges",
            []
        )

        isolated_nodes = detected_changes.get(
            "IsolatedNodes",
            []
        )

        return (len(added_edges) + len(removed_edges) + len(weight_changed_edges) + len(isolated_nodes))

    def _normalize_detected_changes(
        self,
        detected_changes
    ):
        """
        Normalize the graph-change detector output so that the
        downstream graph-evolution pipeline receives a consistent
        primary-change definition.

        The original change detector output is preserved, except
        for PrimaryChangeCount, which is recalculated from the
        actual structural-change lists.
        """

        normalized_changes = dict(
            detected_changes
        )

        primary_change_count = (
            self._calculate_primary_change_count(
                normalized_changes
            )
        )

        normalized_changes[
            "PrimaryChangeCount"
        ] = primary_change_count

        return normalized_changes

    def adaptive_threshold(
        self,
        stored_state,
        current_state,
        features
    ):
        threshold = BASE_THRESHOLD

        # Regime Transition
        if features[
            "RegimeChanged"
        ]:
            threshold -= 0.10

        # Large Structural Change
        if features[
            "SHD"
        ] > 10:
            threshold -= 0.05

        # Increasing Entropy
        if features[
            "EntropyDifference"
        ] > 0.20:
            threshold -= 0.05

        # Very Stable Market
        if current_state.average_confidence > 0.95:
            threshold += 0.05

        # Stable Graph
        if features[
            "Jaccard"
        ] > 0.95:
            threshold += 0.03

        # Clamp threshold
        threshold = max(
            MIN_THRESHOLD,
            threshold
        )

        threshold = min(
            MAX_THRESHOLD,
            threshold
        )

        return float(
            threshold
        )

    def explain(
        self,
        stored_state,
        current_state,
        features,
        score,
        threshold
    ):
        reasons = []

        # Regime
        if features[
            "RegimeChanged"
        ]:
            reasons.append(
                "Regime transition detected."
            )

        else:
            reasons.append(
                "No regime transition detected."
            )

        # Structural Hamming Distance
        if features[
            "SHD"
        ] > 10:
            reasons.append(
                f"Large structural change "
                f"(SHD={features['SHD']})."
            )

        elif features[
            "SHD"
        ] > 5:
            reasons.append(
                f"Moderate structural change "
                f"(SHD={features['SHD']})."
            )

        else:
            reasons.append(
                f"Minor structural change "
                f"(SHD={features['SHD']})."
            )

        # Graph Similarity
        if features[
            "Jaccard"
        ] >= 0.95:
            reasons.append(
                "Graph topology remained highly similar."
            )

        elif features[
            "Jaccard"
        ] >= 0.80:
            reasons.append(
                "Moderate graph topology changes detected."
            )

        else:
            reasons.append(
                "Graph topology changed substantially."
            )

        # Entropy
        if features[
            "EntropyDifference"
        ] > 0.20:
            reasons.append(
                "Regime uncertainty increased."
            )

        elif features[
            "EntropyDifference"
        ] < -0.20:
            reasons.append(
                "Regime uncertainty decreased."
            )

        else:
            reasons.append(
                "Regime uncertainty remained stable."
            )

        # Confidence
        if features[
            "ConfidenceDifference"
        ] < -0.15:
            reasons.append(
                "Regime confidence decreased."
            )

        elif features[
            "ConfidenceDifference"
        ] > 0.15:
            reasons.append(
                "Regime confidence increased."
            )

        else:
            reasons.append(
                "Regime confidence remained stable."
            )

        # Final Decision
        if score >= threshold:
            reasons.append(
                f"Structural Change Score "
                f"({score:.3f}) "
                f"exceeds adaptive threshold "
                f"({threshold:.3f})."
            )

        else:
            reasons.append(
                f"Structural Change Score "
                f"({score:.3f}) "
                f"is below adaptive threshold "
                f"({threshold:.3f})."
            )

        return reasons

    def evaluate(
        self,
        stored_state,
        current_state
    ):
        # Feature Extraction
        features = extract_features(
            stored_state,
            current_state
        )

        # Detect Structural Changes
        detected_changes = detect_graph_changes(
            stored_state.graph,
            current_state.graph
        )

        # Normalize Primary Change Definition
        detected_changes = (
            self._normalize_detected_changes(
                detected_changes
            )
        )

        # Arrange Features
        feature_vector = [
            features[column]
            for column in self.model.feature_names_in_
        ]

        # Structural Change Score
        structural_change_score = float(
            self.model.predict_proba(
                [feature_vector]
            )[0][1]
        )

        # Adaptive Threshold
        threshold = self.adaptive_threshold(
            stored_state,
            current_state,
            features
        )

        # Final Decision
        should_update = (
            structural_change_score
            >=
            threshold
        )

        # Decision Confidence
        margin = abs(
            structural_change_score
            -
            threshold
        )

        if margin >= 0.30:
            decision_confidence = "HIGH"

        elif margin >= 0.15:
            decision_confidence = "MEDIUM"

        else:
            decision_confidence = "LOW"

        reasons = self.explain(
            stored_state,
            current_state,
            features,
            structural_change_score,
            threshold
        )

        decision = {
            "GraphComparison":
                f"StoredGraph({stored_state.window_id})"
                f" -> "
                f"PCMCI({current_state.window_id})",

            "StructuralChangeScore":
                structural_change_score,

            "AdaptiveThreshold":
                threshold,

            "ShouldUpdate":
                should_update,

            "DecisionConfidence":
                decision_confidence,

            "DetectedChanges":
                detected_changes,

            "PrimaryChangeCount":
                detected_changes[
                    "PrimaryChangeCount"
                ],

            "Reasons":
                reasons,

            "FeatureVector":
                features
        }

        return decision