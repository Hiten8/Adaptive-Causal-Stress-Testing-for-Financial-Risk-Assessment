"""Bootstrap-calibrated evidence for causal-graph update decisions."""

from dataclasses import dataclass
from typing import Iterable, List

import networkx as nx
import numpy as np

from src.core.causal_state import CausalState


@dataclass
class ChangeSignificanceResult:
    observed_distance: float
    null_threshold: float
    empirical_p_value: float
    significant: bool
    confidence_level: float
    null_sample_size: int


class BootstrapChangeTest:
    """Compare an observed graph shift with stable-regime bootstrap variation."""

    def __init__(self, confidence_level: float = 0.95):
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        self.confidence_level = confidence_level

    @staticmethod
    def distance(reference: CausalState, candidate: CausalState) -> float:
        """Structural Hamming distance plus mean common-edge weight movement."""
        ref_edges = set(reference.graph.edges())
        new_edges = set(candidate.graph.edges())
        shd = len(ref_edges.symmetric_difference(new_edges))
        common = ref_edges.intersection(new_edges)
        if not common:
            return float(shd)
        mean_weight_shift = np.mean([
            abs(float(reference.graph[u][v].get("weight", 1.0)) - float(candidate.graph[u][v].get("weight", 1.0)))
            for u, v in common
        ])
        return float(shd + mean_weight_shift)

    def assess(
        self,
        deployed_state: CausalState,
        observed_state: CausalState,
        stable_null_states: Iterable[CausalState],
    ) -> ChangeSignificanceResult:
        null_distances: List[float] = [self.distance(deployed_state, state) for state in stable_null_states]
        if not null_distances:
            raise ValueError("At least one stable-regime bootstrap graph is required.")
        observed = self.distance(deployed_state, observed_state)
        threshold = float(np.quantile(null_distances, self.confidence_level))
        # Add-one smoothing prevents reporting an impossible p-value of zero.
        p_value = (sum(value >= observed for value in null_distances) + 1) / (len(null_distances) + 1)
        return ChangeSignificanceResult(
            observed_distance=observed,
            null_threshold=threshold,
            empirical_p_value=float(p_value),
            significant=observed > threshold,
            confidence_level=self.confidence_level,
            null_sample_size=len(null_distances),
        )
