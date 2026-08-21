"""
CausalState

Stores everything related to one causal graph.
"""

from dataclasses import dataclass, field
from typing import Dict
import networkx as nx
import numpy as np


@dataclass
class CausalState:

    # Window Information
    window_id: int
    start_date: str
    end_date: str
    # window_size: int
    # step_size: int

    # Regime Information
    dominant_regime: int
    regime_distribution: Dict[int, float]
    average_confidence: float
    average_entropy: float

    # Graph
    graph: nx.DiGraph
    adjacency_matrix: np.ndarray

    # PCMCI Results
    edge_strength_matrix: np.ndarray
    p_value_matrix: np.ndarray
    variable_names: list

    # Additional Metadata
    graph_metrics: Dict = field(default_factory=dict)
    graph_fingerprint: Dict = field(default_factory=dict)
    change_features: Dict = field(default_factory=dict)
    accepted: bool = False
    candidate_probability: float = 0.0
    notes: Dict = field(default_factory=dict)