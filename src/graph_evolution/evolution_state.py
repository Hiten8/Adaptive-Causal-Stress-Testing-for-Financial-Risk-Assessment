"""
Evolution State

Represents one state in the Graph Evolution Search Space.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from copy import deepcopy
from src.core.causal_state import CausalState

@dataclass
class EvolutionState:
    # Current evolved causal state
    causal_state: CausalState

    # Operations applied to reach this state
    applied_operations: List[Dict] = field(default_factory=list)

    # Remaining primary changes that have not yet been explored
    remaining_changes: List[Dict] = field(default_factory=list)

    # Cumulative probability of this evolution
    cumulative_probability: float = 1.0

    # Number of propagation steps
    depth: int = 0

    # Parent fingerprint (for traceability)
    parent_fingerprint: str = ""

    # Whether this state has already been expanded
    expanded: bool = False

    def clone(self):
        return deepcopy(self)

    def mark_expanded(self):
        self.expanded = True

    def add_operation(
        self,
        operator_name: str,
        primary_change: Dict,
        probability: float
    ):
        
        self.applied_operations.append({
            "operator": operator_name,
            "primary_change": primary_change,
            "probability": probability
        })

        self.cumulative_probability *= probability
        self.depth += 1

    @property
    def fingerprint(self):
        return self.causal_state.graph_fingerprint

    @property
    def graph(self):
        return self.causal_state.graph