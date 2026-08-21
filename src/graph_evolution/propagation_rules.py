"""
Propagation Rules

Purpose
-------
Defines financially meaningful propagation hypotheses
given PRIMARY structural changes detected by the Change Gate.

This module DOES NOT modify graphs.

It only proposes possible secondary changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Propagation Rule
@dataclass
class PropagationRule:
    # Rule name
    name: str

    # Operator to invoke
    operator: str

    # Relative plausibility (NOT final probability)
    base_score: float

    # Additional metadata
    metadata: Dict = field(default_factory=dict)

# Rule Engine
class PropagationRuleEngine:
    def __init__(self):
        pass

    # Public API
    def generate_rules(
        self,
        primary_change: Dict
    ) -> List[PropagationRule]:

        change_type = primary_change["type"]

        if change_type == "EDGE_ADDITION":
            return self.edge_addition_rules(primary_change)

        elif change_type == "EDGE_REMOVAL":
            return self.edge_removal_rules(primary_change)

        elif change_type == "WEIGHT_CHANGE":
            return self.weight_change_rules(primary_change)

        elif change_type == "ISOLATED_NODE":
            return self.isolated_node_rules(primary_change)

        return []

    # EDGE ADDITION
    def edge_addition_rules(
        self,
        change
    ):
        return [
            PropagationRule(
                name="Keep Added Edge",
                operator="EdgeAddition",
                base_score=1.00
            ),

            PropagationRule(
                name="Strengthen Downstream",
                operator="StrengthenDownstream",
                base_score=0.75
            ),

            PropagationRule(
                name="Strengthen Upstream",
                operator="StrengthenUpstream",
                base_score=0.60
            ),

            PropagationRule(
                name="Create Alternative Path",
                operator="AlternativePath",
                base_score=0.45
            )
        ]

    # EDGE REMOVAL
    def edge_removal_rules(
        self,
        change
    ):
        return [
            PropagationRule(
                name="Remove Edge",
                operator="EdgeRemoval",
                base_score=1.00
            ),

            PropagationRule(
                name="Strengthen Remaining Parents",
                operator="StrengthenIncoming",
                base_score=0.80
            ),

            PropagationRule(
                name="Create Bypass",
                operator="BypassConnection",
                base_score=0.55
            ),

            PropagationRule(
                name="Reduce Downstream Influence",
                operator="ReduceOutgoing",
                base_score=0.50
            )
        ]

    # WEIGHT CHANGE
    def weight_change_rules(
        self,
        change
    ):
        return [
            PropagationRule(
                name="Apply New Weight",
                operator="WeightChange",
                base_score=1.00
            ),

            PropagationRule(
                name="Strengthen Neighbours",
                operator="NeighbourStrengthening",
                base_score=0.70
            ),

            PropagationRule(
                name="Weaken Competing Edges",
                operator="CompetingReduction",
                base_score=0.55
            )
        ]

    # ISOLATED NODE
    def isolated_node_rules(
        self,
        change
    ):
        return [
            PropagationRule(
                name="Reconnect Node",
                operator="IsolationRecovery",
                base_score=1.00
            ),

            PropagationRule(
                name="Reconnect to Strongest Parent",
                operator="StrongParentRecovery",
                base_score=0.80
            ),

            PropagationRule(
                name="Reconnect to Strongest Child",
                operator="StrongChildRecovery",
                base_score=0.70
            )
        ]

# Testing
if __name__ == "__main__":

    engine = PropagationRuleEngine()
    test_change = {
        "type": "EDGE_REMOVAL",
        "edge": ("Inflation", "SP500")
    }

    rules = engine.generate_rules(test_change)
    print("=" * 70)
    print("Propagation Rules")
    print("=" * 70)

    for rule in rules:
        print(
            f"{rule.name:30}"
            f"{rule.operator:25}"
            f"{rule.base_score:.2f}"
        )

    print("=" * 70)