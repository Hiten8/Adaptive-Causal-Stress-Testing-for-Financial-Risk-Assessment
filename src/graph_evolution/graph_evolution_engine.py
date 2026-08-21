"""
Graph Evolution Engine

Purpose
-------
Generates plausible candidate causal graphs from the currently
accepted causal graph based on observed primary structural changes.

The engine coordinates:

    Detected Changes
            |
            v
    PropagationRuleEngine
            |
            v
    InteractionEngine
            |
            v
    InteractionPlan
            |
            v
    Evolution Actions
            |
            v
    Candidate CausalStates

This module DOES NOT rank or stress-test graphs.

It generates candidate CausalStates and attaches the
InteractionPlan responsible for generating each candidate.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.core.causal_state import CausalState

from src.graph_evolution.interaction_engine import (
    InteractionEngine,
)

from src.graph_evolution.interaction_plan import (
    InteractionPlan,
    PrimaryChangePlan,
    ActionStep,
)

from src.graph_evolution.operators import (
    get_operator,
)

from src.graph_evolution.propagation_rules import (
    PropagationRuleEngine,
)

from src.regime.regime_profiler import (
    RegimeProfiler,
)


class GraphEvolutionEngine:

    def __init__(self):
        self.rule_engine = (
            PropagationRuleEngine()
        )

        self.interaction_engine = (
            InteractionEngine()
        )

        self.regime_profiler = (
            RegimeProfiler()
        )

    # PUBLIC API
    def generate_candidates(
        self,
        stored_state: CausalState,
        current_state: CausalState,
        decision: dict
    ) -> List[CausalState]:

        """
        Generate candidate causal states from the detected
        structural changes.

        Parameters
        ----------
        stored_state : CausalState
            Currently accepted causal state.

        current_state : CausalState
            Current observed causal state.

        decision : dict
            Output produced by ChangeGate.

        Returns
        -------
        List[CausalState]
            Candidate evolved causal states.
        """

        detected_changes = (
            decision["DetectedChanges"]
        )

        self._print_input_state(
            stored_state,
            current_state,
            detected_changes
        )

        # Profile current regime.
        regime_profile = (
            self.regime_profiler.profile(
                current_state
            )
        )

        self._print_regime_profile(
            regime_profile
        )

        # Build primary structural changes.
        primary_changes = (
            self._build_primary_changes(
                detected_changes
            )
        )

        if not primary_changes:
            print(
                "No primary structural changes detected."
            )

            return []

        print()
        print(
            f"Primary Changes : "
            f"{len(primary_changes)}"
        )

        # Generate propagation rules for EACH primary change separately.
        #
        # This grouping is important because InteractionEngine uses the
        # grouping to determine which rules originate from independent
        # structural changes.
        propagation_rules_per_change = []

        for primary_change in primary_changes:
            rules = (
                self.rule_engine.generate_rules(
                    primary_change
                )
            )

            propagation_rules_per_change.append(
                rules
            )

        # Generate complete interaction plans.
        interaction_plans = (
            self.interaction_engine
            .generate_interaction_plans(
                propagation_rules_per_change=
                    propagation_rules_per_change,

                primary_changes=
                    primary_changes
            )
        )

        print()
        print(
            f"Generated Interaction Plans : "
            f"{len(interaction_plans)}"
        )

        # Execute interaction plans.
        candidates = []
        visited = set()

        for plan in interaction_plans:
            generated_states = (
                self._execute_interaction_plan(
                    stored_state=
                        stored_state,

                    current_state=
                        current_state,

                    plan=
                        plan,

                    regime_profile=
                        regime_profile
                )
            )

            for candidate in generated_states:
                fingerprint = (
                    self.compute_fingerprint(
                        candidate.graph
                    )
                )

                if fingerprint in visited:
                    continue

                visited.add(
                    fingerprint
                )

                candidate.graph_fingerprint = {
                    "hash":
                        fingerprint
                }

                candidates.append(
                    candidate
                )

        print()
        print(
            f"Generated "
            f"{len(candidates)} "
            f"candidate graphs."
        )

        return candidates

    # Primary Change Construction
    def _build_primary_changes(
        self,
        detected_changes: Dict
    ) -> List[Dict]:

        """
        Convert the ChangeGate DetectedChanges structure into
        a flat ordered list of primary structural changes.

        The ordering is:

            1. EDGE_ADDITION
            2. EDGE_REMOVAL
            3. WEIGHT_CHANGE
            4. ISOLATED_NODE
        """

        primary_changes = []

        # Edge additions
        for edge in detected_changes.get(
            "AddedEdges",
            []
        ):
            primary_changes.append(
                {
                    "type":
                        "EDGE_ADDITION",

                    "edge":
                        edge
                }
            )

        # Edge removals
        for edge in detected_changes.get(
            "RemovedEdges",
            []
        ):
            primary_changes.append(
                {
                    "type":
                        "EDGE_REMOVAL",

                    "edge":
                        edge
                }
            )

        # Weight changes
        for change in detected_changes.get(
            "WeightChangedEdges",
            []
        ):
            primary_changes.append(
                {
                    "type":
                        "WEIGHT_CHANGE",

                    "edge":
                        change["edge"],

                    "old_weight":
                        change["old_weight"],

                    "new_weight":
                        change["new_weight"]
                }
            )

        # Isolated nodes
        for node in detected_changes.get(
            "IsolatedNodes",
            []
        ):
            primary_changes.append(
                {
                    "type":
                        "ISOLATED_NODE",

                    "node":
                        node
                }
            )

        return primary_changes

    # Interaction Plan Execution
    def _execute_interaction_plan(
        self,
        stored_state: CausalState,
        current_state: CausalState,
        plan: InteractionPlan,
        regime_profile=None
    ) -> List[CausalState]:

        """
        Execute one complete InteractionPlan.

        Each InteractionPlan contains:

            PrimaryChangePlan
                |
                +-- ActionStep
                +-- ActionStep

        The actions are executed according to the execution order
        stored by PrimaryChangePlan.add_action().

        Returns
        -------
        List[CausalState]
            Candidate states generated by the plan.
        """

        if plan is None:
            return []

        if not plan.primary_change_plans:
            return []

        # The execution starts from the currently accepted graph.
        working_states = [
            deepcopy(
                stored_state
            )
        ]

        # Execute primary changes in the order stored in the plan.
        for primary_plan in (
            plan.primary_change_plans
        ):

            if not primary_plan.action_steps:
                continue

            next_states = []

            # Actions within one primary change are executed in their
            # explicit execution order.
            ordered_actions = sorted(
                primary_plan.action_steps,
                key=lambda step:
                    step.execution_order
            )

            for working_state in working_states:
                states_after_primary = [
                    working_state
                ]

                for action_step in ordered_actions:
                    states_after_action = []

                    for state in states_after_primary:
                        generated = (
                            self._execute_action_step(
                                stored_state=
                                    state,

                                current_state=
                                    current_state,

                                primary_plan=
                                    primary_plan,

                                action_step=
                                    action_step
                            )
                        )

                        states_after_action.extend(
                            generated
                        )

                    states_after_primary = (
                        states_after_action
                    )

                    if not states_after_primary:
                        break

                next_states.extend(
                    states_after_primary
                )

            working_states = (
                next_states
            )

            if not working_states:
                break

        # Attach plan-level metadata and probability.
        candidates = []

        for candidate in working_states:
            candidate = deepcopy(
                candidate
            )

            candidate.candidate_probability = (
                plan.cumulative_score
            )

            candidate.notes[
                "InteractionPlan"
            ] = plan.hypothesis_name

            candidate.notes[
                "InteractionLevel"
            ] = plan.interaction_level

            candidate.notes[
                "InteractionPlanScore"
            ] = plan.cumulative_score

            if regime_profile is not None:
                candidate.notes[
                    "DominantRegime"
                ] = (
                    regime_profile.dominant_regime
                )

                candidate.notes[
                    "VolatilityLevel"
                ] = (
                    regime_profile.volatility_level
                )

                candidate.notes[
                    "TransitionRisk"
                ] = (
                    regime_profile.transition_risk
                )

            candidates.append(
                candidate
            )

        return candidates

    # Action Execution
    def _execute_action_step(
        self,
        stored_state: CausalState,
        current_state: CausalState,
        primary_plan: PrimaryChangePlan,
        action_step: ActionStep
    ) -> List[CausalState]:

        """
        Execute one ActionStep.

        The actual graph operation is delegated to the operator
        associated with the propagation rule.

        The InteractionEngine / InteractionPlan determine WHAT
        should happen.

        The operator determines HOW the graph is modified.
        """

        # Retrieve the actual propagation rule.
        #
        # interaction_engine.py stores the rule object in
        # ActionStep.context["rule_object"].

        rule = (
            action_step.context.get(
                "rule_object"
            )
            if action_step.context
            else None
        )

        if rule is None:
            print(
                "Warning: ActionStep does not contain "
                "the original propagation rule."
            )

            return []

        # Resolve the operator.
        operator = get_operator(
            rule.operator
        )

        if operator is None:
            print(
                f"Warning: No operator found for "
                f"'{rule.operator}'."
            )

            return []

        # Reconstruct the primary change.
        primary_change = (
            deepcopy(
                primary_plan.change
            )
        )

        # Apply the graph operation.
        generated = operator.apply(
            stored_state,
            current_state,
            primary_change
        )

        if generated is None:
            return []

        # Normalize single-state returns.
        if isinstance(
            generated,
            CausalState
        ):

            generated = [
                generated
            ]

        # Annotate each generated candidate.
        candidates = []

        for candidate in generated:
            if candidate is None:
                continue

            candidate.notes[
                "PropagationRule"
            ] = (
                action_step.propagation_rule
            )

            candidate.notes[
                "Action"
            ] = (
                action_step.action_name
            )

            candidate.notes[
                "PrimaryChange"
            ] = (
                deepcopy(
                    primary_change
                )
            )

            candidate.notes[
                "ExecutionOrder"
            ] = (
                action_step.execution_order
            )

            candidate.notes[
                "ActionBaseScore"
            ] = (
                action_step.base_score
            )

            candidates.append(
                candidate
            )

        return candidates

    def _print_input_state(
        self,
        stored_state: CausalState,
        current_state: CausalState,
        detected_changes: Dict
    ):

        print()
        print(
            "Detected Changes:"
        )

        print(
            detected_changes
        )

        print()
        print(
            "Stored Graph Edges"
        )

        print(
            list(
                stored_state.graph.edges()
            )
        )

        print()
        print(
            "Current Graph Edges"
        )

        print(
            list(
                current_state.graph.edges()
            )
        )

        print()

        print(
            "Primary Change Count : "
            f"{detected_changes.get(
                'PrimaryChangeCount',
                'N/A'
            )}"
        )

    def _print_regime_profile(
        self,
        regime_profile
    ):

        print(
            "=" * 70
        )

        print(
            "Graph Evolution Engine"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Dominant Regime : "
            f"{regime_profile.dominant_regime}"
        )

        print(
            f"Volatility      : "
            f"{regime_profile.volatility_level}"
        )

        print(
            f"Transition Risk : "
            f"{regime_profile.transition_risk}"
        )

        print()

    def compute_fingerprint(
        self,
        graph: nx.DiGraph
    ):

        """
        Compute a structural fingerprint for a graph.

        Currently the fingerprint is based on the sorted directed
        edge set, preserving the behaviour of the previous engine.
        """

        edges = tuple(
            sorted(
                graph.edges()
            )
        )

        return hash(
            edges
        )

# Testing
if __name__ == "__main__":

    from src.storage.active_graph_manager import (
        ActiveGraphManager
    )

    from src.storage.causal_state_repository import (
        CausalStateRepository
    )

    from src.change_gate.change_gate import (
        ChangeGate
    )

    manager = (
        ActiveGraphManager()
    )

    repository = (
        CausalStateRepository()
    )

    stored_state = (
        manager.load()
    )

    current_state = (
        repository.load_all()[1]
    )

    gate = (
        ChangeGate()
    )

    decision = gate.evaluate(
        stored_state,
        current_state
    )

    engine = (
        GraphEvolutionEngine()
    )

    candidates = (
        engine.generate_candidates(
            stored_state,
            current_state,
            decision
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        "Candidate Graphs"
    )
    print(
        "=" * 70
    )

    for i, state in enumerate(
        candidates
    ):
        print()
        print(
            f"Candidate {i + 1}"
        )
        print(
            f"Edges : "
            f"{state.graph.number_of_edges()}"
        )
        print(
            f"Plausibility Score : "
            f"{state.candidate_probability:.4f}"
        )
        print(
            f"Interaction Plan : "
            f"{state.notes.get(
                'InteractionPlan',
                'N/A'
            )}"
        )
        print(
            f"Interaction Level : "
            f"{state.notes.get(
                'InteractionLevel',
                'N/A'
            )}"
        )
        print(
            f"Cumulative Plausibility : "
            f"{state.notes.get(
                'InteractionPlanScore',
                'N/A'
            )}"
        )
        print(
            f"Rule : "
            f"{state.notes.get(
                'PropagationRule',
                'N/A'
            )}"
        )