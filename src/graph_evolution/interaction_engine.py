"""
Interaction Engine

Purpose
-------
Generates interaction candidates between multiple
primary structural changes.

This module does NOT modify graphs.

The InteractionEngine is responsible for:

1. Receiving propagation rules generated for structural changes.
2. Normalizing those rules into interaction candidates.
3. Detecting incompatible actions.
4. Preventing contradictory actions on the same edge.
5. Generating valid interaction combinations.
6. Passing those combinations to InteractionPlan for
   final planning/scoring.

The actual graph modification is performed later by
GraphEvolutionEngine.

Architecture
------------
Primary Structural Changes
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
GraphEvolutionEngine
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class InteractionCandidate:
    """
    Represents a single action that may participate in an
    interaction plan.

    This is intentionally independent from graph mutation.
    """

    action: str
    rule: Any
    source_change_index: Optional[int] = None
    edge: Optional[Tuple[str, str]] = None
    node: Optional[str] = None
    base_score: float = 1.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class InteractionHypothesis:
    """
    Represents a candidate combination of Evolution Actions.

    The hypothesis itself does not modify the graph.

    It is later consumed by InteractionPlan.
    """

    hypothesis_name: str
    actions: List[Dict]
    interaction_level: int
    base_score: float
    metadata: Dict = field(default_factory=dict)

class InteractionEngine:
    def __init__(self):
        """
        Initialize the InteractionEngine.

        The engine is intentionally stateless with respect to
        graph structure. Graph state must not be modified here.
        """

        pass

    # Rule Normalization
    def _extract_edge(
        self,
        rule: Any
    ) -> Optional[Tuple[str, str]]:
        """
        Extract the affected edge from a propagation rule.

        Different versions of PropagationRule may expose the
        edge using different attribute names. This helper keeps
        that representation detail isolated.

        Returns
        -------
        tuple or None
            (source, target) if an edge can be identified.
        """

        if rule is None:
            return None

        # Most direct representation
        edge = getattr(rule, "edge", None)

        if edge is not None:
            return self._normalize_edge(edge)

        # Some rule representations may store the affected
        # edge under target_edge.
        edge = getattr(rule, "target_edge", None)

        if edge is not None:
            return self._normalize_edge(edge)

        # Some representations may use affected_edge.
        edge = getattr(rule, "affected_edge", None)

        if edge is not None:
            return self._normalize_edge(edge)

        return None

    def _normalize_edge(
        self,
        edge: Any
    ) -> Optional[Tuple[str, str]]:
        """
        Normalize an edge representation into:

            (source, target)

        Returns None if the supplied value cannot be interpreted
        as an edge.
        """

        if edge is None:
            return None

        if isinstance(edge, tuple):
            if len(edge) == 2:
                return (
                    edge[0],
                    edge[1]
                )

        if isinstance(edge, list):
            if len(edge) == 2:
                return (
                    edge[0],
                    edge[1]
                )

        return None

    # Action Extraction
    def _get_operator(
        self,
        rule: Any
    ) -> Optional[str]:
        """
        Obtain the Evolution Action operator from a rule.

        This is deliberately centralized so that interaction
        generation does not depend on how the propagation-rule
        implementation stores its operator.
        """

        if rule is None:
            return None

        operator = getattr(
            rule,
            "operator",
            None
        )

        if operator is not None:
            return str(operator)

        # Compatibility with dictionary-based rules.
        if isinstance(rule, dict):
            operator = rule.get(
                "operator"
            )

            if operator is not None:
                return str(operator)

            operator = rule.get(
                "action"
            )

            if operator is not None:
                return str(operator)

        return None

    def _get_base_score(
        self,
        rule: Any
    ) -> float:
        """
        Extract the base score associated with a propagation rule.
        """

        if rule is None:
            return 1.0

        score = getattr(
            rule,
            "base_score",
            None
        )

        if score is None and isinstance(
            rule,
            dict
        ):
            score = rule.get(
                "base_score"
            )

        if score is None:
            return 1.0

        try:
            return float(score)

        except (
            TypeError,
            ValueError
        ):
            return 1.0

    # Candidate Construction
    def _create_candidate(
        self,
        rule: Any,
        source_change_index: Optional[int] = None,
        primary_change: Optional[Dict] = None
    ) -> InteractionCandidate:
        """
        Convert a propagation rule into an InteractionCandidate.
        The candidate retains the primary edge or node involved in
        the structural change so that interaction hypotheses remain
        fully traceable.
        """

        operator = self._get_operator(
            rule
        )

        edge = None

        if (
            primary_change is not None
            and
            isinstance(
                primary_change,
                dict
            )
        ):
            edge = primary_change.get("edge")

        if edge is None:
            edge = self._extract_edge(rule)

        score = self._get_base_score(
            rule
        )

        node = None

        if (
            primary_change is not None
            and
            isinstance(
                primary_change,
                dict
            )
        ):
            node = primary_change.get(
                "node"
            )

        return InteractionCandidate(
            action=operator,
            rule=rule,
            source_change_index=
                source_change_index,
            edge=edge,
            node=node,
            base_score=score
        )

    # def _build_candidates(
    #     self,
    #     propagation_rules_per_change: List[List],
    #     primary_changes: Optional[List[Dict]] = None
    # ) -> List[InteractionCandidate]:
    #     """
    #     Flatten propagation rules while retaining the index of
    #     the structural change that produced each rule.

    #     Retaining source_change_index is important because an
    #     interaction should represent a combination of actions
    #     arising from different structural changes, rather than
    #     blindly combining arbitrary rules from the same change.
    #     """

    #     candidates = []

    #     for change_index, rules in enumerate(
    #         propagation_rules_per_change
    #     ):
    #         primary_change = None

    #         if (
    #             primary_changes is not None
    #             and
    #             change_index < len(
    #                 primary_changes
    #             )
    #         ):
    #             primary_change = (
    #                 primary_changes[
    #                     change_index
    #                 ]
    #             )

    #         for rule in rules:
    #             candidate = self._create_candidate(
    #                 rule=rule,

    #             source_change_index=
    #                 change_index,

    #             primary_change=
    #                 primary_change
    #         )

    #     candidates.append(
    #         candidate
    #     )

    #     return candidates
    def _build_candidates(
    self,
    propagation_rules_per_change: List[List],
    primary_changes: Optional[List[Dict]] = None
    ) -> List[InteractionCandidate]:
        """
        Flatten propagation rules while retaining the index of
        the structural change that produced each rule.

        Each propagation rule must become its own
        InteractionCandidate.

        source_change_index is preserved so that the interaction
        engine can distinguish:

        Rule A from Change 0
        Rule B from Change 1

        from alternative rules belonging to the same change.
        """

        candidates = []

        for change_index, rules in enumerate(
            propagation_rules_per_change
        ):
            primary_change = None

            if (
                primary_changes is not None
                and
                change_index < len(
                    primary_changes
                )
            ):

                primary_change = (
                    primary_changes[
                        change_index
                    ]
                )

            for rule in rules:

                candidate = self._create_candidate(
                    rule=rule,

                    source_change_index=
                        change_index,

                    primary_change=
                        primary_change
                )

                candidates.append(
                    candidate
                )

        return candidates
    
    # Conflict Detection
    def _same_edge(
        self,
        candidate_a: InteractionCandidate,
        candidate_b: InteractionCandidate
    ) -> bool:
        """
        Determine whether two candidates operate on the same edge.
        """

        if candidate_a.edge is None:
            return False

        if candidate_b.edge is None:
            return False

        return (
            candidate_a.edge ==
            candidate_b.edge
        )

    def _is_add_operation(
        self,
        operator: Optional[str]
    ) -> bool:
        """
        Determine whether an operator represents an edge addition.
        """

        if operator is None:
            return False

        normalized = operator.upper()

        return normalized in {
            "ADD_EDGE",
            "EDGE_ADDITION",
            "ADDEDGE"
        }

    def _is_remove_operation(
        self,
        operator: Optional[str]
    ) -> bool:
        """
        Determine whether an operator represents an edge removal.
        """

        if operator is None:
            return False

        normalized = operator.upper()

        return normalized in {
            "REMOVE_EDGE",
            "EDGE_REMOVAL",
            "REMOVEEDGE"
        }

    def _are_conflicting_actions(
        self,
        candidate_a: InteractionCandidate,
        candidate_b: InteractionCandidate
    ) -> bool:
        """
        Determine whether two candidates cannot coexist.

        The most important conflict is:

            add_edge(A, B)
            remove_edge(A, B)

        These must NEVER occur in the same interaction
        hypothesis.

        This check is deliberately performed before an
        interaction hypothesis is created.
        """

        if not self._same_edge(
            candidate_a,
            candidate_b
        ):
            return False

        add_a = self._is_add_operation(
            candidate_a.action
        )

        add_b = self._is_add_operation(
            candidate_b.action
        )

        remove_a = self._is_remove_operation(
            candidate_a.action
        )

        remove_b = self._is_remove_operation(
            candidate_b.action
        )

        if (
            (add_a and remove_b)
            or
            (remove_a and add_b)
        ):
            return True

        return False

    def _is_valid_combination(
        self,
        candidates: List[InteractionCandidate]
    ) -> bool:
        """
        Validate an entire candidate combination.

        Every pair is checked for incompatibility.

        This prevents contradictory operations from entering
        InteractionPlan in the first place.
        """

        for candidate_a, candidate_b in combinations(
            candidates,
            2
        ):

            if self._are_conflicting_actions(
                candidate_a,
                candidate_b
            ):
                return False

        return True
    
    # Candidate Deduplication
    def _candidate_signature(
        self,
        candidate: InteractionCandidate
    ) -> Tuple:
        """
        Generate a stable signature for an interaction candidate.
        """

        return (
            candidate.action,
            candidate.edge,
            candidate.node,
            candidate.source_change_index
        )

    def _deduplicate_candidates(
        self,
        candidates: List[InteractionCandidate]
    ) -> List[InteractionCandidate]:
        """
        Remove duplicate candidates.

        Duplicate propagation rules should not result in duplicate
        interaction actions.
        """

        unique = []
        seen = set()

        for candidate in candidates:
            signature = self._candidate_signature(
                candidate
            )

            if signature in seen:
                continue

            seen.add(
                signature
            )

            unique.append(
                candidate
            )

        return unique

    # Hypothesis Action Conversion
    def _candidate_to_action(
        self,
        candidate: InteractionCandidate
    ) -> Dict:
        """
        Convert an InteractionCandidate into the action structure
        expected by the interaction-planning layer.
        """

        return {
            "action": candidate.action,
            "rule": candidate.rule,
            "edge": candidate.edge,
            "source_change_index":
                candidate.source_change_index,
            "base_score":
                candidate.base_score
        }

    # def _build_hypothesis_name(
    #     self,
    #     candidates: List[InteractionCandidate]
    # ) -> str:
    #     """
    #     Generate a readable hypothesis name.
    #     """

    #     operators = [
    #         candidate.action
    #         for candidate in candidates
    #     ]

    #     if len(operators) == 1:
    #         return (
    #             f"Single::{operators[0]}"
    #         )

    #     return "+".join(
    #         operators
    #     )
    def _build_hypothesis_name(
        self,
        candidates: List[InteractionCandidate]
    ) -> str:
        parts = []

        for candidate in candidates:
            operator = candidate.action

            edge = candidate.edge

            # Edge-based primary changes
            if edge is not None:
                try:
                    source, target = edge
                    parts.append(
                        f"{operator}"
                        f"[{source}->{target}]"
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    parts.append(
                        f"{operator}"
                        f"[{edge}]"
                    )

                continue

            # Node-based primary changes
            node = candidate.node
            if node is not None:
                parts.append(
                    f"{operator}"
                    f"[{node}]"
                )

                continue

            if candidate.source_change_index is not None:
                parts.append(
                    f"{operator}"
                    f"[Change"
                    f"{candidate.source_change_index}"
                    f"]"
                )

            else:
                parts.append(
                    str(operator)
                )

        return " + ".join(
                parts
        )

    # Score Calculation
    def _calculate_base_score(
        self,
        candidates: List[InteractionCandidate]
    ) -> float:
        """
        Calculate the base score for an interaction.

        The existing multiplicative scoring behaviour is retained.
        """

        score = 1.0

        for candidate in candidates:
            score *= candidate.base_score

        return score
    
    # Hypothesis Construction
    def _create_hypothesis(
        self,
        candidates: List[InteractionCandidate],
        interaction_level: int
    ) -> InteractionHypothesis:
        """
        Create an InteractionHypothesis from a valid combination
        of interaction candidates.

        The hypothesis is an intermediate representation.

        It is later converted into an InteractionPlan.
        """

        actions = [
            self._candidate_to_action(
                candidate
            )
            for candidate in candidates
        ]

        return InteractionHypothesis(
            hypothesis_name=
                self._build_hypothesis_name(
                    candidates
                ),

            actions=actions,

            interaction_level=
                interaction_level,

            base_score=
                self._calculate_base_score(
                    candidates
                ),

            metadata={
                "source_change_indices": [
                    candidate.source_change_index
                    for candidate in candidates
                ],

                "edges": [
                    candidate.edge
                    for candidate in candidates
                ]
            }
        )

    # Level 1 Interaction Generation
    def _generate_level_one(
        self,
        candidates: List[InteractionCandidate]
    ) -> List[InteractionHypothesis]:
        """
        Generate individual-action hypotheses.

        Each propagation rule is treated as an independent
        candidate at interaction level 1.
        """

        hypotheses = []

        for candidate in candidates:
            hypotheses.append(
                self._create_hypothesis(
                    candidates=[
                        candidate
                    ],
                    interaction_level=1
                )
            )

        return hypotheses

    # Level 2 Interaction Generation
    def _generate_level_two(
        self,
        candidates: List[InteractionCandidate]
    ) -> List[InteractionHypothesis]:
        """
        Generate pairwise interaction hypotheses.

        Two actions may interact only when:

        1. They originate from different primary changes.
        2. They do not represent contradictory operations.
        """

        hypotheses = []

        for pair in combinations(
            candidates,
            2
        ):

            candidate_a = pair[0]
            candidate_b = pair[1]

            # Do not combine alternative propagation rules belonging to
            # the same primary structural change.
            if (
                candidate_a.source_change_index
                ==
                candidate_b.source_change_index
            ):
                continue

            # Reject contradictory operations.
            # Example:
            # ADD_EDGE(A, B)
            # REMOVE_EDGE(A, B)
            # These must never coexist in one hypothesis.

            if self._are_conflicting_actions(
                candidate_a,
                candidate_b
            ):
                continue

            if not self._is_valid_combination(
                list(pair)
            ):
                continue

            hypotheses.append(
                self._create_hypothesis(
                    candidates=list(pair),
                    interaction_level=2
                )
            )

        return hypotheses

    # Level 3 Interaction Generation
    def _generate_level_three(
        self,
        candidates: List[InteractionCandidate]
    ) -> List[InteractionHypothesis]:
        """
        Generate triple interaction hypotheses.

        A valid triple must contain actions from three different
        primary structural changes.

        Every pair within the triple is also checked for
        incompatibility.
        """

        hypotheses = []

        if len(candidates) < 3:
            return hypotheses

        for triple in combinations(
            candidates,
            3
        ):

            source_indices = [
                candidate.source_change_index
                for candidate in triple
            ]

            # All actions must originate from different primary changes.
            if len(
                set(source_indices)
            ) != 3:
                continue

            # Validate all pairwise combinations inside the triple.
            if not self._is_valid_combination(
                list(triple)
            ):
                continue

            hypotheses.append(
                self._create_hypothesis(
                    candidates=list(triple),
                    interaction_level=3
                )
            )

        return hypotheses

    # Public Interaction Generation
    def generate_interactions(
        self,
        propagation_rules_per_change: List[List],
        primary_changes: Optional[List[Dict]] = None
    ) -> List[InteractionHypothesis]:
        """
        Generate valid interaction hypotheses.

        Parameters
        ----------
        propagation_rules_per_change : List[List]

            Propagation rules grouped according to their
            originating primary structural change.

            Example:

                [
                    [RuleA, RuleB],
                    [RuleC, RuleD]
                ]

        Returns
        -------
        List[InteractionHypothesis]

            Valid level 1, level 2 and level 3 hypotheses.

        Notes
        -----
        This method does not modify the graph.
        """

        if not propagation_rules_per_change:
            return []

        # Build normalized candidates.
        candidates = self._build_candidates(
            propagation_rules_per_change = propagation_rules_per_change,
            primary_changes = primary_changes
        )

        if not candidates:
            return []

        # Remove duplicate candidates.
        candidates = self._deduplicate_candidates(
            candidates
        )

        if not candidates:
            return []

        # Generate hypotheses.
        hypotheses = []

        # Level 1
        hypotheses.extend(
            self._generate_level_one(
                candidates
            )
        )

        # Level 2
        hypotheses.extend(
            self._generate_level_two(
                candidates
            )
        )

        # Level 3
        hypotheses.extend(
            self._generate_level_three(
                candidates
            )
        )

        return hypotheses

    # Primary Change Information
    def _extract_change_information(
        self,
        rule: Any,
        change_index: int,
        primary_changes: Optional[List[Dict]]
    ):
        """
        Obtain the primary-change information required by
        PrimaryChangePlan.

        Preferred source
        ----------------
        primary_changes[change_index]

        Fallback
        --------
        Information embedded inside the propagation rule.

        Final fallback
        --------------
        A minimal placeholder representation is created so that
        InteractionPlan can still be constructed.
        """

        # Preferred source: explicitly supplied primary changes.
        if (
            primary_changes is not None
            and
            change_index < len(primary_changes)
        ):
            change = primary_changes[
                change_index
            ]

            if isinstance(
                change,
                dict
            ):
                change_type = (
                    change.get(
                        "type"
                    )
                    or
                    change.get(
                        "change_type"
                    )
                    or
                    "UNKNOWN"
                )

                return (
                    str(change_type),
                    change
                )

        # Fallback: inspect the propagation rule.
        if isinstance(
            rule,
            dict
        ):
            embedded_change = (
                rule.get(
                    "change"
                )
            )

            if isinstance(
                embedded_change,
                dict
            ):
                change_type = (
                    embedded_change.get(
                        "type"
                    )
                    or
                    embedded_change.get(
                        "change_type"
                    )
                    or
                    "UNKNOWN"
                )

                return (
                    str(change_type),
                    embedded_change
                )

            change_type = (
                rule.get(
                    "change_type"
                )
                or
                rule.get(
                    "type"
                )
            )

            if change_type is not None:

                return (
                    str(change_type),
                    rule
                )

        # Attribute-based fallback.
        embedded_change = getattr(
            rule,
            "change",
            None
        )

        if isinstance(
            embedded_change,
            dict
        ):
            change_type = (
                embedded_change.get(
                    "type"
                )
                or
                embedded_change.get(
                    "change_type"
                )
                or
                "UNKNOWN"
            )

            return (
                str(change_type),
                embedded_change
            )

        change_type = getattr(
            rule,
            "change_type",
            None
        )

        if change_type is not None:
            return (
                str(change_type),
                {
                    "type":
                        str(change_type)
                }
            )

        return (
            "UNKNOWN",
            {
                "change_id":
                    change_index
            }
        )

    # Propagation Rule Representation
    def _rule_to_plan_string(
        self,
        rule: Any
    ) -> str:
        """
        Convert a propagation rule into the string representation
        expected by ActionStep.propagation_rule.

        The InteractionPlan dataclass defines propagation_rule as
        a string, so the interaction engine must not pass the
        complete rule object directly into that field.
        """

        if rule is None:
            return "UNKNOWN"

        if isinstance(
            rule,
            str
        ):
            return rule

        # Prefer an explicit rule name if available.
        if isinstance(
            rule,
            dict
        ):

            for key in (
                "rule_name",
                "name",
                "description",
                "rule"
            ):

                value = rule.get(
                    key
                )

                if value is not None:
                    return str(
                        value
                    )

        for attribute in (
            "rule_name",
            "name",
            "description"
        ):

            value = getattr(
                rule,
                attribute,
                None
            )

            if value is not None:
                return str(
                    value
                )

        return str(
            rule
        )

    # InteractionPlan Construction
    def _hypothesis_to_interaction_plan(
        self,
        hypothesis: InteractionHypothesis,
        primary_changes: Optional[List[Dict]] = None
    ):
        """
        Convert an InteractionHypothesis into the new
        InteractionPlan hierarchy.

        Structure produced:

            InteractionPlan
                |
                +-- PrimaryChangePlan
                |       |
                |       +-- ActionStep
                |       +-- ActionStep
                |
                +-- PrimaryChangePlan
                        |
                        +-- ActionStep

        This is the main bridge between InteractionEngine and
        interaction_plan.py.
        """

        from src.graph_evolution.interaction_plan import (
            InteractionPlan,
            PrimaryChangePlan,
            ActionStep
        )

        plan = InteractionPlan(
            hypothesis_name=
                hypothesis.hypothesis_name,

            interaction_level=
                hypothesis.interaction_level,

            metadata=
                dict(
                    hypothesis.metadata
                ),

            notes={
                "source":
                    "InteractionEngine"
            }
        )

        # Group actions according to their originating primary change.
        grouped_actions = {}

        for action in hypothesis.actions:
            change_index = action.get(
                "source_change_index"
            )

            if change_index is None:
                change_index = 0

            if change_index not in grouped_actions:
                grouped_actions[
                    change_index
                ] = []

            grouped_actions[
                change_index
            ].append(
                action
            )

        # Construct one PrimaryChangePlan for each primary change.
        for change_index in sorted(
            grouped_actions.keys()
        ):

            actions = grouped_actions[
                change_index
            ]

            # Obtain change information.
            first_rule = None

            if actions:
                first_rule = actions[0].get(
                    "rule"
                )

            change_type, change = (
                self._extract_change_information(
                    rule=first_rule,
                    change_index=change_index,
                    primary_changes=primary_changes
                )
            )

            primary_plan = PrimaryChangePlan(
                change_id=
                    change_index,

                change_type=
                    change_type,

                change=
                    change,

                notes={
                    "source":
                        "InteractionEngine"
                }
            )

            # Convert each interaction action into ActionStep.
            for action in actions:
                rule = action.get(
                    "rule"
                )

                action_name = (
                    action.get(
                        "action"
                    )
                    or
                    "UNKNOWN"
                )

                base_score = action.get(
                    "base_score"
                )

                if base_score is None:
                    base_score = self._get_base_score(
                        rule
                    )

                try:
                    base_score = float(
                        base_score
                    )

                except (
                    TypeError,
                    ValueError
                ):
                    base_score = 1.0

                context = {
                    "edge":
                        action.get(
                            "edge"
                        ),

                    "source_change_index":
                        change_index,

                    "rule_object":
                        rule
                }

                action_step = ActionStep(
                    action_name=
                        str(
                            action_name
                        ),

                    propagation_rule=
                        self._rule_to_plan_string(
                            rule
                        ),

                    base_score=
                        base_score,

                    context=
                        context,

                    notes={
                        "source":
                            "InteractionEngine"
                    }
                )

                primary_plan.add_action(
                    action_step
                )

            # Add the completed primary-change plan to the interaction plan.
            plan.add_primary_change(
                primary_plan
            )

        return plan

    # Public InteractionPlan Generation
    def generate_interaction_plans(
        self,
        propagation_rules_per_change: List[List],
        primary_changes: Optional[List[Dict]] = None
    ):
        """
        Generate InteractionPlan objects from propagation rules.

        Parameters
        ----------
        propagation_rules_per_change : List[List]

            Propagation rules grouped by primary structural change.

        primary_changes : Optional[List[Dict]]

            The original primary structural changes.

            Example:

                [
                    {
                        "type":
                            "EDGE_ADDITION",
                        "edge":
                            ("GDP", "SP500")
                    },

                    {
                        "type":
                            "EDGE_REMOVAL",
                        "edge":
                            ("Inflation", "SP500")
                    }
                ]

            Supplying this is recommended because InteractionPlan
            explicitly records the originating primary change.

        Returns
        -------
        List[InteractionPlan]

            Fully constructed interaction plans.
        """

        hypotheses = self.generate_interactions(
            propagation_rules_per_change = propagation_rules_per_change,
            primary_changes = primary_changes
        )

        plans = []

        for hypothesis in hypotheses:
            plan = (
                self._hypothesis_to_interaction_plan(
                    hypothesis=hypothesis,
                    primary_changes=primary_changes
                )
            )

            plans.append(
                plan
            )

        return plans

    def build_interaction_plans(
        self,
        propagation_rules_per_change: List[List],
        primary_changes: Optional[List[Dict]] = None
    ):
        """
        Alias for generate_interaction_plans().

        Provided for readability at call sites where the intention
        is explicitly to construct executable interaction plans.
        """

        return self.generate_interaction_plans(
            propagation_rules_per_change=
                propagation_rules_per_change,

            primary_changes=
                primary_changes
        )

def print_hypotheses(
    hypotheses: List[InteractionHypothesis]
):
    print("=" * 70)
    print("Interaction Hypotheses")
    print("=" * 70)
    print()

    if not hypotheses:
        print(
            "No valid interaction hypotheses generated."
        )

        print()
        print("=" * 70)

        return

    for i, hypothesis in enumerate(
        hypotheses,
        start=1
    ):
        print(
            f"Hypothesis {i}"
        )

        print(
            f"Name  : "
            f"{hypothesis.hypothesis_name}"
        )

        print(
            f"Level : "
            f"{hypothesis.interaction_level}"
        )

        print(
            f"Score : "
            f"{hypothesis.base_score:.3f}"
        )

        print(
            "Actions"
        )

        for action in hypothesis.actions:
            print(
                f"   - "
                f"{action.get('action')}"
            )

            edge = action.get(
                "edge"
            )

            if edge is not None:
                print(
                    f"     Edge : "
                    f"{edge}"
                )

            source_change_index = action.get(
                "source_change_index"
            )

            if source_change_index is not None:
                print(
                    f"     Source Change : "
                    f"{source_change_index}"
                )

        print()

    print("=" * 70)

def print_interaction_plans(
    plans
):
    from src.graph_evolution.interaction_plan import (
        print_interaction_plan
    )

    for plan in plans:
        print_interaction_plan(
            plan
        )

# Testing
if __name__ == "__main__":

    from src.graph_evolution.propagation_rules import (
        PropagationRuleEngine
    )

    engine = PropagationRuleEngine()

    interaction_engine = InteractionEngine()

    changes = [
        {
            "type":
                "EDGE_ADDITION",

            "edge":
                ("GDP", "SP500")
        },

        {
            "type":
                "EDGE_REMOVAL",

            "edge":
                ("Inflation", "SP500")
        }
    ]

    propagation_rules = []

    for change in changes:
        propagation_rules.append(
            engine.generate_rules(
                change
            )
        )

    # Generate intermediate hypotheses.
    hypotheses = (
        interaction_engine.generate_interactions(
            propagation_rules_per_change = propagation_rules,
            primary_changes = changes
        )
    )

    print_hypotheses(
        hypotheses
    )

    # Generate final InteractionPlan objects.
    # The original primary changes are supplied so that each
    # PrimaryChangePlan contains the actual originating change.
    plans = (
        interaction_engine.generate_interaction_plans(
            propagation_rules_per_change=
                propagation_rules,

            primary_changes=
                changes
        )
    )

    print_interaction_plans(
        plans
    )