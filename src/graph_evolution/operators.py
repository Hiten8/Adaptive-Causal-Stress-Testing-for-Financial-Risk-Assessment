"""
Graph Evolution Operators

Purpose
-------
Implements graph evolution operators that generate
SECONDARY structural changes from PRIMARY observed changes.

Each operator receives:

    Stored CausalState
    Current PCMCI CausalState
    Primary Change

and returns:

    List[CausalState]

Important
---------
The operators implemented here are currently PROVISIONAL
GRAPH HEURISTICS.

They are intended to make the complete pipeline executable:

    Primary Change
          |
          v
    Propagation Rule
          |
          v
    Evolution Operator
          |
          v
    Candidate CausalState

The heuristics can later be replaced with financially
meaningful graph-evolution mechanisms without changing
the surrounding architecture.
"""

from copy import deepcopy
from abc import ABC, abstractmethod
import networkx as nx
from src.core.causal_state import CausalState

class BaseOperator(ABC):
    @abstractmethod
    def apply(
        self,
        stored_state: CausalState,
        current_state: CausalState,
        primary_change: dict
    ):
        pass

def _copy_candidate(
    stored_state,
    graph,
    operator_name
):
    """
    Create a candidate CausalState from the stored state.
    """

    candidate = deepcopy(
        stored_state
    )
    candidate.graph = graph
    candidate.notes[
        "EvolutionOperator"
    ] = operator_name

    return candidate


def _get_edge_weight(
    graph,
    u,
    v,
    default=1.0
):
    """
    Safely retrieve an edge weight.

    The existing project does not require every edge to have
    a weight, so a default value is used when necessary.
    """

    if not graph.has_edge(
        u,
        v
    ):
        return default

    weight = graph[u][v].get(
        "weight",
        default
    )

    try:
        return float(
            weight
        )

    except (
        TypeError,
        ValueError
    ):
        return default


def _set_edge_weight(
    graph,
    u,
    v,
    weight
):
    """
    Set the weight of an existing edge.
    """

    if graph.has_edge(u, v):
        graph[u][v]["weight"] = weight

def _strengthen_weight(
    graph,
    u,
    v,
    factor=1.10
):
    """
    Provisional weight-strengthening heuristic.

    Existing edge weight is increased by the supplied factor.
    """

    if not graph.has_edge(u, v):
        return

    current_weight = _get_edge_weight(
        graph,
        u,
        v
    )

    _set_edge_weight(
        graph,
        u,
        v,
        current_weight * factor
    )


def _weaken_weight(
    graph,
    u,
    v,
    factor=0.90
):
    """
    Provisional weight-reduction heuristic.

    Existing edge weight is reduced by the supplied factor.
    """

    if not graph.has_edge(u, v):
        return

    current_weight = _get_edge_weight(
        graph,
        u,
        v
    )

    _set_edge_weight(
        graph,
        u,
        v,
        current_weight * factor
    )


def _add_edge_if_missing(
    graph,
    u,
    v
):
    """
    Add a directed edge only if it does not already exist.

    Self-loops are explicitly prevented.
    """

    if u == v:
        return False

    if graph.has_edge(
        u,
        v
    ):
        return False

    graph.add_edge(
        u,
        v
    )

    return True

# Edge Addition Propagation
class EdgeAdditionOperator(
    BaseOperator
):

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):

        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

        candidate = _copy_candidate(
            stored_state,
            graph,
            "EdgeAddition"
        )

        candidates.append(
            candidate
        )

        return candidates

# Strengthen Downstream
class StrengthenDownstreamOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    For a newly added edge u -> v, strengthen the existing
    outgoing influence of v.

    In this prototype, "strengthen" means increasing the
    weights of v's existing outgoing edges.

    If no outgoing weighted edges exist, the graph structure
    is left unchanged.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        # Ensure the primary edge exists.
        if not graph.has_edge(
            u,
            v
        ):
            graph.add_edge(
                u,
                v
            )

        # Strengthen existing downstream edges.
        downstream_nodes = list(
            graph.successors(
                v
            )
        )

        for node in downstream_nodes:
            if node == u:
                continue

            _strengthen_weight(
                graph,
                v,
                node
            )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "StrengthenDownstream"
        )

        candidates.append(
            candidate
        )

        return candidates

# Strengthen Upstream
class StrengthenUpstreamOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    For an added edge u -> v, strengthen the existing
    incoming edges of u.

    This represents stronger upstream influence feeding
    into the newly connected source node.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):

        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        # Ensure the primary edge exists.
        if not graph.has_edge(
            u,
            v
        ):
            graph.add_edge(
                u,
                v
            )

        upstream_nodes = list(
            graph.predecessors(
                u
            )
        )

        for node in upstream_nodes:
            if node == v:
                continue

            _strengthen_weight(
                graph,
                node,
                u
            )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "StrengthenUpstream"
        )

        candidates.append(
            candidate
        )

        return candidates

# Alternative Path
class AlternativePathOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    For an added edge u -> v, attempt to create an alternative
    incoming path to v through one of u's upstream nodes.

    Example:

        A -> U -> V

    may produce:

        A -> V

    if A -> V does not already exist.

    Only one deterministic alternative connection is created.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):

        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if not graph.has_edge(
            u,
            v
        ):
            graph.add_edge(
                u,
                v
            )

        upstream_nodes = sorted(
            list(
                graph.predecessors(
                    u
                )
            ),
            key=str
        )

        alternative_created = False

        for node in upstream_nodes:
            if node == v:
                continue

            if _add_edge_if_missing(
                graph,
                node,
                v
            ):
                alternative_created = True
                break

        # If no upstream alternative is available, try a
        # downstream alternative from v.
        if not alternative_created:
            downstream_nodes = sorted(
                list(
                    graph.successors(
                        v
                    )
                ),
                key=str
            )

            for node in downstream_nodes:
                if node == u:
                    continue

                if _add_edge_if_missing(
                    graph,
                    u,
                    node
                ):
                    alternative_created = True
                    break

        candidate = _copy_candidate(
            stored_state,
            graph,
            "AlternativePath"
        )

        candidates.append(
            candidate
        )

        return candidates

# Edge Removal Propagation
class EdgeRemovalOperator(
    BaseOperator
):

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if graph.has_edge(
            u,
            v
        ):
            graph.remove_edge(
                u,
                v
            )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "EdgeRemoval"
        )

        candidates.append(
            candidate
        )

        return candidates

# Strengthen Remaining Parents
class StrengthenIncomingOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    After removing u -> v, strengthen the remaining incoming
    edges of v.

    This represents compensation by the remaining parents
    of the affected node.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if graph.has_edge(
            u,
            v
        ):
            graph.remove_edge(
                u,
                v
            )

        remaining_parents = list(
            graph.predecessors(
                v
            )
        )

        for parent in remaining_parents:
            _strengthen_weight(
                graph,
                parent,
                v
            )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "StrengthenIncoming"
        )

        candidates.append(
            candidate
        )

        return candidates

# Bypass Connection
class BypassConnectionOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    If u -> v is removed, connect u directly to one of v's
    existing downstream nodes.

    Example:

        U -> V -> W

    becomes:

        U -> W

    while U -> V is removed.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if graph.has_edge(
            u,
            v
        ):
            graph.remove_edge(
                u,
                v
            )

        downstream_nodes = sorted(
            list(
                graph.successors(
                    v
                )
            ),
            key=str
        )

        for node in downstream_nodes:
            if node == u:
                continue

            if _add_edge_if_missing(
                graph,
                u,
                node
            ):
                break

        candidate = _copy_candidate(
            stored_state,
            graph,
            "BypassConnection"
        )

        candidates.append(
            candidate
        )

        return candidates

# Reduce Downstream Influence
class ReduceOutgoingOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    After removal of u -> v, reduce the weights of v's
    remaining outgoing edges.

    This represents reduced downstream influence.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if graph.has_edge(
            u,
            v
        ):
            graph.remove_edge(
                u,
                v
            )

        downstream_nodes = list(
            graph.successors(
                v
            )
        )

        for node in downstream_nodes:
            _weaken_weight(
                graph,
                v,
                node
            )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "ReduceOutgoing"
        )

        candidates.append(
            candidate
        )

        return candidates

# Weight Change Propagation
class WeightChangeOperator(
    BaseOperator
):

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        if graph.has_edge(
            u,
            v
        ):
            graph[u][v]["weight"] = primary_change[
                "new_weight"
            ]

        candidate = _copy_candidate(
            stored_state,
            graph,
            "WeightChange"
        )

        candidates.append(
            candidate
        )

        return candidates

# Neighbour Strengthening
class NeighbourStrengtheningOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    For a weight change on u -> v, strengthen the weights
    of the immediate neighbouring edges of u and v.

    Only existing edges are modified.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        # Apply primary weight change first.
        if graph.has_edge(
            u,
            v
        ):
            graph[u][v]["weight"] = primary_change[
                "new_weight"
            ]

        # Strengthen incoming neighbours of u.
        for node in list(
            graph.predecessors(
                u
            )
        ):
            _strengthen_weight(
                graph,
                node,
                u
            )

        # Strengthen outgoing neighbours of v.
        for node in list(
            graph.successors(
                v
            )
        ):
            _strengthen_weight(
                graph,
                v,
                node
            )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "NeighbourStrengthening"
        )

        candidates.append(
            candidate
        )

        return candidates

# Competing Edge Reduction
class CompetingReductionOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    For a changed edge u -> v, reduce the weights of other
    incoming edges into v.

    The changed edge itself is not reduced.
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        u, v = primary_change[
            "edge"
        ]

        # Apply primary weight change.
        if graph.has_edge(
            u,
            v
        ):
            graph[u][v]["weight"] = primary_change[
                "new_weight"
            ]

        # Reduce competing parents.
        for parent in list(
            graph.predecessors(
                v
            )
        ):

            if parent == u:
                continue

            _weaken_weight(
                graph,
                parent,
                v
            )

        candidate = _copy_candidate(
            stored_state,
        graph,
            "CompetingReduction"
        )

        candidates.append(
            candidate
        )

        return candidates

# Isolation Recovery
class IsolationRecoveryOperator(
    BaseOperator
):

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        node = primary_change[
            "node"
        ]

        # Connect isolated node to strongest node.
        # This remains a provisional heuristic.
        # The strongest node is approximated using degree.

        degrees = dict(
            graph.degree()
        )

        if degrees:
            target = max(
                degrees,
                key=degrees.get
            )

            if target != node:
                graph.add_edge(
                    node,
                    target
                )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "IsolationRecovery"
        )

        candidates.append(
            candidate
        )

        return candidates

# Strongest Parent Recovery
class StrongParentRecoveryOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    Reconnect an isolated node to the node having the highest
    degree and use that node as the parent.

        strongest_node -> isolated_node
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        node = primary_change[
            "node"
        ]

        degrees = dict(
            graph.degree()
        )

        # Exclude the isolated node itself.
        degrees.pop(
            node,
            None
        )

        if degrees:
            strongest_parent = max(
                degrees,
                key=degrees.get
            )

            if (
                strongest_parent != node
                and
                not graph.has_edge(
                    strongest_parent,
                    node
                )
            ):

                graph.add_edge(
                    strongest_parent,
                    node
                )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "StrongParentRecovery"
        )

        candidates.append(
            candidate
        )

        return candidates

# Strongest Child Recovery
class StrongChildRecoveryOperator(
    BaseOperator
):
    """
    Provisional heuristic:

    Reconnect an isolated node to the node having the highest
    degree and use the isolated node as the parent.

        isolated_node -> strongest_node
    """

    def apply(
        self,
        stored_state,
        current_state,
        primary_change
    ):
        candidates = []

        graph = deepcopy(
            stored_state.graph
        )

        node = primary_change[
            "node"
        ]

        degrees = dict(
            graph.degree()
        )

        degrees.pop(
            node,
            None
        )

        if degrees:
            strongest_child = max(
                degrees,
                key=degrees.get
            )

            if (
                strongest_child != node
                and
                not graph.has_edge(
                    node,
                    strongest_child
                )
            ):

                graph.add_edge(
                    node,
                    strongest_child
                )

        candidate = _copy_candidate(
            stored_state,
            graph,
            "StrongChildRecovery"
        )

        candidates.append(
            candidate
        )

        return candidates

# Operator Factory
def get_operator(
    operator_name
):
    """
    Return the graph-evolution operator corresponding to
    a propagation-rule operator name.

    Every operator currently defined in propagation_rules.py
    is registered here.

    Unknown operators return None rather than raising an
    exception, preserving the previous factory behaviour.
    """

    mapping = {
        # Edge Addition
        "EDGE_ADDITION":
            EdgeAdditionOperator(),

        "EdgeAddition":
            EdgeAdditionOperator(),

        "StrengthenDownstream":
            StrengthenDownstreamOperator(),

        "StrengthenUpstream":
            StrengthenUpstreamOperator(),

        "AlternativePath":
            AlternativePathOperator(),

        # Edge Removal
        "EDGE_REMOVAL":
            EdgeRemovalOperator(),

        "EdgeRemoval":
            EdgeRemovalOperator(),

        "StrengthenIncoming":
            StrengthenIncomingOperator(),

        "BypassConnection":
            BypassConnectionOperator(),

        "ReduceOutgoing":
            ReduceOutgoingOperator(),

        # Weight Change
        "WEIGHT_CHANGE":
            WeightChangeOperator(),

        "WeightChange":
            WeightChangeOperator(),

        "NeighbourStrengthening":
            NeighbourStrengtheningOperator(),

        "CompetingReduction":
            CompetingReductionOperator(),

        # Isolation
        "ISOLATED_NODE":
            IsolationRecoveryOperator(),

        "IsolationRecovery":
            IsolationRecoveryOperator(),

        "StrongParentRecovery":
            StrongParentRecoveryOperator(),

        "StrongChildRecovery":
            StrongChildRecoveryOperator(),

    }

    return mapping.get(
        operator_name
    )

# Testing
if __name__ == "__main__":

    from src.graph_evolution.propagation_rules import (
        PropagationRuleEngine
    )

    engine = PropagationRuleEngine()

    test_changes = [
        {
            "type":
                "EDGE_ADDITION",

            "edge":
                (
                    "GDP",
                    "SP500"
                )
        },

        {
            "type":
                "EDGE_REMOVAL",

            "edge":
                (
                    "Inflation",
                    "SP500"
                )
        },

        {
            "type":
                "WEIGHT_CHANGE",

            "edge":
                (
                    "GDP",
                    "SP500"
                ),

            "old_weight":
                0.50,

            "new_weight":
                0.80
        },

        {
            "type":
                "ISOLATED_NODE",

            "node":
                "Inflation"
        }
    ]

    print("=" * 80)
    print("Graph Evolution Operators")
    print("=" * 80)
    print()

    for change in test_changes:
        print(
            f"Primary Change : "
            f"{change}"
        )

        rules = (
            engine.generate_rules(
                change
            )
        )

        for rule in rules:
            operator = get_operator(
                rule.operator
            )

            status = (
                "AVAILABLE"
                if operator is not None
                else "MISSING"
            )

            print(
                f"   {rule.operator:30}"
                f" -> {status}"
            )

        print()
    print("=" * 80)