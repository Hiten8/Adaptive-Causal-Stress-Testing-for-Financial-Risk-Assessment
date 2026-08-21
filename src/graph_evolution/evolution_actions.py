"""
Evolution Actions

Purpose
-------
Implements reusable graph evolution actions.

These actions are independent of the observed primary
change. They simply modify a graph in a plausible way.

GraphEvolutionEngine decides WHICH actions to execute.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import networkx as nx
from src.core.causal_state import CausalState

# Base Evolution Action
class EvolutionAction(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(
        self,
        state: CausalState,
        context: dict
    ):
        pass


# Apply Added Edge
class ApplyAddedEdge(EvolutionAction):

    def __init__(self):
        super().__init__("ApplyAddedEdge")

    def apply(self, state, context):
        candidate = deepcopy(state)
        u, v = context["edge"]

        if not candidate.graph.has_edge(u, v):
            candidate.graph.add_edge(u, v)

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Apply Removed Edge
class ApplyRemovedEdge(EvolutionAction):

    def __init__(self):
        super().__init__("ApplyRemovedEdge")

    def apply(self, state, context):
        candidate = deepcopy(state)
        u, v = context["edge"]

        if candidate.graph.has_edge(u, v):
            candidate.graph.remove_edge(u, v)

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Apply Weight Change
class ApplyWeightChange(EvolutionAction):

    def __init__(self):
        super().__init__("ApplyWeightChange")

    def apply(self, state, context):
        candidate = deepcopy(state)
        u, v = context["edge"]

        if candidate.graph.has_edge(u, v):
            candidate.graph[u][v]["weight"] = context["new_weight"]

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Strengthen Downstream
class StrengthenDownstream(EvolutionAction):

    def __init__(self):
        super().__init__("StrengthenDownstream")

    def apply(self, state, context):
        candidate = deepcopy(state)
        _, node = context["edge"]

        for _, child in candidate.graph.out_edges(node):
            weight = candidate.graph[node][child].get("weight", 1.0)
            candidate.graph[node][child]["weight"] = weight * 1.10

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Strengthen Incoming
class StrengthenIncoming(EvolutionAction):

    def __init__(self):
        super().__init__("StrengthenIncoming")

    def apply(self, state, context):
        candidate = deepcopy(state)
        _, node = context["edge"]

        for parent, _ in candidate.graph.in_edges(node):
            weight = candidate.graph[parent][node].get("weight", 1.0)
            candidate.graph[parent][node]["weight"] = weight * 1.10

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Create Alternative Path
class CreateAlternativePath(EvolutionAction):

    def __init__(self):
        super().__init__("CreateAlternativePath")

    def apply(self, state, context):
        candidate = deepcopy(state)
        u, v = context["edge"]
        successors = list(candidate.graph.successors(v))

        if successors:
            candidate.graph.add_edge(u, successors[0])

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Reconnect Isolated Node
class ReconnectNode(EvolutionAction):

    def __init__(self):
        super().__init__("ReconnectNode")

    def apply(self, state, context):
        candidate = deepcopy(state)
        node = context["node"]
        degrees = dict(candidate.graph.degree())

        if len(degrees) == 0:
            return candidate

        target = max(
            degrees,
            key=degrees.get
        )

        if node != target:
            candidate.graph.add_edge(
                node,
                target
            )

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Reduce Outgoing Influence
class ReduceOutgoingInfluence(EvolutionAction):

    def __init__(self):
        super().__init__("ReduceOutgoingInfluence")

    def apply(self, state, context):
        candidate = deepcopy(state)
        node = context["edge"][0]

        for _, child in candidate.graph.out_edges(node):
            weight = candidate.graph[node][child].get("weight", 1.0)
            candidate.graph[node][child]["weight"] = weight * 0.85

        candidate.notes["EvolutionAction"] = self.name

        return candidate

# Action Factory
def get_action(action_name):
    actions = {
        "ApplyAddedEdge":
            ApplyAddedEdge(),

        "ApplyRemovedEdge":
            ApplyRemovedEdge(),

        "ApplyWeightChange":
            ApplyWeightChange(),

        "StrengthenDownstream":
            StrengthenDownstream(),

        "StrengthenIncoming":
            StrengthenIncoming(),

        "CreateAlternativePath":
            CreateAlternativePath(),

        "ReconnectNode":
            ReconnectNode(),

        "ReduceOutgoingInfluence":
            ReduceOutgoingInfluence()
    }

    return actions.get(action_name)