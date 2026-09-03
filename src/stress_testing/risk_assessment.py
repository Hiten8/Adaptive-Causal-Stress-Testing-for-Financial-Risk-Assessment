"""Counterfactual interventions and probability-weighted risk aggregation."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from src.core.causal_state import CausalState


@dataclass(frozen=True)
class StressScenario:
    name: str
    interventions: Dict[str, float]
    description: str = ""


@dataclass
class CandidateOutcome:
    probability: float
    loss: float
    values: Dict[str, float]


@dataclass
class StressTestResult:
    scenario: StressScenario
    point_estimate: float
    expected_loss: float
    plausible_worst_case: float
    outcomes: List[CandidateOutcome]


class StressTestEngine:
    """Apply linear SCM-style interventions to an ensemble of causal graphs.

    The iterative propagation accommodates the lagged/cyclic graphs that PCMCI can
    return.  It is deliberately transparent: each directed edge contributes
    ``weight * parent_change`` to its child on every propagation round.
    """

    def __init__(self, propagation_steps: int = 3, probability_floor: float = 0.05):
        self.propagation_steps = propagation_steps
        self.probability_floor = probability_floor

    def simulate(self, state: CausalState, scenario: StressScenario) -> Dict[str, float]:
        values = {str(node): 0.0 for node in state.graph.nodes()}
        for variable, shock in scenario.interventions.items():
            if variable not in values:
                raise ValueError(f"Intervention variable '{variable}' is not in the causal graph.")
            values[variable] = float(shock)

        fixed = set(scenario.interventions)
        for _ in range(self.propagation_steps):
            next_values = dict(values)
            for source, target, data in state.graph.edges(data=True):
                if target not in fixed:
                    next_values[str(target)] += float(data.get("weight", 1.0)) * values[str(source)]
            values = next_values
        return values

    @staticmethod
    def _loss(values: Dict[str, float], loss_variable: str, loss_scale: float) -> float:
        if loss_variable not in values:
            raise ValueError(f"Loss variable '{loss_variable}' is not in the causal graph.")
        # A negative return represents a positive portfolio loss.
        return max(0.0, -values[loss_variable] * loss_scale)

    def evaluate(
        self,
        candidates: Iterable[CausalState],
        scenario: StressScenario,
        loss_variable: str,
        loss_scale: float = 1.0,
    ) -> StressTestResult:
        candidates = list(candidates)
        if not candidates:
            raise ValueError("At least one candidate graph is required for stress testing.")
        raw_probabilities = np.array([max(0.0, float(state.candidate_probability)) for state in candidates])
        probabilities = raw_probabilities / raw_probabilities.sum() if raw_probabilities.sum() else np.full(len(candidates), 1 / len(candidates))
        outcomes = []
        for state, probability in zip(candidates, probabilities):
            values = self.simulate(state, scenario)
            outcomes.append(CandidateOutcome(float(probability), self._loss(values, loss_variable, loss_scale), values))

        expected = sum(item.probability * item.loss for item in outcomes)
        likely = [item.loss for item in outcomes if item.probability >= self.probability_floor]
        most_likely = max(outcomes, key=lambda item: item.probability)
        return StressTestResult(
            scenario=scenario,
            point_estimate=most_likely.loss,
            expected_loss=float(expected),
            plausible_worst_case=max(likely) if likely else max(item.loss for item in outcomes),
            outcomes=outcomes,
        )
