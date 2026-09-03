"""Orchestration for statistically gated, uncertainty-aware graph updates."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd

from src.core.causal_state import CausalState
from src.stress_testing.bootstrap import BootstrapCandidateGenerator, BootstrapEnsemble
from src.stress_testing.change_significance import BootstrapChangeTest, ChangeSignificanceResult


@dataclass
class UpdateReview:
    gate_accepted: bool
    statistically_significant: bool
    significance: ChangeSignificanceResult
    ensemble: Optional[BootstrapEnsemble]
    selected_candidate: Optional[CausalState]
    audit_record: Dict[str, object]


class AdaptiveUpdateWorkflow:
    """Combine the existing gate with bootstrap-calibrated statistical evidence.

    This class intentionally does not overwrite the active graph.  The caller can
    inspect ``audit_record`` and explicitly persist ``selected_candidate`` through
    ``ActiveGraphManager`` after approval.
    """

    def __init__(self, change_test: Optional[BootstrapChangeTest] = None, generator: Optional[BootstrapCandidateGenerator] = None):
        self.change_test = change_test or BootstrapChangeTest()
        self.generator = generator or BootstrapCandidateGenerator()

    def review(
        self,
        deployed_state: CausalState,
        observed_state: CausalState,
        stable_null_states: Iterable[CausalState],
        gate_decision: Dict[str, object],
        observed_window: pd.DataFrame,
        window_metadata: Dict[str, object],
    ) -> UpdateReview:
        significance = self.change_test.assess(deployed_state, observed_state, stable_null_states)
        gate_accepted = bool(gate_decision.get("ShouldUpdate", False))
        should_update = gate_accepted and significance.significant
        ensemble = self.generator.generate(observed_window, window_metadata) if should_update else None
        selected = ensemble.candidates[0] if ensemble and ensemble.candidates else None
        audit_record = {
            "gate_accepted": gate_accepted,
            "observed_distance": significance.observed_distance,
            "bootstrap_threshold": significance.null_threshold,
            "empirical_p_value": significance.empirical_p_value,
            "statistically_significant": significance.significant,
            "update_approved": should_update,
            "candidate_count": len(ensemble.candidates) if ensemble else 0,
            "selected_candidate_probability": selected.candidate_probability if selected else None,
        }
        return UpdateReview(gate_accepted, significance.significant, significance, ensemble, selected, audit_record)
