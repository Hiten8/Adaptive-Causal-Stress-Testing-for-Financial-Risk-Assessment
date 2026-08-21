"""
Regime Profiler

Purpose
-------
Transforms an HMM regime assignment into a dynamic
Regime Profile based on characteristics rather than
fixed regime IDs.

This allows the Graph Evolution Engine to adapt
automatically even if the HMM discovers a different
number of hidden regimes.
"""

from dataclasses import dataclass, field
from typing import Dict
from src.core.causal_state import CausalState

# Regime Profile
@dataclass
class RegimeProfile:
    dominant_regime: int
    confidence_level: str
    uncertainty_level: str
    volatility_level: str
    transition_risk: str
    characteristics: Dict = field(default_factory=dict)

# Regime Profiler
class RegimeProfiler:
    def __init__(self):
        pass

    # Public API
    def profile(
        self,
        state: CausalState
    ) -> RegimeProfile:

        confidence = self._confidence_level(
            state.average_confidence
        )

        uncertainty = self._uncertainty_level(
            state.average_entropy
        )

        volatility = self._volatility_level(
            state
        )

        transition = self._transition_risk(
            confidence,
            uncertainty
        )

        characteristics = {
            "dominant_probability":
                max(state.regime_distribution.values()),

            "num_possible_regimes":
                len(state.regime_distribution),

            "graph_density":
                state.graph_metrics.get(
                    "density",
                    None
                ),

            "average_degree":
                state.graph_metrics.get(
                    "average_degree",
                    None
                )
        }

        return RegimeProfile(
            dominant_regime=state.dominant_regime,
            confidence_level=confidence,
            uncertainty_level=uncertainty,
            volatility_level=volatility,
            transition_risk=transition,
            characteristics=characteristics
        )

    # Confidence
    def _confidence_level(
        self,
        confidence
    ):

        if confidence >= 0.90:
            return "VERY_HIGH"

        elif confidence >= 0.75:
            return "HIGH"

        elif confidence >= 0.60:
            return "MEDIUM"

        return "LOW"

    # Entropy
    def _uncertainty_level(
        self,
        entropy
    ):

        if entropy <= 0.20:
            return "VERY_LOW"

        elif entropy <= 0.40:
            return "LOW"

        elif entropy <= 0.70:
            return "MEDIUM"

        return "HIGH"

    # Volatility
    def _volatility_level(
        self,
        state
    ):
        """
        Placeholder.

        Later this can use:

        - VIX
        - Rolling volatility
        - Market stress index

        For now we approximate using entropy.
        """

        entropy = state.average_entropy
        if entropy >= 0.70:
            return "HIGH"

        elif entropy >= 0.40:
            return "MEDIUM"

        return "LOW"

    # Transition Risk
    def _transition_risk(
        self,
        confidence,
        uncertainty
    ):

        if confidence == "LOW":
            return "HIGH"

        if uncertainty == "HIGH":
            return "HIGH"

        if confidence == "MEDIUM":
            return "MEDIUM"

        return "LOW"

# Testing
if __name__ == "__main__":

    from src.storage.causal_state_repository import (
        CausalStateRepository
    )

    repository = CausalStateRepository()
    profiler = RegimeProfiler()
    state = repository.load_all()[0]
    profile = profiler.profile(state)

    print("=" * 70)
    print("Regime Profile")
    print("=" * 70)
    print(f"Dominant Regime   : {profile.dominant_regime}")
    print(f"Confidence        : {profile.confidence_level}")
    print(f"Uncertainty       : {profile.uncertainty_level}")
    print(f"Volatility        : {profile.volatility_level}")
    print(f"Transition Risk   : {profile.transition_risk}")
    print()
    print("Characteristics")

    for key, value in profile.characteristics.items():
        print(f"  {key:<25}: {value}")

    print("=" * 70)