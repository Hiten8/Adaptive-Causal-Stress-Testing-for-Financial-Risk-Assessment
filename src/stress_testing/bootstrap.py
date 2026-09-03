"""Moving-block bootstrap ensembles for regime-specific causal graphs."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.causal.pcmci_engine import run_pcmci_frame
from src.core.causal_state import CausalState


GraphSignature = Tuple[Tuple[str, str, float], ...]


@dataclass
class BootstrapEnsemble:
    """Distinct discovered graphs and their empirical bootstrap probabilities."""

    candidates: List[CausalState]
    probabilities: Dict[GraphSignature, float]
    draws: int
    block_size: int


class BootstrapCandidateGenerator:
    """Generate candidate causal graphs without breaking temporal ordering.

    Moving-block resampling is used instead of independent row resampling because
    PCMCI consumes time-series data.  Candidate probability is the observed graph
    frequency across bootstrap draws, not an arbitrary ranking score.
    """

    def __init__(self, n_resamples: int = 100, block_size: Optional[int] = None, seed: int = 42):
        if n_resamples < 1:
            raise ValueError("n_resamples must be at least one.")
        self.n_resamples = n_resamples
        self.block_size = block_size
        self.seed = seed

    @staticmethod
    def graph_signature(state: CausalState, precision: int = 6) -> GraphSignature:
        return tuple(sorted(
            (str(source), str(target), round(float(data.get("weight", 1.0)), precision))
            for source, target, data in state.graph.edges(data=True)
        ))

    def _resample(self, frame: pd.DataFrame, rng) -> pd.DataFrame:
        length = len(frame)
        if length < 4:
            raise ValueError("At least four observations are required for bootstrap causal discovery.")
        block_size = self.block_size or max(2, int(length ** 0.5))
        starts = rng.integers(0, length, size=(length + block_size - 1) // block_size)
        indices = []
        for start in starts:
            indices.extend((start + offset) % length for offset in range(block_size))
        return frame.iloc[indices[:length]].reset_index(drop=True)

    def generate(
        self,
        frame: pd.DataFrame,
        window_metadata: Dict,
        discovery_fn: Callable[[pd.DataFrame, Dict], CausalState] = run_pcmci_frame,
    ) -> BootstrapEnsemble:
        """Run discovery over moving-block resamples and aggregate distinct graphs."""
        rng = np.random.default_rng(self.seed)
        counts: Counter = Counter()
        representatives: Dict[GraphSignature, CausalState] = {}

        for draw in range(self.n_resamples):
            state = discovery_fn(self._resample(frame, rng), window_metadata)
            signature = self.graph_signature(state)
            counts[signature] += 1
            if signature not in representatives:
                representatives[signature] = deepcopy(state)
                representatives[signature].notes = dict(representatives[signature].notes)
                representatives[signature].notes["bootstrap_first_draw"] = draw

        candidates = []
        probabilities = {}
        for signature, count in counts.most_common():
            candidate = representatives[signature]
            probability = count / self.n_resamples
            candidate.candidate_probability = probability
            candidate.notes["bootstrap_frequency"] = count
            candidate.notes["bootstrap_draws"] = self.n_resamples
            candidates.append(candidate)
            probabilities[signature] = probability

        return BootstrapEnsemble(
            candidates=candidates,
            probabilities=probabilities,
            draws=self.n_resamples,
            block_size=self.block_size or max(2, int(len(frame) ** 0.5)),
        )
