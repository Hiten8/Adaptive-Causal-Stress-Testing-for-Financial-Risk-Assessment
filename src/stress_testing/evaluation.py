"""Evaluation helpers for static, naive-adaptive, and gated-adaptive policies."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class EvaluationResult:
    mean_absolute_error: float
    update_count: int
    false_update_rate: float


class AdaptiveModelEvaluator:
    """Score model policies against realized historical stress losses."""

    @staticmethod
    def evaluate(predicted_losses: Sequence[float], realized_losses: Sequence[float], updates: Sequence[bool], stable_periods: Sequence[bool]) -> EvaluationResult:
        if not (len(predicted_losses) == len(realized_losses) == len(updates) == len(stable_periods)):
            raise ValueError("All evaluation inputs must have equal length.")
        if not predicted_losses:
            raise ValueError("At least one evaluation observation is required.")
        errors = np.abs(np.asarray(predicted_losses, dtype=float) - np.asarray(realized_losses, dtype=float))
        stable_updates = [update for update, stable in zip(updates, stable_periods) if stable]
        return EvaluationResult(
            mean_absolute_error=float(errors.mean()),
            update_count=sum(bool(update) for update in updates),
            false_update_rate=(sum(bool(update) for update in stable_updates) / len(stable_updates)) if stable_updates else 0.0,
        )
