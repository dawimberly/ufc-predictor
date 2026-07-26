"""Model probability → confidence score / label."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceBands:
    high: float = 0.65
    medium: float = 0.58


def confidence_score(prob: float) -> float:
    """
    Map win probability to [0, 1] confidence.

    0.5 → 0; 1.0 → 1. Symmetric around a coin flip.
    """
    p = float(prob)
    p = min(1.0, max(0.0, p))
    return abs(p - 0.5) * 2.0


def confidence_label(prob: float, bands: ConfidenceBands | None = None) -> str:
    bands = bands or ConfidenceBands()
    p = max(float(prob), 1.0 - float(prob))  # strength of pick side
    if p >= bands.high:
        return "High"
    if p >= bands.medium:
        return "Medium"
    return "Low"


def attach_confidence(prob: float) -> dict[str, float | str]:
    return {
        "prob": float(prob),
        "confidence": confidence_score(prob),
        "confidence_label": confidence_label(prob),
    }
