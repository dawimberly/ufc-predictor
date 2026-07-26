"""Core betting primitives."""

from sports_bot.core.confidence import attach_confidence, confidence_label, confidence_score
from sports_bot.core.kelly import KellyConfig, kelly_stake, raw_kelly

__all__ = [
    "KellyConfig",
    "kelly_stake",
    "raw_kelly",
    "attach_confidence",
    "confidence_label",
    "confidence_score",
]
