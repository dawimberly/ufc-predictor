"""Sport plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Matchup:
    event: str
    sport: str
    selection_a: str
    selection_b: str
    start_time: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPick:
    matchup: Matchup
    selection: str
    prob: float
    features: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


class SportAdapter(ABC):
    """Implement per sport (UFC first, then NFL/NBA/etc.)."""

    sport: str = "generic"

    @abstractmethod
    def upcoming_matchups(self) -> list[Matchup]:
        raise NotImplementedError

    @abstractmethod
    def score_matchup(self, matchup: Matchup) -> ModelPick:
        raise NotImplementedError

    def settle_results(self) -> list[dict[str, Any]]:
        """Return recently completed results for bank settlement (optional)."""
        return []
