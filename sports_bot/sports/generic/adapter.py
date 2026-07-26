"""Generic / multi-sport placeholder adapter."""

from __future__ import annotations

from sports_bot.sports.base import Matchup, ModelPick, SportAdapter


class GenericAdapter(SportAdapter):
    sport = "generic"

    def __init__(self, sport_name: str = "generic") -> None:
        self.sport = sport_name

    def upcoming_matchups(self) -> list[Matchup]:
        return []

    def score_matchup(self, matchup: Matchup) -> ModelPick:
        return ModelPick(matchup=matchup, selection=matchup.selection_a, prob=0.5, reasons=["stub"])
