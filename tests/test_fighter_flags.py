"""Tests for manual fighter integrity flags."""

from __future__ import annotations

import pandas as pd

from src.fighter_flags import (
    format_flag_badge,
    lookup_fighter_flag,
    reload_fighter_flags,
    should_skip_fight,
)
from src.strategy import StrategyConfig, extract_bet_candidates


def test_sutherland_and_montanha_flagged():
    reload_fighter_flags()
    assert lookup_fighter_flag("Louie Sutherland") is not None
    assert lookup_fighter_flag("José Montanha") is not None
    assert lookup_fighter_flag("Jose Luiz") is not None
    skip, detail = should_skip_fight("Louie Sutherland", "Jose Montanha")
    assert skip is True
    assert "Louie Sutherland" in detail or "Jose Montanha" in detail
    badge = format_flag_badge("Louie Sutherland", "Jose Montanha")
    assert badge and "FLAG" in badge


def test_extract_bet_skips_flagged_fight(monkeypatch):
    reload_fighter_flags()
    row = pd.Series(
        {
            "fight_id": "hw_flag_test",
            "fighter_1": "Louie Sutherland",
            "fighter_2": "Jose Montanha",
            "prob_f1_win": 0.72,
            "prob_f2_win": 0.28,
            "odds_f1": 1.55,
            "odds_f2": 2.50,
            "ensemble_disagreement": 0.001,
            "interval_width": 0.10,
            "confidence_label": "high",
        }
    )
    # Ensure odds helpers see decimals if needed
    monkeypatch.setattr(
        "src.strategy.market_probs",
        lambda _r: (0.60, 0.40),
        raising=False,
    )
    monkeypatch.setattr(
        "src.strategy.fight_decimal_odds",
        lambda _r: (1.55, 2.50),
        raising=False,
    )
    cand = extract_bet_candidates(
        row,
        config=StrategyConfig(min_edge=0.02, min_model_prob=0.50),
        apply_uncertainty_gates=False,
    )
    assert cand is None
