"""Tests for HA path-risk / drawdown controls."""

from __future__ import annotations

from src.high_accuracy_strategy import (
    card_risk_drawdown_multiplier,
    max_parlay_budget_share,
    stake_allocation_power,
)
from src.strategy import allocate_card_budget_pct


def test_path_risk_paper_vs_live():
    assert max_parlay_budget_share(live=False) == 0.40
    assert max_parlay_budget_share(live=True) == 0.30
    assert stake_allocation_power(live=True) < stake_allocation_power(live=False)


def test_drawdown_multiplier_steps():
    assert card_risk_drawdown_multiplier(0.0, live=False) == 1.0
    soft = card_risk_drawdown_multiplier(0.25, live=False)
    hard = card_risk_drawdown_multiplier(0.50, live=False)
    assert soft < 1.0
    assert hard <= soft


def test_parlay_share_cap_prefers_singles():
    tickets = [
        {
            "pick": "A",
            "is_parlay": False,
            "edge": 0.12,
            "prob": 0.75,
            "confidence": "high",
            "kelly_pct": 2.0,
            "decimal_odds": 1.9,
        },
        {
            "pick": "B+C",
            "is_parlay": True,
            "n_legs": 2,
            "edge": 0.20,
            "prob": 0.55,
            "confidence": "high",
            "kelly_pct": 3.0,
            "decimal_odds": 4.0,
            "combined_odds": 4.0,
        },
    ]
    out = allocate_card_budget_pct(tickets, 100.0, profile="paper", inplace=False)
    parlays = [t for t in out if t.get("is_parlay")]
    singles = [t for t in out if not t.get("is_parlay")]
    parlay_pct = sum(float(t["stake_pct"]) for t in parlays)
    assert parlay_pct <= 40.0 + 0.15
    assert singles
    assert sum(float(t["suggested_stake"]) for t in out) <= 100.0 + 0.05
