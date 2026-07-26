"""High-accuracy / low-volume strategy hard rules."""

from __future__ import annotations

import pandas as pd

from src.high_accuracy_strategy import (
    ALLOWED_PROP_KEYS,
    PARLAY_MAX_LEGS,
    PROP_PARLAYS_ENABLED,
    ROUND_ROBINS_ENABLED,
    apply_hardcoded_profile_defaults,
    clamp_max_tickets,
    prop_allowed,
    strategy_rules_summary,
)
from src.parlay_builder import parlay_max_legs_for_profile
from src.strategy import apply_max_tickets_per_card, build_parlay_candidates, StrategyConfig


def test_only_over_1_5_allowed():
    assert prop_allowed("over_1_5_rounds")
    assert not prop_allowed("ko_tko")
    assert not prop_allowed("goes_to_decision")
    assert ALLOWED_PROP_KEYS == frozenset({"over_1_5_rounds"})


def test_round_robins_and_prop_parlays_off():
    assert ROUND_ROBINS_ENABLED is False
    assert PROP_PARLAYS_ENABLED is False


def test_parlay_max_legs_hardcoded_to_2():
    assert PARLAY_MAX_LEGS == 2
    assert parlay_max_legs_for_profile() == 2


def test_clamp_max_tickets_1_to_4():
    assert clamp_max_tickets(0) == 1
    assert clamp_max_tickets(3) == 3
    assert clamp_max_tickets(9) == 4
    assert clamp_max_tickets(None) in (2, 3)


def test_profile_floors_cannot_loosen_below_ha():
    loose = {
        "alert_min_edge": 0.01,
        "singles_min_model_prob": 0.50,
        "singles_min_confidence": "low",
        "max_bets_per_card": 10,
        "parlay_min_edge": 0.01,
        "parlay_min_combined_prob": 0.10,
        "parlay_min_ev": 0.01,
        "parlay_max_legs": 5,
        "prop_min_model_prob": 0.10,
        "prop_min_edge": 0.01,
        "alert_max_parlays": 8,
        "max_parlays_show": 8,
    }
    out = apply_hardcoded_profile_defaults(loose, live=False)
    assert out["alert_min_edge"] >= 0.06
    assert out["singles_min_model_prob"] >= 0.70
    assert out["singles_min_confidence"] == "medium"
    assert out["max_bets_per_card"] <= 4
    assert out["parlay_max_legs"] == 2
    assert out["prop_min_model_prob"] >= 0.78


def test_build_parlay_only_2_legs_both_strong(monkeypatch):
    # Minimal card with odds/probs — builder returns [] or 2-leg only
    rows = pd.DataFrame(
        [
            {
                "fight_id": "a",
                "event_name": "UFC",
                "fighter_1": "A",
                "fighter_2": "B",
                "prob_f1_win": 0.75,
                "prob_f2_win": 0.25,
                "f1_odds": 1.50,
                "f2_odds": 2.80,
                "confidence_label": "high",
                "ensemble_disagreement": 0.01,
                "interval_width": 0.10,
            },
            {
                "fight_id": "b",
                "event_name": "UFC",
                "fighter_1": "C",
                "fighter_2": "D",
                "prob_f1_win": 0.72,
                "prob_f2_win": 0.28,
                "f1_odds": 1.55,
                "f2_odds": 2.60,
                "confidence_label": "high",
                "ensemble_disagreement": 0.01,
                "interval_width": 0.10,
            },
            {
                "fight_id": "c",
                "event_name": "UFC",
                "fighter_1": "E",
                "fighter_2": "F",
                "prob_f1_win": 0.71,
                "prob_f2_win": 0.29,
                "f1_odds": 1.60,
                "f2_odds": 2.40,
                "confidence_label": "high",
                "ensemble_disagreement": 0.01,
                "interval_width": 0.10,
            },
        ]
    )
    cfg = StrategyConfig(
        min_edge=0.05,
        min_model_prob=0.60,
        min_confidence="medium",
        parlay_min_edge=0.05,
        parlay_min_combined_prob=0.20,
        parlay_max_legs=5,  # request 5 — hard cap still 2
        parlay_min_leg_prob=0.60,
    )
    parlays = build_parlay_candidates(rows, config=cfg)
    assert all(len(p.legs) == 2 for p in parlays)


def test_ticket_cap_prefers_singles():
    singles = [
        {"event_name": "Card", "fight": f"F{i}", "pick": "X", "prob": 0.8 - i * 0.01, "edge": 0.1, "uncertainty_action": "allow"}
        for i in range(4)
    ]
    parlays = [
        {"event_name": "Card", "expected_value": 0.5, "picks": "a+b", "legs": [{}, {}]},
        {"event_name": "Card", "expected_value": 0.4, "picks": "c+d", "legs": [{}, {}]},
    ]
    ks, kp, os_, op = apply_max_tickets_per_card(singles, parlays, max_tickets=3)
    assert len(ks) + len(kp) == 3
    assert len(ks) == 3  # singles fill budget first
    assert len(kp) == 0


def test_strategy_rules_summary_shape():
    s = strategy_rules_summary()
    assert s["props"]["only_over_1_5"] is True
    assert s["parlays"]["max_legs"] == 2
    assert s["round_robins_enabled"] is False
