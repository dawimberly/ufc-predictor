"""Decision-layer selectivity: confidence floors + max bets per card."""

from __future__ import annotations

from src.strategy import (
    apply_max_bets_per_card,
    confidence_meets_minimum,
    single_quality_score,
)


def test_confidence_meets_minimum():
    assert confidence_meets_minimum("high", "medium")
    assert confidence_meets_minimum("medium", "medium")
    assert not confidence_meets_minimum("low", "medium")
    assert confidence_meets_minimum("high", "high")
    assert not confidence_meets_minimum("medium", "high")


def test_quality_score_prefers_high_prob_low_uncertainty():
    strong = {
        "prob": 0.78,
        "edge": 0.08,
        "uncertainty_action": "allow",
        "interval_width": 0.20,
        "ensemble_disagreement": 0.01,
    }
    weak = {
        "prob": 0.58,
        "edge": 0.09,
        "uncertainty_action": "tighten",
        "interval_width": 0.50,
        "ensemble_disagreement": 0.09,
    }
    assert single_quality_score(strong) > single_quality_score(weak)


def test_max_bets_per_card_cap():
    singles = [
        {
            "event_name": "Card A",
            "fight": f"A{i} vs B{i}",
            "pick": f"A{i}",
            "prob": 0.70 - i * 0.01,
            "edge": 0.10 - i * 0.005,
            "uncertainty_action": "allow",
            "interval_width": 0.25,
            "ensemble_disagreement": 0.02,
        }
        for i in range(6)
    ]
    kept, overflow = apply_max_bets_per_card(singles, max_bets=3)
    assert len(kept) == 3
    assert len(overflow) == 3
    assert kept[0]["card_rank"] == 1
    assert all(s.get("skip_reason", "") == "" for s in kept)


def test_max_bets_per_card_is_per_event():
    singles = []
    for ev in ("Card A", "Card B"):
        for i in range(4):
            singles.append(
                {
                    "event_name": ev,
                    "fight": f"{ev}-{i}",
                    "pick": "X",
                    "prob": 0.75 - i * 0.02,
                    "edge": 0.08,
                    "uncertainty_action": "allow",
                    "interval_width": 0.2,
                    "ensemble_disagreement": 0.01,
                }
            )
    kept, overflow = apply_max_bets_per_card(singles, max_bets=2)
    assert len(kept) == 4  # 2 per card
    assert len(overflow) == 4
    assert sum(1 for s in kept if s["event_name"] == "Card A") == 2
