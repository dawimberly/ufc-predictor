"""Tests for automatic 2-leg + 3-leg Ollama parlay recommendations."""

from __future__ import annotations

import pandas as pd

from src.grok_analysis import merge_ollama_reasons_into_parlays
from src.strategy import build_auto_parlay_recommendations


def _card(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        p = 0.90 - i * 0.05
        rows.append(
            {
                "fight_id": f"f{i}",
                "fighter_1": f"A{i}",
                "fighter_2": f"B{i}",
                "predicted_winner": f"A{i}",
                "prob_f1_win": p,
                "prob_f2_win": 1.0 - p,
                "predicted_prob": p,
                "confidence_label": "high",
                "weight_class": "LW",
            }
        )
    return pd.DataFrame(rows)


def test_auto_parlays_one_2_and_one_3() -> None:
    recs = build_auto_parlay_recommendations(_card(5))
    ns = sorted(int(r["n_legs"]) for r in recs)
    assert ns == [2, 3]
    two = next(r for r in recs if r["n_legs"] == 2)
    three = next(r for r in recs if r["n_legs"] == 3)
    assert two["combined_prob"] > three["combined_prob"]
    assert two["advisory"] is True
    assert three["stake_usd"] == 0.0
    assert len(two["legs"]) == 2
    assert len(three["legs"]) == 3


def test_auto_parlays_prefers_ha_singles() -> None:
    ha = [
        {
            "fight_id": "f4",
            "fight": "A4 vs B4",
            "suggested_stake": 5.0,
            "stake_pct": 2.0,
        },
        {
            "fight_id": "f3",
            "fight": "A3 vs B3",
            "suggested_stake": 4.0,
            "stake_pct": 1.5,
        },
    ]
    recs = build_auto_parlay_recommendations(_card(5), ha_singles=ha)
    two = next(r for r in recs if r["n_legs"] == 2)
    # Best HA-preferring 2-leg should include the preferred fights when possible
    ids = {leg["fight_id"] for leg in two["legs"]}
    assert "f3" in ids or "f4" in ids


def test_merge_parlay_reasons() -> None:
    base = build_auto_parlay_recommendations(_card(4))
    narrated = merge_ollama_reasons_into_parlays(
        base,
        [{"id": base[0]["id"], "n_legs": base[0]["n_legs"], "reason": "Stacked favorites"}],
    )
    assert narrated[0]["reason"].startswith("ADVISORY:")
    assert "Stacked favorites" in narrated[0]["reason"]
    assert narrated[0]["stake_pct"] == 0.0
