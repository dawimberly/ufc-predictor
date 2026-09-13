"""Round-total prop key mapping must use the book line point."""

from __future__ import annotations

import pandas as pd

from src.odds_providers.prop_odds_common import map_rounds_total, remap_totals_prop_keys
from src.props import event_from_record, method_probs_from_row, prop_model_prob, settle_prop


def test_map_rounds_total_points():
    assert map_rounds_total("over", 1.5) == ("over_1_5_rounds", "Over 1.5")
    assert map_rounds_total("under", 1.5) == ("under_1_5_rounds", "Under 1.5")
    assert map_rounds_total("O", 2.5) == ("over_2_5_rounds", "Over 2.5")
    assert map_rounds_total("U", 4.5) == ("under_4_5_rounds", "Under 4.5")


def test_remap_cached_mybookie_mislabels():
    df = pd.DataFrame(
        [
            {
                "fighter_1": "Islam Makhachev",
                "fighter_2": "Ian Machado Garry",
                "prop_key": "over_1_5_rounds",
                "selection": "Over 1.5",
                "market_key": "totals",
                "point": 4.5,
            },
            {
                "fighter_1": "Neil Magny",
                "fighter_2": "Ramiz Brahimaj",
                "prop_key": "over_1_5_rounds",
                "selection": "Over 1.5",
                "market_key": "totals",
                "point": 2.5,
            },
        ]
    )
    fixed = remap_totals_prop_keys(df)
    assert fixed.iloc[0]["prop_key"] == "over_4_5_rounds"
    assert fixed.iloc[0]["selection"] == "Over 4.5"
    assert fixed.iloc[1]["prop_key"] == "over_2_5_rounds"
    assert fixed.iloc[1]["selection"] == "Over 2.5"


def test_event_from_record():
    assert event_from_record({"event_name": "UFC 330"}) == "UFC 330"
    assert event_from_record({"event": "Fight Night"}) == "Fight Night"
    assert event_from_record(pd.Series({"event_name": "UFC 330"})) == "UFC 330"


def test_over_2_5_model_and_settle():
    row = pd.Series(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "scheduled_rounds": 3,
            "prob_f1_win": 0.55,
            "prob_f2_win": 0.45,
            "f1_finish_rate": 0.2,
            "f2_finish_rate": 0.2,
            "f1_ko_rate": 0.1,
            "f2_ko_rate": 0.1,
            "f1_sub_avg": 0.2,
            "f2_sub_avg": 0.2,
            "method": "Decision",
            "round": 3,
        }
    )
    probs = method_probs_from_row(row)
    assert "over_2_5_rounds" in probs
    assert 0.0 < prop_model_prob("over_2_5_rounds", row, probs) < 1.0
    assert settle_prop("over_2_5_rounds", row) is True
    row_r1 = row.copy()
    row_r1["method"] = "KO/TKO"
    row_r1["round"] = 1
    assert settle_prop("over_2_5_rounds", row_r1) is False
    assert settle_prop("under_2_5_rounds", row_r1) is True
