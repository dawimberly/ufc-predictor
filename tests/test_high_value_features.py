"""Tests for Phase 1 high-value features (leakage-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.high_value_features import (
    HIGH_VALUE_DIFF_COLUMNS,
    build_hv_matchup_features,
    division_peak_age,
    hv_layoff_flags,
)


def test_hv_layoff_thresholds() -> None:
    assert hv_layoff_flags(21) == (1.0, 0.0)
    assert hv_layoff_flags(22) == (0.0, 0.0)
    assert hv_layoff_flags(365) == (0.0, 1.0)
    assert hv_layoff_flags(364) == (0.0, 0.0)
    assert hv_layoff_flags(float("nan")) == (0.0, 0.0)


def test_division_peak_age_table() -> None:
    assert division_peak_age("Lightweight") == 30.0
    assert division_peak_age("Heavyweight") == 33.0
    assert division_peak_age("Women's Strawweight") == 28.0


def test_hv_matchup_td_pressure_sign() -> None:
    f1 = {
        "td_acc": 0.2,
        "td_defense": 0.8,
        "control_time_per_min": 1.0,
        "hv_short_notice_flag": 0.0,
        "hv_long_layoff_flag": 0.0,
        "first_fight_new_wc_flag": 0.0,
        "finish_rate_l5": 0.5,
        "division_age_adj": -1.0,
        "wins_vs_better_record_l5": 0.2,
        "ko_losses_career_flag": 0.0,
    }
    f2 = {
        "td_acc": 0.6,
        "td_defense": 0.4,
        "control_time_per_min": 2.0,
        "hv_short_notice_flag": 1.0,
        "hv_long_layoff_flag": 0.0,
        "first_fight_new_wc_flag": 1.0,
        "finish_rate_l5": 0.3,
        "division_age_adj": 2.0,
        "wins_vs_better_record_l5": 0.0,
        "ko_losses_career_flag": 1.0,
    }
    out = build_hv_matchup_features(f1, f2)
    for col in HIGH_VALUE_DIFF_COLUMNS:
        assert col in out
    # f1 faces high opp TD acc with strong TD def → positive pressure for f1
    assert out["hv_td_pressure_diff"] > 0
    assert out["hv_control_clash"] == 2.0
    assert out["finish_rate_l5_diff"] == 0.2
    assert out["ko_losses_career_flag_diff"] == -1.0


def test_hv_rolling_no_leakage_first_fight() -> None:
    from src.high_value_features import apply_hv_rolling_extras

    hist = pd.DataFrame(
        {
            "fighter": ["A", "A", "B", "B"],
            "opponent": ["B", "C", "A", "D"],
            "fight_id": ["1", "2", "1", "3"],
            "event_date": pd.to_datetime(
                ["2020-01-01", "2020-06-01", "2020-01-01", "2020-07-01"]
            ),
            "won": [1, 0, 0, 1],
            "is_ko": [0, 1, 0, 0],
            "finish": [1.0, 1.0, 0.0, 0.0],
            "win_rate": [np.nan, 1.0, np.nan, 0.0],
            "days_since_last_fight": [np.nan, 152.0, np.nan, 182.0],
            "age": [28.0, 28.5, 30.0, 30.5],
            "weight_class": ["Lightweight", "Welterweight", "Lightweight", "Lightweight"],
        }
    )
    out = apply_hv_rolling_extras(hist, date_col="event_date")
    # First appearance: no prior KO losses, finish_rate_l5 NaN (shift)
    first_a = out[(out["fighter"] == "A") & (out["fight_id"] == "1")].iloc[0]
    assert float(first_a["ko_losses_career_flag"]) == 0.0
    assert pd.isna(first_a["finish_rate_l5"]) or first_a["finish_rate_l5"] != first_a["finish"]
    # Second A fight: first_fight_new_wc should be 1 (LW → WW)
    second_a = out[(out["fighter"] == "A") & (out["fight_id"] == "2")].iloc[0]
    assert float(second_a["first_fight_new_wc_flag"]) == 1.0
    assert float(second_a["hv_short_notice_flag"]) == 0.0  # 152d
