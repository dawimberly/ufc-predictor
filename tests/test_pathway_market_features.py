"""Tests for UFC pathway + market feature blocks (leakage-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_features import (
    decimal_odds_to_implied_pair,
    model_minus_market,
    shrink_proba_toward_market,
)
from src.pathway_features import (
    PATHWAY_DIFF_COLUMNS,
    apply_pathway_rolling_extras,
    build_pathway_matchup_features,
)


def test_pathway_first_fight_rates_are_nan_or_shifted() -> None:
    hist = pd.DataFrame(
        {
            "fighter": ["A", "A", "B", "B"],
            "opponent": ["B", "C", "A", "D"],
            "fight_id": ["1", "2", "1", "3"],
            "event_date": pd.to_datetime(
                ["2020-01-01", "2020-06-01", "2020-01-01", "2020-07-01"]
            ),
            "won": [1, 0, 0, 1],
            "is_ko": [1, 0, 0, 0],
            "is_sub": [0, 0, 0, 0],
            "is_dec": [0, 1, 1, 0],
            "round": [1, 3, 3, 2],
            "finish": [1.0, 0.0, 0.0, 1.0],
            "takedowns_attempted": [2.0, 1.0, 4.0, 0.0],
            "sig_strikes_per_min": [5.0, 3.0, 2.0, 4.0],
            "td_defense": [0.7, 0.6, 0.5, 0.8],
            "_opp_elo_prefight": [1500.0, 1600.0, 1550.0, 1400.0],
        }
    )
    out = apply_pathway_rolling_extras(hist)
    first_a = out[(out["fighter"] == "A") & (out["fight_id"] == "1")].iloc[0]
    assert pd.isna(first_a["ko_win_rate_career"])
    assert pd.isna(first_a["last_loss_opp_elo"])
    second_a = out[(out["fighter"] == "A") & (out["fight_id"] == "2")].iloc[0]
    assert float(second_a["ko_win_rate_career"]) == 1.0
    assert float(second_a["r1_finish_rate_career"]) == 1.0
    # A won fight 1 — still no prior loss
    assert pd.isna(second_a["last_loss_opp_elo"])


def test_last_loss_opp_elo_after_defeat() -> None:
    hist = pd.DataFrame(
        {
            "fighter": ["A", "A", "A"],
            "won": [0, 1, 0],
            "is_ko": [1, 0, 0],
            "is_sub": [0, 0, 0],
            "is_dec": [0, 1, 1],
            "round": [2, 3, 3],
            "finish": [1.0, 0.0, 0.0],
            "takedowns_attempted": [1.0, 1.0, 1.0],
            "sig_strikes_per_min": [3.0, 3.0, 3.0],
            "_opp_elo_prefight": [1700.0, 1400.0, 1600.0],
        }
    )
    out = apply_pathway_rolling_extras(hist)
    # Fight 2 sees last loss opp elo = 1700
    assert float(out.iloc[1]["last_loss_opp_elo"]) == 1700.0
    # Fight 3 still sees 1700 (loss in fight 1; fight 2 was a win)
    assert float(out.iloc[2]["last_loss_opp_elo"]) == 1700.0


def test_pathway_matchup_columns_present() -> None:
    f1 = {
        "ko_win_rate_l5": 0.4,
        "ko_win_rate_career": 0.3,
        "sub_win_rate_l5": 0.1,
        "sub_win_rate_career": 0.1,
        "dec_win_rate_l5": 0.5,
        "dec_win_rate_career": 0.5,
        "ko_loss_rate_l5": 0.05,
        "ko_loss_rate_career": 0.1,
        "sub_loss_rate_l5": 0.0,
        "sub_loss_rate_career": 0.05,
        "dec_loss_rate_l5": 0.1,
        "dec_loss_rate_career": 0.1,
        "r1_finish_rate_l5": 0.2,
        "r1_finish_rate_career": 0.25,
        "late_finish_rate_l5": 0.05,
        "late_finish_rate_career": 0.1,
        "distance_rate_l5": 0.4,
        "distance_rate_career": 0.45,
        "cardio_decay_proxy": 0.04,
        "finish_timing_skew": 0.15,
        "last_loss_opp_elo": 1550.0,
        "td_att_rate_career": 3.0,
        "sub_att_rate_career": 0.1,
        "td_defense": 0.8,
        "pace_l5": 4.0,
        "stance_southpaw": 1.0,
        "stance_orthodox": 0.0,
    }
    f2 = {
        "ko_win_rate_l5": 0.2,
        "ko_win_rate_career": 0.5,
        "sub_win_rate_l5": 0.2,
        "sub_win_rate_career": 0.15,
        "dec_win_rate_l5": 0.3,
        "dec_win_rate_career": 0.35,
        "ko_loss_rate_l5": 0.2,
        "ko_loss_rate_career": 0.25,
        "sub_loss_rate_l5": 0.1,
        "sub_loss_rate_career": 0.1,
        "dec_loss_rate_l5": 0.2,
        "dec_loss_rate_career": 0.2,
        "r1_finish_rate_l5": 0.1,
        "r1_finish_rate_career": 0.1,
        "late_finish_rate_l5": 0.15,
        "late_finish_rate_career": 0.15,
        "distance_rate_l5": 0.5,
        "distance_rate_career": 0.5,
        "cardio_decay_proxy": 0.1,
        "finish_timing_skew": -0.05,
        "last_loss_opp_elo": 1450.0,
        "td_att_rate_career": 1.0,
        "sub_att_rate_career": 0.2,
        "td_defense": 0.5,
        "pace_l5": 2.0,
        "stance_southpaw": 0.0,
        "stance_orthodox": 1.0,
    }
    out = build_pathway_matchup_features(f1, f2, scheduled_rounds=5)
    for col in PATHWAY_DIFF_COLUMNS:
        assert col in out
    assert out["ko_win_rate_l5_diff"] == 0.2
    assert out["path_stance_mismatch"] == 1.0
    assert out["is_five_round"] == 1.0
    assert out["last_loss_opp_elo_diff"] == 100.0


def test_market_devig_and_cal_shrink() -> None:
    p1, p2 = decimal_odds_to_implied_pair(1.5, 2.5)
    assert abs(p1 + p2 - 1.0) < 1e-9
    assert p1 > p2
    proba = np.array([0.8, 0.6])
    mkt = np.array([0.5, 0.55])
    width = np.array([0.5, 0.1])
    cal = shrink_proba_toward_market(proba, mkt, width, width_threshold=0.4, shrink=0.5)
    assert abs(cal[0] - 0.65) < 1e-9
    assert abs(cal[1] - 0.6) < 1e-9
    resid = model_minus_market(proba, mkt)
    assert abs(resid[0] - 0.3) < 1e-9


def test_config_pathway_flags_exist_and_default_off() -> None:
    import os

    import config

    os.environ["ENABLE_PATHWAY_FEATURES"] = "false"
    os.environ["ENABLE_MARKET_FEATURES"] = "false"
    config.refresh_runtime_env()
    assert config.ENABLE_PATHWAY_FEATURES is False
    assert config.ENABLE_MARKET_FEATURES is False
    for c in config.PATHWAY_FEATURE_COLUMNS:
        assert c not in config.FEATURE_COLUMNS
    for c in config.MARKET_FEATURE_COLUMNS:
        assert c not in config.FEATURE_COLUMNS
