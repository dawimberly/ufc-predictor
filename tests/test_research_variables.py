"""Prior-only research variables: streaks, reversals, rank. No current-fight leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.research_variables import RESEARCH_FEATURE_COLUMNS, attach_research_variables


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_id": ["a", "b", "c"],
            "event_date": ["2020-01-01", "2020-06-01", "2021-01-01"],
            "event": ["E1", "E2", "E3"],
            "fighter_1": ["Ann", "Ann", "Ann"],
            "fighter_2": ["Bea", "Bea", "Bea"],
            "f1_win": [1, 0, 1],
            "method": ["Decision - Split", "KO/TKO", "Decision - Unanimous"],
            "round": [3, 1, 5],
            "is_title_fight": [0, 0, 1],
            "weight_class": ["Women's Strawweight"] * 3,
            "scheduled_rounds": [3, 3, 5],
            "stance_matchup": [1, 0, 1],
        }
    )


def _stats() -> pd.DataFrame:
    rows = []
    for event, fighter, rev, sig, total in (
        ("E1", "Ann", 2, "10 of 20", "20 of 40"),
        ("E1", "Bea", 0, "8 of 20", "10 of 40"),
        ("E2", "Ann", 0, "1 of 10", "2 of 10"),
        ("E2", "Bea", 4, "1 of 10", "2 of 10"),
    ):
        rows.append(
            {
                "EVENT": event,
                "BOUT": "Ann vs Bea",
                "ROUND": "Round 1",
                "FIGHTER": fighter,
                "REV.": rev,
                "SIG.STR.": sig,
                "TOTAL STR.": total,
            }
        )
    return pd.DataFrame(rows)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EVENT": ["E1", "E2", "E3"],
            "DATE": ["2020-01-01", "2020-06-01", "2021-01-01"],
            "LOCATION": [
                "Las Vegas, Nevada, USA",
                "Austin, Texas, USA",
                "Abu Dhabi, UAE",
            ],
        }
    )


def _rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-06-01"],
            "R_fighter": ["Ann"],
            "B_fighter": ["Bea"],
            "R_match_weightclass_rank": [1],
            "B_match_weightclass_rank": [np.nan],
            "R_Pound-for-Pound_rank": [3],
            "B_Pound-for-Pound_rank": [np.nan],
            "R_Weight_lbs": [115],
            "B_Weight_lbs": [125],
            "empty_arena": [1],
        }
    )


def test_research_columns_are_trainable_and_market_is_not() -> None:
    for col in RESEARCH_FEATURE_COLUMNS:
        assert col in config.FEATURE_COLUMNS
    assert "mkt_implied_prob" not in config.FEATURE_COLUMNS
    assert "line_move" not in config.FEATURE_COLUMNS


def test_prior_fights_only_and_public_rank(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_PATHWAY_FEATURES", raising=False)
    out = attach_research_variables(
        _history(),
        stats=_stats(),
        events=_events(),
        rankings=_rankings(),
        tott=pd.DataFrame(),
        download=False,
    )
    first = out.iloc[0]
    assert first["win_streak_diff"] == 0
    assert first["rounds_fought_diff"] == 0
    assert pd.isna(first["split_dec_rate_career_diff"])
    assert pd.isna(first["rev_rate_diff"])

    second = out.iloc[1]
    # Ann won the split; Bea lost it. Current KO must not change the split rate.
    assert second["win_streak_diff"] == 1
    assert second["lose_streak_diff"] == -1
    assert second["longest_win_streak_diff"] == 1
    assert second["split_dec_rate_career_diff"] == 0
    assert second["rounds_fought_diff"] == 0
    assert second["rev_rate_diff"] == 2
    # Rank 1 vs unranked 16. Positive means fighter 1 is ranked ahead.
    assert second["rank_diff"] == 15
    assert second["p4p_rank_diff"] == 13
    assert second["listed_weight_diff"] == -10
    assert second["empty_arena"] == 1
    assert second["is_womens"] == 1
    assert str(second["location"]).startswith("Austin")
    assert second["event_outside_usa"] == 0

    third = out.iloc[2]
    # Entering the third fight Ann's last result is a loss, so her win streak is 0.
    # The unanimous result of fight c is not in the split rate (still 1 of 2).
    assert third["win_streak_diff"] == -1
    assert third["lose_streak_diff"] == 1
    assert third["split_dec_rate_career_diff"] == 0
    assert third["rounds_fought_diff"] == 0
    assert third["title_bouts_diff"] == 0
    assert third["is_five_round"] == 1
    assert third["path_stance_mismatch"] == 1
    assert pd.isna(third["rank_diff"])
    assert third["empty_arena"] == 0
    assert third["event_outside_usa"] == 1
    assert str(third["location"]).startswith("Abu Dhabi")
