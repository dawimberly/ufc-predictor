"""Odds merge matching + diagnostics."""

from __future__ import annotations

import pandas as pd

from src.predictor import (
    LAST_ODDS_MATCH_META,
    _fighter_name_key,
    _names_match,
    merge_predictions_with_odds,
)


def test_unicode_name_normalization():
    assert _fighter_name_key("Milošević") == _fighter_name_key("Milosevic")
    assert _names_match("Nina Milošević", "Nina Milosevic")
    assert _names_match("Aleksandar Rakic", "Aleksandar Rakić")


def test_merge_diagnostics_name_mismatch():
    card = pd.DataFrame(
        {
            "fighter_1": ["Cody Gibson"],
            "fighter_2": ["Abdul Hussein"],
            "prob_f1_win": [0.55],
            "prob_f2_win": [0.45],
        }
    )
    market = pd.DataFrame(
        {
            "fighter_1": ["Aleksandar Rakic"],
            "fighter_2": ["Marcin Tybura"],
            "f1_odds": [1.5],
            "f2_odds": [2.5],
            "bookmaker": ["TestBook"],
            "odds_source": ["the_odds_api"],
        }
    )
    merged = merge_predictions_with_odds(card, market, fetch_if_missing=False)
    assert int(merged["odds_matched"].sum()) == 0
    assert LAST_ODDS_MATCH_META["reason"] == "name_mismatch"
    assert LAST_ODDS_MATCH_META["api_events"] == 1
    assert LAST_ODDS_MATCH_META["matched"] == 0
    assert LAST_ODDS_MATCH_META["unmatched"]


def test_merge_diagnostics_ok():
    card = pd.DataFrame(
        {
            "fighter1": ["Aleksandar Rakic"],
            "fighter2": ["Marcin Tybura"],
            "prob_f1_win": [0.6],
            "prob_f2_win": [0.4],
        }
    )
    market = pd.DataFrame(
        {
            "fighter_1": ["Aleksandar Rakić"],
            "fighter_2": ["Marcin Tybura"],
            "f1_odds": [1.4],
            "f2_odds": [2.8],
            "bookmaker": ["BetOnline.ag"],
            "odds_source": ["the_odds_api"],
        }
    )
    merged = merge_predictions_with_odds(card, market, fetch_if_missing=False)
    assert int(merged["odds_matched"].sum()) == 1
    assert LAST_ODDS_MATCH_META["reason"] == "ok"
    assert LAST_ODDS_MATCH_META["matched"] == 1
