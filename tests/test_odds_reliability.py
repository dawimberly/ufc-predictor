"""Unit tests for odds reliability guards (placeholders, blank unmatched, suspect edges)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.odds_providers.odds_reliability import (
    HARD_BLANK_EDGE,
    SUSPECT_EDGE,
    blank_unmatched_odds_rows,
    edge_is_suspect,
    harden_merged_odds_frame,
    is_placeholder_auth,
    usable_cookie,
    usable_session_token,
)
from src.predictor import attach_edge_columns, merge_predictions_with_odds, rank_predictions_by_edge
from src.strategy import edge_is_actionable


def test_placeholder_auth_rejected():
    assert is_placeholder_auth("")
    assert is_placeholder_auth("your_cookie")
    assert is_placeholder_auth("YOUR_SESSION")
    assert is_placeholder_auth("changeme")
    assert usable_cookie("your_cookie") == ""
    assert usable_cookie("short") == ""
    assert usable_session_token("session") == ""
    assert usable_session_token("token") == ""
    real = "a" * 24
    assert usable_cookie(real) == real
    assert usable_session_token("abc12345xyz") == "abc12345xyz"


def test_blank_unmatched_clears_edges():
    df = pd.DataFrame(
        {
            "fighter_1": ["A", "B"],
            "fighter_2": ["C", "D"],
            "odds_matched": [True, False],
            "f1_odds": [1.8, 1.5],
            "f2_odds": [2.1, 2.5],
            "edge_pct": [5.0, -100.0],
            "edge_f1": [0.05, -1.0],
            "edge_f2": [0.02, 0.01],
            "best_edge": [0.05, -1.0],
            "odds_source": ["DK", "DK"],
        }
    )
    out = blank_unmatched_odds_rows(df)
    assert bool(out.loc[0, "odds_matched"])
    assert pd.isna(out.loc[1, "f1_odds"])
    assert pd.isna(out.loc[1, "edge_pct"])
    assert pd.isna(out.loc[1, "best_edge"])
    assert out.loc[1, "odds_source"] == ""


def test_suspect_edge_flagged_and_hard_blanked():
    assert edge_is_suspect(0.26)
    assert not edge_is_suspect(0.20)
    assert edge_is_suspect(0.35, hard=True)
    assert SUSPECT_EDGE == 0.25
    assert HARD_BLANK_EDGE == 0.30

    df = pd.DataFrame(
        {
            "odds_matched": [True, True],
            "edge_pct": [40.0, 10.0],  # percent points
            "best_edge": [0.40, 0.10],
        }
    )
    out = harden_merged_odds_frame(df)
    assert bool(out.loc[0, "edge_suspect"])
    assert pd.isna(out.loc[0, "edge_pct"])  # >30% blanked
    assert not bool(out.loc[1, "edge_suspect"])
    assert float(out.loc[1, "edge_pct"]) == 10.0


def test_merge_unmatched_no_fake_edge():
    preds = pd.DataFrame(
        {
            "fighter_1": ["Alice Alpha", "Bob Beta"],
            "fighter_2": ["Carol Gamma", "Dana Delta"],
            "prob_f1_win": [0.6, 0.55],
            "prob_f2_win": [0.4, 0.45],
            "predicted_winner": ["Alice Alpha", "Bob Beta"],
        }
    )
    market = pd.DataFrame(
        {
            "fighter_1": ["Alice Alpha"],
            "fighter_2": ["Carol Gamma"],
            "f1_odds": [1.9],
            "f2_odds": [2.0],
            "bookmaker": ["DraftKings"],
        }
    )
    merged = merge_predictions_with_odds(preds, market, fetch_if_missing=False)
    assert int(merged["odds_matched"].sum()) == 1
    unmatched = merged[~merged["odds_matched"].astype(bool)].iloc[0]
    assert pd.isna(unmatched["f1_odds"])
    assert pd.isna(unmatched["edge_pct"])
    assert pd.isna(unmatched["best_edge"])

    ranked = rank_predictions_by_edge(merged, min_edge=0.0)
    assert len(ranked) == 1
    assert bool(ranked.iloc[0]["odds_matched"])


def test_attach_edge_skips_unmatched():
    preds = pd.DataFrame(
        {
            "prob_f1_win": [0.9, 0.55],
            "prob_f2_win": [0.1, 0.45],
            "f1_odds": [1.2, np.nan],
            "f2_odds": [5.0, np.nan],
            "odds_matched": [True, False],
        }
    )
    out = attach_edge_columns(preds)
    assert pd.notna(out.loc[0, "edge_pct"]) or bool(out.loc[0].get("edge_suspect"))
    assert pd.isna(out.loc[1, "edge_pct"])


def test_edge_is_actionable_caps_and_suspect():
    assert edge_is_actionable(0.12)
    assert not edge_is_actionable(0.30)
    assert not edge_is_actionable(0.10, edge_suspect=True)
    assert not edge_is_actionable(-0.05)
