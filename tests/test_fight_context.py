"""Tests for read-only fight context overlay (display only)."""

from __future__ import annotations

import pandas as pd

from src.fight_context import (
    build_fight_context,
    disagreement_band,
    format_fight_context_lines,
)


def test_context_market_na_without_odds() -> None:
    row = pd.Series(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "predicted_winner": "A",
            "predicted_prob": 0.62,
            "odds_matched": False,
        }
    )
    ctx = build_fight_context(row)
    assert "n/a" in ctx["market"].lower()
    lines = format_fight_context_lines(ctx)
    assert lines
    assert "A vs B" in lines[0]


def test_context_disagreement_badge_not_ci() -> None:
    row = {
        "fighter_1": "A",
        "fighter_2": "B",
        "predicted_winner": "A",
        "predicted_prob": 0.7,
        "odds_matched": True,
        "f1_odds": 1.5,
        "f2_odds": 2.5,
        "implied_prob_f1": 0.62,
        "interval_width": 0.55,
        "ensemble_disagreement": 0.002,
    }
    ctx = build_fight_context(row)
    assert "ci" not in ctx
    assert "Disagree: low" in ctx.get("disagree", "")
    assert "implied" in ctx["market"].lower()
    lines = format_fight_context_lines(ctx)
    assert any("Disagree" in ln for ln in lines)
    assert not any(ln.startswith("CI:") for ln in lines)


def test_disagreement_bands() -> None:
    assert disagreement_band(0.005) == "low"
    assert disagreement_band(0.02) == "mid"
    assert disagreement_band(0.05) == "high"
    assert disagreement_band(None) is None


def test_context_hides_method_when_missing() -> None:
    ctx = build_fight_context({"fighter_1": "A", "fighter_2": "B"})
    assert "method" not in ctx


def test_context_decision_profile_when_present() -> None:
    ctx = build_fight_context(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "predicted_winner": "A",
            "f1_dec_win_rate_l5": 0.4,
            "f1_split_dec_win_rate_career": 0.1,
            "f1_decision_finish_share_career": 0.55,
            "odds_matched": False,
        }
    )
    assert "decision" in ctx
    assert "dec-win" in ctx["decision"]
    assert any("Decision profile" in ln for ln in format_fight_context_lines(ctx))


def test_context_judges_from_row_fields() -> None:
    ctx = build_fight_context(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "judge_names": "Sal D'Amato; Derek Cleary; Chris Lee",
            "panel_event_country_share": 1.0,
            "event_country": "usa",
            "odds_matched": False,
        }
    )
    assert "judges" in ctx
    assert "majority event-country" in ctx["judges"]
