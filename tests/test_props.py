"""Tests for UFC prop betting module."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src.props import (
    method_flags,
    method_probs_from_row,
    prop_display_label,
    prop_short_label,
    rank_prop_singles,
    settle_prop,
    simulate_prop_bets,
    synthetic_market_odds,
)


@pytest.fixture(autouse=True)
def enable_props(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROPS", True)


def test_env_bool_parsing(monkeypatch):
    monkeypatch.setenv("ENABLE_PROPS", "true")
    assert config.env_bool("ENABLE_PROPS") is True
    monkeypatch.setenv("ENABLE_PROPS", '"true"')
    assert config.env_bool("ENABLE_PROPS") is True
    monkeypatch.setenv("ENABLE_PROPS", "1")
    assert config.env_bool("ENABLE_PROPS") is True
    monkeypatch.setenv("ENABLE_PROPS", "false")
    assert config.env_bool("ENABLE_PROPS") is False


def test_method_flags_ko_sub_dec():
    assert method_flags("KO/TKO") == (1, 0, 0)
    assert method_flags("SUB Armbar") == (0, 1, 0)
    assert method_flags("Decision - Unanimous") == (0, 0, 1)


def test_settle_goes_to_decision():
    row = pd.Series({"method": "Decision - Unanimous", "round": 3, "f1_win": 1})
    assert settle_prop("goes_to_decision", row) is True
    row2 = pd.Series({"method": "KO/TKO", "round": 1, "f1_win": 1})
    assert settle_prop("goes_to_decision", row2) is False


def test_settle_round_1_finish():
    row = pd.Series({"method": "KO/TKO", "round": 1})
    assert settle_prop("round_1_finish", row) is True
    row2 = pd.Series({"method": "KO/TKO", "round": 2})
    assert settle_prop("round_1_finish", row2) is False


def test_settle_over_under_1_5():
    over = pd.Series({"method": "DECISION - UNANIMOUS", "round": 3})
    under = pd.Series({"method": "KO/TKO", "round": 1})
    mid = pd.Series({"method": "SUBMISSION", "round": 2})
    assert settle_prop("over_1_5_rounds", over) is True
    assert settle_prop("under_1_5_rounds", over) is False
    assert settle_prop("over_1_5_rounds", under) is False
    assert settle_prop("under_1_5_rounds", under) is True
    assert settle_prop("over_1_5_rounds", mid) is True
    assert settle_prop("under_1_5_rounds", mid) is False
    # Partition: every settled fight is exactly one of over/under
    for rnd, method in ((1, "KO/TKO"), (2, "SUB"), (3, "U-DEC"), (5, "DECISION")):
        row = pd.Series({"method": method, "round": rnd})
        o = settle_prop("over_1_5_rounds", row)
        u = settle_prop("under_1_5_rounds", row)
        assert o is not None and u is not None
        assert bool(o) != bool(u)


def test_under_1_5_model_prob_complements_over():
    row = pd.Series(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "prob_f1_win": 0.6,
            "f1_ko_rate": 0.3,
            "f2_ko_rate": 0.2,
            "f1_finish_rate": 0.5,
            "f2_finish_rate": 0.4,
        }
    )
    p = method_probs_from_row(row)
    assert "under_1_5_rounds" in p
    assert abs(p["over_1_5_rounds"] + p["under_1_5_rounds"] - 1.0) < 1e-6


def test_method_probs_sum_reasonable():
    row = pd.Series(
        {
            "fighter_1": "Alice",
            "fighter_2": "Bob",
            "prob_f1_win": 0.6,
            "f1_ko_rate": 0.25,
            "f2_ko_rate": 0.15,
            "f1_sub_avg": 0.4,
            "f2_sub_avg": 0.2,
            "f1_finish_rate": 0.5,
            "f2_finish_rate": 0.4,
            "ko_rate_diff": 0.1,
            "sub_avg_diff": 0.05,
        }
    )
    probs = method_probs_from_row(row)
    assert 0.9 < probs["ko"] + probs["sub"] + probs["dec"] < 1.1
    assert probs["pick_name"] == "Alice"


def test_synthetic_market_odds():
    odds = synthetic_market_odds(0.4, vig=0.08)
    assert odds > 2.0


def test_prop_display_label():
    row = pd.Series({"fighter_1": "Chandler", "fighter_2": "Ruffy", "prob_f1_win": 0.55})
    label = prop_display_label("fighter_ko", row)
    assert "KO/TKO" in label
    assert "Chandler" in label
    assert "Chandler vs Ruffy" in label


def test_prop_short_label_fighter_method():
    row = pd.Series({"fighter_1": "Pereira", "fighter_2": "Hill", "prob_f1_win": 0.68})
    assert prop_short_label("fighter_ko", row) == "Pereira by KO/TKO"
    assert prop_short_label("fighter_sub", row) == "Pereira by Submission"
    assert prop_short_label("ko_tko", row) == "KO/TKO"


def test_simulate_prop_bets_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROPS", False)
    preds = pd.DataFrame()
    trades, summary = simulate_prop_bets(preds)
    assert trades.empty
    assert summary["trades"] == 0.0


def test_rank_prop_singles_includes_synthetic_when_no_live(monkeypatch):
    """Synthetic Over 1.5 props populate when live lines are absent (HA model floor)."""
    monkeypatch.setattr(config, "PROP_MIN_MODEL_PROB", 0.78)
    monkeypatch.setattr(config, "ENABLE_PROPS", True)
    monkeypatch.setattr(config, "PROP_MARKETS", ["over_1_5_rounds"])
    preds = pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "event_name": "UFC Test",
                "fighter_1": "Alice",
                "fighter_2": "Bob",
                "prob_f1_win": 0.62,
                "f1_ko_rate": 0.12,
                "f2_ko_rate": 0.10,
                "f1_sub_avg": 0.2,
                "f2_sub_avg": 0.2,
                # Low finish rates → high Over 1.5 model prob
                "f1_finish_rate": 0.25,
                "f2_finish_rate": 0.25,
                "ko_rate_diff": 0.0,
            }
        ]
    )
    ranked, meta = rank_prop_singles(preds, book="BetNow.eu", prop_odds=pd.DataFrame())
    assert ranked
    assert meta["strict_count"] >= 1
    assert all(r["odds_source"] == "synthetic" for r in ranked)
    assert all(r["prop_key"] == "over_1_5_rounds" for r in ranked)
    assert ranked[0]["prob"] >= 0.78
    assert ranked[0].get("prop_type")
    assert ranked[0].get("source_label") == "synthetic"


def test_method_probs_uses_decision_finish_share():
    """Higher decision_finish_share should tilt mass toward decision / Over 1.5."""
    base = {
        "fighter_1": "A",
        "fighter_2": "B",
        "prob_f1_win": 0.55,
        "f1_ko_rate": 0.25,
        "f2_ko_rate": 0.25,
        "f1_sub_avg": 0.3,
        "f2_sub_avg": 0.3,
        "f1_finish_rate": 0.55,
        "f2_finish_rate": 0.55,
        "ko_rate_diff": 0.0,
        "sub_avg_diff": 0.0,
    }
    low = method_probs_from_row(
        pd.Series(
            {
                **base,
                "f1_decision_finish_share_l5": 0.15,
                "f2_decision_finish_share_l5": 0.15,
            }
        )
    )
    high = method_probs_from_row(
        pd.Series(
            {
                **base,
                "f1_decision_finish_share_l5": 0.85,
                "f2_decision_finish_share_l5": 0.85,
            }
        )
    )
    assert high["dec"] > low["dec"]
    assert high["over_1_5_rounds"] >= low["over_1_5_rounds"] - 1e-9


def test_settle_fighter_ko_uses_model_pick_not_winner():
    """fighter_ko must use model pick; actual winner must not rewrite pick_side."""
    row = pd.Series(
        {
            "fighter_1": "Alice",
            "fighter_2": "Bob",
            "prob_f1_win": 0.70,
            "winner": "Bob",
            "f1_win": 0,
            "method": "KO/TKO",
            "round": 1,
        }
    )
    # Model picks Alice; Bob won by KO → fighter_ko (Alice by KO) loses
    assert settle_prop("fighter_ko", row) is False


def test_method_probs_prefer_pathway_l5_ko():
    base = {
        "fighter_1": "A",
        "fighter_2": "B",
        "prob_f1_win": 0.55,
        "f1_ko_rate": 0.15,
        "f2_ko_rate": 0.15,
        "f1_sub_avg": 0.3,
        "f2_sub_avg": 0.3,
        "f1_finish_rate_l5": 0.4,
        "f2_finish_rate_l5": 0.4,
    }
    low = method_probs_from_row(pd.Series(base))
    high = method_probs_from_row(
        pd.Series(
            {
                **base,
                "f1_ko_win_rate_l5": 0.45,
                "f2_ko_win_rate_l5": 0.40,
            }
        )
    )
    assert high["ko"] > low["ko"]


def test_method_probs_r1_from_pathway_rates():
    row = pd.Series(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "prob_f1_win": 0.55,
            "f1_finish_rate_l5": 0.5,
            "f2_finish_rate_l5": 0.5,
            "f1_r1_finish_rate_l5": 0.40,
            "f2_r1_finish_rate_l5": 0.35,
        }
    )
    p = method_probs_from_row(row)
    assert p["round_1_finish"] >= 0.30
    assert abs(p["over_1_5_rounds"] + p["under_1_5_rounds"] - 1.0) < 1e-6
