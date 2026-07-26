"""Tests for cross-book arb scanner."""

from __future__ import annotations

from src.arb_scanner import (
    arb_math,
    arb_row_alert_key,
    fight_pair_key,
    format_arb_alert_message,
    is_dk_mybookie_row,
    scan_moneyline_arbs,
    strong_arb_rows,
)


def test_arb_math_true_arb():
    m = arb_math(2.20, 2.20, stake_total=100)
    assert m["inv_sum"] < 1.0
    assert m["profit_pct"] > 0
    assert abs(m["stake_a"] + m["stake_b"] - 100) < 0.01


def test_arb_math_no_arb():
    m = arb_math(1.50, 2.50, stake_total=100)
    assert m["inv_sum"] > 1.0
    assert m["profit_pct"] == 0.0
    assert m["overround_pct"] > 0


def test_scan_moneyline_finds_near_arb():
    quotes = [
        {
            "book": "DraftKings",
            "fighter_1": "Shane Collins",
            "fighter_2": "Otari Tanzilovi",
            "f1_odds": 1.56,
            "f2_odds": 2.50,
            "pair_key": fight_pair_key("Shane Collins", "Otari Tanzilovi"),
        },
        {
            "book": "MyBookie",
            "fighter_1": "Otari Tanzilovi",
            "fighter_2": "Shane Collins",
            "f1_odds": 2.61,
            "f2_odds": 1.48,
            "pair_key": fight_pair_key("Otari Tanzilovi", "Shane Collins"),
        },
    ]
    rows = scan_moneyline_arbs(quotes, near_margin_pct=3.0, stake_total=100)
    assert rows
    top = rows[0]
    assert top["fight"].lower().count("collins") >= 1
    assert top["is_near"] or top["is_arb"]
    assert top["overround_pct"] < 3.5


def test_is_dk_mybookie_row():
    dk_mb = {
        "side_a": {"book": "DraftKings"},
        "side_b": {"book": "MyBookie"},
    }
    dk_bn = {
        "side_a": {"book": "DraftKings"},
        "side_b": {"book": "BetNow.eu"},
    }
    assert is_dk_mybookie_row(dk_mb)
    assert not is_dk_mybookie_row(dk_bn)


def test_strong_arb_rows_filters_threshold_and_books():
    scan = {
        "moneyline": [
            {
                "market": "moneyline",
                "fight": "A vs B",
                "is_arb": True,
                "profit_pct": 3.1,
                "side_a": {"book": "DraftKings", "fighter": "A", "american": "+150"},
                "side_b": {"book": "MyBookie", "fighter": "B", "american": "+160"},
                "stake_a": 48.0,
                "stake_b": 52.0,
                "stake_total": 100.0,
            },
            {
                "market": "moneyline",
                "fight": "C vs D",
                "is_arb": True,
                "profit_pct": 1.5,
                "side_a": {"book": "DraftKings"},
                "side_b": {"book": "MyBookie"},
            },
            {
                "market": "moneyline",
                "fight": "E vs F",
                "is_arb": True,
                "profit_pct": 4.0,
                "side_a": {"book": "DraftKings"},
                "side_b": {"book": "BetNow.eu"},
            },
        ],
        "props": [],
    }
    strong = strong_arb_rows(scan, threshold_pct=2.5, dk_mybookie_only=True)
    assert len(strong) == 1
    assert strong[0]["fight"] == "A vs B"


def test_arb_row_alert_key_stable():
    row = {
        "market": "moneyline",
        "fight": "X vs Y",
        "side_a": {"book": "DraftKings", "odds": 2.1},
        "side_b": {"book": "MyBookie", "odds": 2.05},
    }
    assert arb_row_alert_key(row) == arb_row_alert_key(dict(row))


def test_format_arb_alert_message_includes_profit_and_stakes():
    row = {
        "market": "moneyline",
        "fight": "Fighter A vs Fighter B",
        "is_arb": True,
        "profit_pct": 2.8,
        "side_a": {"fighter": "Fighter A", "book": "DraftKings", "american": "+180"},
        "side_b": {"fighter": "Fighter B", "book": "MyBookie", "american": "+175"},
        "stake_a": 47.0,
        "stake_b": 53.0,
        "stake_total": 100.0,
    }
    msg = format_arb_alert_message(row)
    assert "+2.80%" in msg
    assert "DraftKings" in msg
    assert "MyBookie" in msg
    assert "$47" in msg
