"""Replay + compare past cards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from src.replay import (
    build_compare_rows,
    format_replay_summary,
    list_past_events,
    resolve_replay_events,
    summarize_compare,
    write_replay_csv,
)


def _mini_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_name": "UFC 329",
                "event_date": "2025-11-01",
                "fighter_1": "A One",
                "fighter_2": "B Two",
                "f1_win": 1,
                "f1_odds": 1.80,
                "f2_odds": 2.10,
                "fight_id": "f1",
            },
            {
                "event_name": "UFC 329",
                "event_date": "2025-11-01",
                "fighter_1": "C Three",
                "fighter_2": "D Four",
                "f1_win": 0,
                "f1_odds": 2.50,
                "f2_odds": 1.55,
                "fight_id": "f2",
            },
            {
                "event_name": "UFC 328",
                "event_date": "2025-10-01",
                "fighter_1": "E Five",
                "fighter_2": "F Six",
                "f1_win": 1,
                "f1_odds": 1.90,
                "f2_odds": 1.95,
                "fight_id": "f3",
            },
        ]
    )


def test_list_and_resolve_past_events():
    feats = _mini_features()
    catalog = list_past_events(feats)
    assert [e["event"] for e in catalog] == ["UFC 328", "UFC 329"]

    assert resolve_replay_events(event="329", features=feats) == ["UFC 329"]
    assert resolve_replay_events(date="2025-10-01", features=feats) == ["UFC 328"]
    assert resolve_replay_events(last=1, features=feats) == ["UFC 329"]
    assert resolve_replay_events(last=2, features=feats) == ["UFC 328", "UFC 329"]

    with pytest.raises(ValueError, match="not found"):
        resolve_replay_events(event="UFC 999", features=feats)


def test_build_compare_rows_pnl_and_skips():
    preds = pd.DataFrame(
        [
            {
                "event_name": "UFC 329",
                "event_date": "2025-11-01",
                "fighter_1": "A One",
                "fighter_2": "B Two",
                "predicted_winner": "A One",
                "actual_winner": "A One",
                "correct": 1,
                "predicted_prob": 0.62,
                "best_edge": 0.08,
                "f1_odds": 1.80,
                "f2_odds": 2.10,
                "fight_id": "f1",
            },
            {
                "event_name": "UFC 329",
                "event_date": "2025-11-01",
                "fighter_1": "C Three",
                "fighter_2": "D Four",
                "predicted_winner": "C Three",
                "actual_winner": "D Four",
                "correct": 0,
                "predicted_prob": 0.55,
                "best_edge": 0.02,
                "f1_odds": 2.50,
                "f2_odds": 1.55,
                "fight_id": "f2",
            },
        ]
    )
    alerts = {
        "singles": [
            {
                "fight_id": "f1",
                "fight": "A One vs B Two",
                "pick": "A One",
                "suggested_stake": 10.0,
            }
        ],
        "skipped": [
            {
                "fight_id": "f2",
                "fight": "C Three vs D Four",
                "pick": "C Three",
                "skip_reason": "min_edge",
            }
        ],
    }
    rows = build_compare_rows(preds, alerts, stake=10.0)
    assert rows[0]["bet_taken"] is True
    assert rows[0]["pnl"] == pytest.approx(8.0)  # 10 * (1.8 - 1)
    assert rows[0]["correct"] == 1
    assert rows[1]["bet_taken"] is False
    assert rows[1]["skip_reason"] == "min_edge"

    summary = summarize_compare(rows, event_name="UFC 329", alerts=alerts)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["bets_taken"] == 1
    assert summary["pnl"] == pytest.approx(8.0)
    assert summary["skip_counts"]["min_edge"] == 1


def test_fail_closed_pnl_without_odds():
    preds = pd.DataFrame(
        [
            {
                "event_name": "UFC X",
                "fighter_1": "A",
                "fighter_2": "B",
                "predicted_winner": "A",
                "actual_winner": "A",
                "correct": 1,
                "predicted_prob": 0.6,
                "best_edge": 0.1,
                "f1_odds": None,
                "f2_odds": None,
                "fight_id": "fx",
            }
        ]
    )
    alerts = {
        "singles": [{"fight_id": "fx", "fight": "A vs B", "pick": "A", "suggested_stake": 10}],
        "skipped": [],
    }
    rows = build_compare_rows(preds, alerts)
    assert rows[0]["bet_taken"] is True
    assert rows[0]["pnl"] == ""
    summary = summarize_compare(rows, event_name="UFC X")
    assert summary["pnl"] is None
    assert summary["incomplete_pnl"] is True


def test_format_and_csv(tmp_path):
    report = {
        "overall": {
            "events": ["UFC 329"],
            "n_events": 1,
            "fights": 2,
            "scored": 2,
            "correct": 1,
            "accuracy": 0.5,
            "bets_taken": 1,
            "bets_won": 1,
            "bets_lost": 0,
            "pnl": 8.0,
            "roi": 0.8,
            "incomplete_pnl": False,
            "skip_counts": {"min_edge": 1},
            "per_event": [
                {
                    "event": "UFC 329",
                    "accuracy": 0.5,
                    "bets_taken": 1,
                    "pnl": 8.0,
                }
            ],
        },
        "csv_path": None,
    }
    text = format_replay_summary(report)
    assert "Pick accuracy: 50.0%" in text
    assert "PnL @ opening odds: $+8.00" in text
    assert "min_edge" in text

    path = write_replay_csv(
        [{"event": "UFC 329", "predicted_winner": "A", "actual_winner": "A", "correct": 1}],
        tmp_path / "replay.csv",
    )
    assert path.is_file()
    assert "predicted_winner" in path.read_text(encoding="utf-8")


def test_replay_cli_help():
    from src.replay import build_replay_parser

    p = build_replay_parser()
    args = p.parse_args(["--event", "UFC 329", "-o", "out.csv"])
    assert args.event == "UFC 329"
    assert args.output == Path("out.csv")
