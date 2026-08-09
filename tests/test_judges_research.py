"""Tests for decision profile + mmadecisions parse + judge deviation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.decision_profile import apply_decision_profile_rolling, _method_is_split
from src.judge_scoring_deviation import expand_round_rows, judge_deviation_summary
from src.mmadecisions import parse_decision_html


def test_split_method_flags() -> None:
    assert _method_is_split("Decision - Split")
    assert _method_is_split("S-DEC")
    assert not _method_is_split("Decision - Unanimous")


def test_decision_profile_rolling_smoke() -> None:
    hist = pd.DataFrame(
        {
            "fighter": ["A", "A", "A", "A"],
            "won": [1, 0, 1, 1],
            "is_dec": [1, 1, 0, 1],
            "method": [
                "Decision - Unanimous",
                "Decision - Split",
                "KO/TKO",
                "Decision - Split",
            ],
        }
    )
    out = apply_decision_profile_rolling(hist)
    assert "dec_win_rate_career" in out.columns
    assert "split_dec_win_rate_career" in out.columns
    assert "decision_finish_share_career" in out.columns
    # First row has no prior
    assert pd.isna(out.loc[0, "dec_win_rate_career"]) or out.loc[0, "dec_win_rate_career"] == out.loc[0, "dec_win_rate_career"]


def test_parse_mmadecisions_sample_html() -> None:
    path = Path("data/cache/mmadecisions_sample.html")
    if not path.is_file():
        return  # optional if inspect script not run
    row = parse_decision_html(path.read_text(encoding="utf-8"), decision_id=14180)
    assert row is not None
    assert row["is_ufc"] is True
    assert row["n_judges"] >= 3
    assert row["n_rounds"] >= 3
    assert all(len(j["rounds"]) == row["n_rounds"] for j in row["judges"])


def test_judge_deviation_shrink() -> None:
    decisions = [
        {
            "decision_id": 1,
            "is_ufc": True,
            "fighter_1": "A",
            "fighter_2": "B",
            "n_rounds": 3,
            "n_judges": 3,
            "judges": [
                {
                    "judge_id": 1,
                    "judge_name": "Judge One",
                    "rounds": [
                        {"round": 1, "score_f1": 10, "score_f2": 9},
                        {"round": 2, "score_f1": 9, "score_f2": 10},
                        {"round": 3, "score_f1": 10, "score_f2": 9},
                    ],
                },
                {
                    "judge_id": 2,
                    "judge_name": "Judge Two",
                    "rounds": [
                        {"round": 1, "score_f1": 10, "score_f2": 9},
                        {"round": 2, "score_f1": 10, "score_f2": 9},
                        {"round": 3, "score_f1": 10, "score_f2": 9},
                    ],
                },
                {
                    "judge_id": 3,
                    "judge_name": "Judge Three",
                    "rounds": [
                        {"round": 1, "score_f1": 10, "score_f2": 9},
                        {"round": 2, "score_f1": 10, "score_f2": 9},
                        {"round": 3, "score_f1": 10, "score_f2": 9},
                    ],
                },
            ],
        }
    ]
    rounds = expand_round_rows(decisions)
    summary = judge_deviation_summary(rounds, min_rounds=50)
    assert not summary.empty
    # Small-N → not reliable
    assert bool(summary.iloc[0]["reliable"]) is False
    assert "disagreement_rate_shrunk" in summary.columns
