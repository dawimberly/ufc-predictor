"""Leakage-safety tests for as-of career stats and same-card Elo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import _compute_elo_state, _rolling_stats
from src.sherdog import sherdog_record_as_of


def test_sherdog_refuses_profile_wl_without_dated_fights(tmp_path, monkeypatch):
    monkeypatch.setattr("src.sherdog.SHERDOG_FIGHTERS_CACHE", tmp_path / "fighters.csv")
    monkeypatch.setattr("src.sherdog.SHERDOG_FIGHTS_CACHE", tmp_path / "fights.csv")
    monkeypatch.setattr("src.sherdog.SHERDOG_INDEX_CACHE", tmp_path / "index.json")
    pd.DataFrame(
        [
            {
                "name": "No History",
                "sherdog_id": "99",
                "url": "",
                "nickname": "",
                "weight_class": "LW",
                "height_in": 70,
                "reach_in": 72,
                "birth_date": "",
                "nationality": "",
                "team": "",
                "wins": 20,
                "losses": 2,
                "draws": 0,
                "source": "sherdog",
                "fetched_at": "",
            }
        ]
    ).to_csv(tmp_path / "fighters.csv", index=False)
    pd.DataFrame(
        columns=[
            "sherdog_id",
            "fighter",
            "opponent",
            "result",
            "method",
            "event",
            "bout_date",
            "weight_class",
            "source",
        ]
    ).to_csv(tmp_path / "fights.csv", index=False)

    rec = sherdog_record_as_of("No History", "2024-06-01")
    assert pd.isna(rec["sherdog_wins"])
    assert pd.isna(rec["sherdog_win_rate"])
    assert pd.isna(rec["sherdog_fight_count"])


def test_sherdog_requires_as_of_date(tmp_path, monkeypatch):
    monkeypatch.setattr("src.sherdog.SHERDOG_FIGHTERS_CACHE", tmp_path / "fighters.csv")
    monkeypatch.setattr("src.sherdog.SHERDOG_FIGHTS_CACHE", tmp_path / "fights.csv")
    monkeypatch.setattr("src.sherdog.SHERDOG_INDEX_CACHE", tmp_path / "index.json")
    pd.DataFrame(
        [
            {
                "sherdog_id": "1",
                "fighter": "Jane Fighter",
                "opponent": "Alice",
                "result": "WIN",
                "method": "KO",
                "event": "E1",
                "bout_date": "2023-01-01",
                "weight_class": "Flyweight",
                "source": "sherdog",
            }
        ]
    ).to_csv(tmp_path / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Jane Fighter",
                "sherdog_id": "1",
                "url": "",
                "nickname": "",
                "weight_class": "Flyweight",
                "height_in": 64,
                "reach_in": 64,
                "birth_date": "",
                "nationality": "",
                "team": "",
                "wins": 10,
                "losses": 0,
                "draws": 0,
                "source": "sherdog",
                "fetched_at": "",
            }
        ]
    ).to_csv(tmp_path / "fighters.csv", index=False)

    rec = sherdog_record_as_of("Jane Fighter", None)
    assert pd.isna(rec["sherdog_wins"])


def test_same_card_elo_frozen_at_card_open():
    """Later same-night bouts must not see earlier same-night Elo updates."""
    history = pd.DataFrame(
        [
            {
                "fight_id": "a",
                "event_date": "2024-01-01",
                "fighter": "A",
                "fighter_1": "A",
                "fighter_2": "B",
                "winner": "A",
                "side": 1,
                "won": 1,
            },
            {
                "fight_id": "a",
                "event_date": "2024-01-01",
                "fighter": "B",
                "fighter_1": "A",
                "fighter_2": "B",
                "winner": "A",
                "side": 2,
                "won": 0,
            },
            {
                "fight_id": "b",
                "event_date": "2024-01-01",
                "fighter": "A",
                "fighter_1": "A",
                "fighter_2": "C",
                "winner": "C",
                "side": 1,
                "won": 0,
            },
            {
                "fight_id": "b",
                "event_date": "2024-01-01",
                "fighter": "C",
                "fighter_1": "A",
                "fighter_2": "C",
                "winner": "C",
                "side": 2,
                "won": 1,
            },
        ]
    )
    elo_df, _ = _compute_elo_state(history)
    by_id = elo_df.set_index("fight_id")
    # Both same-day fights start from equal default Elo for A.
    assert abs(float(by_id.loc["a", "f1_elo"]) - float(by_id.loc["b", "f1_elo"])) < 1e-9


def test_rolling_stats_ignore_career_static_columns():
    """Career-wide *_static strike fields must not enter rolling history features."""
    long = pd.DataFrame(
        {
            "fight_id": ["1", "1", "2", "2"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2024-01-01", "2024-01-01"]),
            "fighter": ["A", "B", "A", "C"],
            "fighter_1": ["A", "A", "A", "A"],
            "fighter_2": ["B", "B", "C", "C"],
            "winner": ["A", "A", "A", "A"],
            "side": [1, 2, 1, 2],
            "won": [1, 0, 1, 0],
            "ko_win": [0, 0, 0, 0],
            "sub_win": [0, 0, 0, 0],
            "finish": [0, 0, 0, 0],
            "sig_strike_acc_static": [0.99, 0.99, 0.99, 0.99],
            "td_acc_static": [0.99, 0.99, 0.99, 0.99],
            "sub_avg_static": [5.0, 5.0, 5.0, 5.0],
        }
    )
    out = _rolling_stats(long)
    assert out["sig_strike_acc"].isna().all()
    assert out["td_acc"].isna().all()
    assert "sub_avg" in out.columns
