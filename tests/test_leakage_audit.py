"""Leakage-safety tests for as-of career stats and same-card Elo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data_loader import _add_pipeline_aliases
from src.feature_engineering import (
    _compute_elo_state,
    _rolling_stats,
    apply_historical_stat_fallbacks,
    build_feature_matrix,
)
from src.sherdog import sherdog_record_as_of

FIXTURE = Path(__file__).parent / "fixtures" / "sample_fights.csv"


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


def test_physical_attrs_do_not_backfill_from_future_fights():
    """Later height/reach/age must not fill earlier missing rows."""
    long = pd.DataFrame(
        {
            "fight_id": ["1", "1", "2", "2"],
            "event_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01"]),
            "fighter": ["A", "B", "A", "C"],
            "fighter_1": ["A", "A", "A", "A"],
            "fighter_2": ["B", "B", "C", "C"],
            "winner": ["A", "A", "A", "A"],
            "side": [1, 2, 1, 2],
            "won": [1, 0, 1, 0],
            "ko_win": [0, 0, 0, 0],
            "sub_win": [0, 0, 0, 0],
            "finish": [0, 0, 0, 0],
            "height_in": [np.nan, 70.0, 69.0, 66.0],
            "reach_in": [np.nan, 71.0, 72.0, 65.0],
            "age": [np.nan, 28.0, 31.0, 27.0],
        }
    )
    out = _rolling_stats(long)
    early_a = out.loc[(out["fighter"] == "A") & (out["fight_id"] == "1")].iloc[0]
    late_a = out.loc[(out["fighter"] == "A") & (out["fight_id"] == "2")].iloc[0]
    assert pd.isna(early_a["height_in"])
    assert pd.isna(early_a["reach_in"])
    assert pd.isna(early_a["age"])
    assert float(late_a["height_in"]) == 69.0


def _feature_values_equal(a, b, *, tol: float = 1e-9) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def test_feature_columns_match_truncated_history_rebuild():
    """Every FEATURE_COLUMNS value must match a past-only rebuild of the matrix.

    Walk-forward loads a precomputed full-history CSV; this equality check is the
    proof that those values do not depend on later fights.
    """
    fights = _add_pipeline_aliases(pd.read_csv(FIXTURE, parse_dates=["date"]))
    full = build_feature_matrix(fights)
    assert not full.empty

    scored_ids = list(full["fight_id"])
    # Mid and late scored fights exercise rolling / Elo / SOS against prior history.
    probe_ids = [scored_ids[0], scored_ids[len(scored_ids) // 2], scored_ids[-1]]
    cols = [c for c in config.FEATURE_COLUMNS if c in full.columns]
    assert cols, "FEATURE_COLUMNS missing from feature matrix"

    full_by_id = full.set_index("fight_id")
    for fight_id in probe_ids:
        target_date = pd.Timestamp(fights.loc[fights["fight_id"] == fight_id, "date"].iloc[0])
        truncated = fights[
            (fights["date"] < target_date) | (fights["fight_id"] == fight_id)
        ].copy()
        rebuilt = build_feature_matrix(truncated)
        assert fight_id in set(rebuilt["fight_id"]), fight_id
        full_row = full_by_id.loc[fight_id]
        trunc_row = rebuilt.set_index("fight_id").loc[fight_id]
        mismatches = [
            c
            for c in cols
            if not _feature_values_equal(full_row[c], trunc_row[c])
        ]
        assert mismatches == [], f"{fight_id} leaked via {mismatches[:12]}"


def test_reference_year_fallback_does_not_fill_earlier_rows():
    """Pre-2025 medians must not be written onto earlier historical rows."""
    features = pd.DataFrame(
        {
            "event_date": ["2020-01-01", "2024-06-01", "2025-02-01"],
            "weight_class": ["Lightweight", "Lightweight", "Lightweight"],
            "fighter_1": ["A", "B", "C"],
            "fighter_2": ["D", "E", "F"],
            "f1_age": [30.0, np.nan, np.nan],
            "f2_age": [28.0, 29.0, np.nan],
        }
    )
    out = apply_historical_stat_fallbacks(features, reference_year=2025)
    assert pd.isna(out.loc[1, "f1_age"])
    assert float(out.loc[2, "f1_age"]) == 30.0


def test_reference_year_fallback_leaves_missing_without_earlier_rows():
    """No pre-reference rows means no fill, including from the rows being scored."""
    features = pd.DataFrame(
        {
            "event_date": ["2025-03-01", "2025-04-01"],
            "weight_class": ["Lightweight", "Lightweight"],
            "fighter_1": ["A", "B"],
            "fighter_2": ["C", "D"],
            "f1_age": [np.nan, 33.0],
            "f2_age": [np.nan, np.nan],
        }
    )
    out = apply_historical_stat_fallbacks(features, reference_year=2025)
    assert pd.isna(out.loc[0, "f1_age"])
    assert float(out.loc[1, "f1_age"]) == 33.0
    assert pd.isna(out.loc[0, "f2_age"])
    assert pd.isna(out.loc[1, "f2_age"])


def test_undated_result_does_not_move_elo():
    """A fight with no date cannot update ratings or borrow a later rating."""
    from src.feature_engineering import ELO_START

    def _bouts(include_undated: bool) -> pd.DataFrame:
        rows = [
            {
                "fight_id": "dated-early",
                "event_date": "2020-01-01",
                "fighter_1": "Alex",
                "fighter_2": "Ben",
                "winner": "Alex",
                "side": 1,
            },
            {
                "fight_id": "dated-late",
                "event_date": "2024-06-01",
                "fighter_1": "Alex",
                "fighter_2": "Drew",
                "winner": "Drew",
                "side": 1,
            },
        ]
        if include_undated:
            rows.append(
                {
                    "fight_id": "no-date",
                    "event_date": None,
                    "fighter_1": "Alex",
                    "fighter_2": "Casey",
                    "winner": "Alex",
                    "side": 1,
                }
            )
        return pd.DataFrame(rows)

    with_undated, ratings_with = _compute_elo_state(_bouts(True))
    without, ratings_without = _compute_elo_state(_bouts(False))
    dated = with_undated[with_undated["fight_id"] != "no-date"].set_index("fight_id")
    base = without.set_index("fight_id")
    for fight_id in ("dated-early", "dated-late"):
        assert abs(float(dated.loc[fight_id, "f1_elo"]) - float(base.loc[fight_id, "f1_elo"])) < 1e-9
        assert abs(float(dated.loc[fight_id, "f2_elo"]) - float(base.loc[fight_id, "f2_elo"])) < 1e-9
    undated = with_undated.set_index("fight_id").loc["no-date"]
    assert float(undated["f1_elo"]) == ELO_START
    assert float(undated["f2_elo"]) == ELO_START
    assert abs(float(ratings_with["Alex"]) - float(ratings_without["Alex"])) < 1e-9


def test_undated_gym_snapshot_skips_labeled_history(tmp_path, monkeypatch):
    from src.gym_data import attach_gym_features, clear_gym_cache

    csv_path = tmp_path / "gyms.csv"
    pd.DataFrame(
        [
            {
                "fighter_name": "Alex Pereira",
                "gym": "Current Camp",
                "location": "Las Vegas",
                "strengths": "striking",
                "notes": "",
            }
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr("src.gym_data._gyms_csv_path", lambda: csv_path)
    clear_gym_cache()
    historical = pd.DataFrame(
        [
            {
                "fighter_1": "Alex Pereira",
                "fighter_2": "Ciryl Gane",
                "event_date": "2020-01-01",
                "f1_win": 1,
                "location": "Las Vegas",
            }
        ]
    )
    blocked = attach_gym_features(historical)
    assert blocked.loc[0, "f1_gym"] == ""
    assert int(blocked.loc[0, "f1_local_advantage"]) == 0

    upcoming = historical.drop(columns=["f1_win"])
    shown = attach_gym_features(upcoming)
    assert shown.loc[0, "f1_gym"] == "Current Camp"

    dated = csv_path
    pd.DataFrame(
        [
            {
                "fighter_name": "Alex Pereira",
                "gym": "Old Camp",
                "location": "Las Vegas",
                "strengths": "",
                "notes": "",
                "as_of": "2019-01-01",
            }
        ]
    ).to_csv(dated, index=False)
    clear_gym_cache()
    known = attach_gym_features(historical)
    assert known.loc[0, "f1_gym"] == "Old Camp"

    pd.DataFrame(
        [
            {
                "fighter_name": "Alex Pereira",
                "gym": "Later Camp",
                "location": "Las Vegas",
                "strengths": "",
                "notes": "",
                "as_of": "2024-01-01",
            }
        ]
    ).to_csv(dated, index=False)
    clear_gym_cache()
    future = attach_gym_features(historical)
    assert future.loc[0, "f1_gym"] == ""
