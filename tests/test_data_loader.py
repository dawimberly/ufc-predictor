"""Unit tests for data loading and cleaning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import (
    _collapse_odds_snapshots,
    _fighters_same_person,
    _is_cleaned_fights_df,
    _lookup_odds_for_fight,
    _normalize_odds_frame,
    _read_fights_csv,
    _resolve_winner,
    clean_fighter_name,
    merge_historical_odds,
)
from ufc_betting_bot.modules.edge import as_of_price_from_row, validated_as_of_price


def test_fighters_same_person_last_name_match():
    assert _fighters_same_person("Jon Jones", "Jonathan Jones")
    assert not _fighters_same_person("Jon Jones", "Tom Silva")


def test_fighters_same_person_no_short_substring_false_positive():
    """Short tokens must not match via bare substring."""
    assert not _fighters_same_person("Al", "Alice Fighter")
    assert not _fighters_same_person("Bo", "Bob Fighter")


def test_resolve_winner_canonical_names():
    row = pd.Series(
        {"winner": "Alice Fighter", "fighter1": "Alice Fighter", "fighter2": "Bob Fighter"}
    )
    assert _resolve_winner(row) == "Alice Fighter"


def test_resolve_winner_f1_alias():
    row = pd.Series({"winner": "red", "fighter1": "Alice Fighter", "fighter2": "Bob Fighter"})
    assert _resolve_winner(row) == "Alice Fighter"


def test_is_cleaned_fights_df():
    raw = pd.DataFrame({"fighter_1": ["A"], "fighter_2": ["B"]})
    clean = pd.DataFrame(
        {
            "fight_id": ["x"],
            "fighter1": ["A"],
            "fighter2": ["B"],
            "date": ["2020-01-01"],
        }
    )
    assert not _is_cleaned_fights_df(raw)
    assert _is_cleaned_fights_df(clean)


def test_read_fights_csv_skips_reclean(tmp_path):
    path = tmp_path / "fights.csv"
    pd.DataFrame(
        {
            "fight_id": ["id1", "id2"],
            "fighter1": ["Alice Fighter", "Bob Fighter"],
            "fighter2": ["Bob Fighter", "Carol Fighter"],
            "date": ["2020-01-01", "2020-06-01"],
            "event": ["UFC 1", "UFC 2"],
            "winner": ["Alice Fighter", "Bob Fighter"],
            "weight_class": ["LW", "LW"],
        }
    ).to_csv(path, index=False)

    loaded = _read_fights_csv(path)
    assert len(loaded) == 2
    assert "fighter_1" in loaded.columns
    assert clean_fighter_name(loaded.loc[0, "fighter_1"]) == "Alice Fighter"


def test_normalize_odds_frame_kaggle_jerzyszocik_schema():
    """jerzyszocik daily dataset uses event_date + R_/B_ fighter columns."""
    raw = pd.DataFrame(
        {
            "event_date": ["2025-03-08"],
            "event_name": ["UFC 313"],
            "R_fighter": ["Alex Pereira"],
            "B_fighter": ["Jamahal Hill"],
            "R_odds": [-180.0],
            "B_odds": [150.0],
        }
    )
    odds = _normalize_odds_frame(raw, source="kaggle:jerzyszocik")
    assert len(odds) == 1
    assert odds.loc[0, "f1_odds"] == -180.0
    assert odds.loc[0, "f2_odds"] == 150.0
    assert odds.loc[0, "event"] == "UFC 313"


def test_lookup_odds_for_fight_fuzzy_name_on_date():
    odds = pd.DataFrame(
        {
            "event": [""],
            "date": pd.to_datetime(["2025-01-11"]),
            "fighter1": ["Brandon Royval"],
            "fighter2": ["Manel Kape"],
            "f1_odds": [200.0],
            "f2_odds": [-245.0],
            "f1_opening_odds": [np.nan],
            "f2_opening_odds": [np.nan],
            "f1_closing_odds": [200.0],
            "f2_closing_odds": [-245.0],
            "opening_odds_observed_at": [None],
            "source": ["test"],
        }
    )
    matched = _lookup_odds_for_fight(
        odds,
        event="",
        fight_date=pd.Timestamp("2025-01-11"),
        fighter1="B Royval",
        fighter2="Manel Kape",
    )
    assert matched is not None
    assert matched["f1_odds"] == 200.0
    assert matched["f2_odds"] == -245.0


def test_normalize_odds_frame_ultimate_schema():
    raw = pd.DataFrame(
        {
            "date": ["2025-01-11"],
            "R_fighter": ["Brandon Royval"],
            "B_fighter": ["Manel Kape"],
            "R_odds": [200.0],
            "B_odds": [-245.0],
        }
    )
    odds = _normalize_odds_frame(raw, source="test")
    assert len(odds) == 1
    assert odds.loc[0, "f1_odds"] == 200.0
    assert odds.loc[0, "f2_odds"] == -245.0


def test_lookup_odds_for_fight_by_date_and_fighters():
    odds = pd.DataFrame(
        {
            "event": [""],
            "date": pd.to_datetime(["2025-01-11"]),
            "fighter1": ["Brandon Royval"],
            "fighter2": ["Manel Kape"],
            "f1_odds": [200.0],
            "f2_odds": [-245.0],
            "f1_opening_odds": [np.nan],
            "f2_opening_odds": [np.nan],
            "f1_closing_odds": [200.0],
            "f2_closing_odds": [-245.0],
            "opening_odds_observed_at": [None],
            "source": ["test"],
        }
    )
    matched = _lookup_odds_for_fight(
        odds,
        event="UFC Fight Night: Royval vs. Kape",
        fight_date=pd.Timestamp("2025-01-11"),
        fighter1="Brandon Royval",
        fighter2="Manel Kape",
    )
    assert matched is not None
    assert matched["f1_odds"] == 200.0
    assert matched["f2_odds"] == -245.0


def test_merge_historical_odds_fills_matching_fight(monkeypatch):
    fights = pd.DataFrame(
        {
            "fight_id": ["x1", "x2"],
            "event": ["UFC Fight Night", "UFC 999"],
            "date": pd.to_datetime(["2025-01-11", "2025-01-11"]),
            "fighter1": ["Brandon Royval", "No Match A"],
            "fighter2": ["Manel Kape", "No Match B"],
            "winner": ["", ""],
            "weight_class": ["Flyweight", "LW"],
        }
    )
    odds = pd.DataFrame(
        {
            "event": [""],
            "date": pd.to_datetime(["2025-01-11"]),
            "fighter1": ["Brandon Royval"],
            "fighter2": ["Manel Kape"],
            "f1_odds": [200.0],
            "f2_odds": [-245.0],
            "f1_opening_odds": [np.nan],
            "f2_opening_odds": [np.nan],
            "f1_closing_odds": [200.0],
            "f2_closing_odds": [-245.0],
            "opening_odds_observed_at": [None],
            "source": ["test"],
        }
    )
    monkeypatch.setattr(
        "src.data_loader.build_unified_odds_table",
        lambda: odds,
    )
    merged = merge_historical_odds(fights)
    assert merged.loc[0, "f1_odds"] == 200.0
    assert merged.loc[0, "f2_odds"] == -245.0
    assert merged.loc[0, "f1_closing_odds"] == 200.0
    assert pd.isna(merged.loc[0, "f1_opening_odds"])
    assert pd.isna(merged.loc[0, "opening_odds_observed_at"])
    assert len(merged) == 2
    assert pd.isna(merged.loc[1, "f1_odds"])
    # Closing-only provenance is not strict as-of eligible.
    price, timing = validated_as_of_price(merged.loc[0])
    assert price is None
    assert timing == "closing_not_as_of"


def test_normalize_preserves_snapshot_observation_timestamps():
    raw = pd.DataFrame(
        {
            "event_date": ["2025-03-08", "2025-03-08"],
            "event_name": ["UFC 313", "UFC 313"],
            "fighter_1": ["Alex Pereira", "Alex Pereira"],
            "fighter_2": ["Jamahal Hill", "Jamahal Hill"],
            "odds_1": [1.65, 1.55],
            "odds_2": [2.30, 2.50],
            "quote_timestamp": [
                "2025-02-01T12:00:00+00:00",
                "2025-03-07T18:00:00+00:00",
            ],
        }
    )
    odds = _normalize_odds_frame(raw, source="kaggle:jerzyszocik")
    assert len(odds) == 2
    assert odds.loc[0, "quote_observed_at"] == "2025-02-01T12:00:00+00:00"
    assert pd.isna(odds.loc[0, "f1_opening_odds"])


def test_collapse_daily_snapshots_sets_opening_from_earliest_pre_fight():
    snapshots = pd.DataFrame(
        {
            "event": ["UFC 313", "UFC 313", "UFC 313"],
            "date": pd.to_datetime(["2025-03-08", "2025-03-08", "2025-03-08"]),
            "fighter1": ["Alex Pereira", "Alex Pereira", "Alex Pereira"],
            "fighter2": ["Jamahal Hill", "Jamahal Hill", "Jamahal Hill"],
            "f1_odds": [1.80, 1.65, 1.40],
            "f2_odds": [2.10, 2.30, 3.00],
            "f1_opening_odds": [np.nan, np.nan, np.nan],
            "f2_opening_odds": [np.nan, np.nan, np.nan],
            "f1_closing_odds": [np.nan, np.nan, np.nan],
            "f2_closing_odds": [np.nan, np.nan, np.nan],
            "opening_odds_observed_at": [None, None, None],
            "quote_observed_at": [
                "2025-02-01T12:00:00+00:00",
                "2025-03-07T18:00:00+00:00",
                "2025-03-09T01:00:00+00:00",  # after fight start — ignored
            ],
            "source": ["snap", "snap", "snap"],
        }
    )
    collapsed = _collapse_odds_snapshots(snapshots)
    assert len(collapsed) == 1
    row = collapsed.iloc[0]
    assert float(row["f1_opening_odds"]) == 1.80
    assert float(row["f2_opening_odds"]) == 2.10
    assert float(row["f1_closing_odds"]) == 1.65
    assert float(row["f1_odds"]) == 1.65
    assert row["opening_odds_observed_at"] == "2025-02-01T12:00:00+00:00"


def test_collapse_unlabeled_source_is_closing_only_never_opening():
    unlabeled = pd.DataFrame(
        {
            "event": ["UFC Fight Night"],
            "date": pd.to_datetime(["2025-01-11"]),
            "fighter1": ["Brandon Royval"],
            "fighter2": ["Manel Kape"],
            "f1_odds": [200.0],
            "f2_odds": [-245.0],
            "f1_opening_odds": [np.nan],
            "f2_opening_odds": [np.nan],
            "f1_closing_odds": [np.nan],
            "f2_closing_odds": [np.nan],
            "opening_odds_observed_at": [None],
            "quote_observed_at": [None],
            "source": ["kaggle:ultimate_ufc_dataset"],
        }
    )
    collapsed = _collapse_odds_snapshots(unlabeled)
    row = collapsed.iloc[0]
    assert float(row["f1_closing_odds"]) == 200.0
    assert pd.isna(row["f1_opening_odds"])
    assert row["opening_odds_observed_at"] is None


def test_merge_persists_opening_provenance(monkeypatch):
    fights = pd.DataFrame(
        {
            "fight_id": ["x1"],
            "event": ["UFC 313"],
            "date": pd.to_datetime(["2025-03-08"]),
            "fighter1": ["Alex Pereira"],
            "fighter2": ["Jamahal Hill"],
            "winner": [""],
            "weight_class": ["LHW"],
        }
    )
    odds = pd.DataFrame(
        {
            "event": ["UFC 313"],
            "date": pd.to_datetime(["2025-03-08"]),
            "fighter1": ["Alex Pereira"],
            "fighter2": ["Jamahal Hill"],
            "f1_odds": [1.65],
            "f2_odds": [2.30],
            "f1_opening_odds": [1.80],
            "f2_opening_odds": [2.10],
            "f1_closing_odds": [1.65],
            "f2_closing_odds": [2.30],
            "opening_odds_observed_at": ["2025-02-01T12:00:00+00:00"],
            "source": ["snap"],
        }
    )
    monkeypatch.setattr("src.data_loader.build_unified_odds_table", lambda: odds)
    merged = merge_historical_odds(fights)
    assert float(merged.loc[0, "f1_opening_odds"]) == 1.80
    assert merged.loc[0, "opening_odds_observed_at"] == "2025-02-01T12:00:00+00:00"
    price = as_of_price_from_row(merged.loc[0])
    assert price is not None
    assert price.timing == "opening"
