"""DraftKings moneylines are sliced from the main Odds API payload."""

from __future__ import annotations

import pandas as pd

import config
from src.odds_providers.draftkings import (
    fetch_draftkings_odds,
    persist_draftkings_from_payload,
    rows_from_odds_api_events,
)


def _event() -> dict:
    return {
        "id": "evt1",
        "commence_time": "2026-10-17T21:00:00Z",
        "home_team": "Chad Anheliger",
        "away_team": "Steven Koslow",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Chad Anheliger", "price": 2.4},
                            {"name": "Steven Koslow", "price": 1.6},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Chad Anheliger", "price": 2.45},
                            {"name": "Steven Koslow", "price": 1.58},
                        ],
                    }
                ],
            },
        ],
    }


def test_rows_keep_only_draftkings():
    df = rows_from_odds_api_events([_event()])
    assert len(df) == 1
    assert df.iloc[0]["bookmaker"] == "DraftKings"
    assert df.iloc[0]["f1_odds"] == 2.45
    assert df.iloc[0]["f2_odds"] == 1.58


def test_persist_and_fetch_use_cache_without_http(tmp_path, monkeypatch):
    cache = tmp_path / "draftkings_odds.csv"
    monkeypatch.setattr("src.odds_providers.draftkings.DK_CACHE_PATH", cache)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)
    monkeypatch.setattr(config, "ODDS_API_ODDS_FORMAT", "decimal")

    saved = persist_draftkings_from_payload([_event()])
    assert len(saved) == 1
    assert cache.is_file()

    def _no_http(*_args, **_kwargs):
        raise AssertionError("DraftKings fetch should reuse the saved cache")

    monkeypatch.setattr("src.odds_providers.draftkings.requests.get", _no_http)
    loaded = fetch_draftkings_odds(force_refresh=False)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 1
    assert loaded.iloc[0]["bookmaker"] == "DraftKings"
    assert float(loaded.iloc[0]["f1_odds"]) == 2.45
