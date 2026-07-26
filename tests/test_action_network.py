"""Action Network UFC odds backup (fail-soft)."""

from __future__ import annotations

import pandas as pd

import config
from src.odds_providers.action_network import _parse_scoreboard, fetch_action_network_odds


def test_parse_scoreboard_moneyline():
    payload = {
        "competitions": [
            {
                "id": 1,
                "start_time": "2026-08-01T00:00:00Z",
                "meta": {"title": "UFC Test"},
                "competitors": [
                    {
                        "side": "home",
                        "player": {"full_name": "Home Fighter"},
                    },
                    {
                        "side": "away",
                        "player": {"full_name": "Away Fighter"},
                    },
                ],
                "odds": [
                    {"ml_home": -200, "ml_away": 170, "book_id": 1},
                    {"ml_home": -190, "ml_away": 160, "book_id": 2},
                ],
            }
        ]
    }
    df = _parse_scoreboard(payload)
    assert len(df) == 1
    assert df.iloc[0]["fighter_1"] == "Home Fighter"
    assert df.iloc[0]["fighter_2"] == "Away Fighter"
    assert float(df.iloc[0]["f1_odds"]) > 1.0
    assert float(df.iloc[0]["f2_odds"]) > 1.0


def test_fetch_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", False)
    df = fetch_action_network_odds(force_refresh=True)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_fail_soft_on_http_error(monkeypatch):
    monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", True)

    def boom(*args, **kwargs):
        raise RuntimeError("blocked")

    monkeypatch.setattr("src.odds_providers.action_network.requests.get", boom)
    df = fetch_action_network_odds(force_refresh=True)
    assert df.empty
