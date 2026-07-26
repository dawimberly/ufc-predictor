"""Odds API dashboard wiring: loaders, prop summary, fail-closed labels."""

from __future__ import annotations

import pandas as pd

import config
from src.dashboard_service import BOOK_LOADERS, active_book_loaders
from src.odds_providers.prop_odds_common import prop_odds_summary
from src.props import is_live_prop_odds_source
from src.strategy import attach_prop_stakes, budget_aware_alerts


def test_odds_api_always_in_active_loaders(monkeypatch):
    monkeypatch.setattr(config, "BETNOW_ENABLED", False)
    monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", False)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)
    loaders = active_book_loaders()
    assert "Odds API" in loaders
    assert loaders["Odds API"] == BOOK_LOADERS["Odds API"]
    assert "BetNow.eu" not in loaders
    assert "DraftKings" not in loaders
    assert "MyBookie" not in loaders


def test_scraper_loaders_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "BETNOW_ENABLED", True)
    monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", True)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", True)
    loaders = active_book_loaders()
    assert set(loaders) >= {"Odds API", "BetNow.eu", "DraftKings", "MyBookie"}


def test_prop_odds_summary_counts_the_odds_api():
    df = pd.DataFrame(
        {
            "odds_source": ["the_odds_api", "live", "synthetic"],
        }
    )
    summary = prop_odds_summary(df)
    assert summary["live"] == 2
    assert summary["synthetic"] == 1


def test_is_live_prop_odds_source_odds_api():
    assert is_live_prop_odds_source("the_odds_api")
    assert is_live_prop_odds_source("live")
    assert not is_live_prop_odds_source("synthetic")


def test_odds_api_budget_alerts_not_disabled():
    alerts = {
        "singles": [
            {
                "pick": "A",
                "fight": "A vs B",
                "edge": 0.1,
                "edge_pct": 10.0,
                "odds": 2.0,
                "kelly_fraction": 0.05,
                "suggested_stake": 0.0,
            }
        ],
        "parlays": [],
    }
    budget = config.default_budget_state()
    budget["use_betnow"] = False
    budget["use_draftkings"] = False
    budget["use_mybookie"] = False
    out = budget_aware_alerts(alerts, budget, "Odds API")
    assert not out.get("book_disabled")
    assert out.get("singles")


def test_odds_api_prop_stakes_not_disabled():
    singles = [
        {
            "pick": "Over 1.5",
            "fight": "A vs B",
            "edge": 0.08,
            "odds": 1.9,
            "prob": 0.6,
            "odds_source": "the_odds_api",
        }
    ]
    budget = config.default_budget_state()
    budget["use_betnow"] = False
    budget["use_draftkings"] = False
    budget["use_mybookie"] = False
    out = attach_prop_stakes(singles, budget, "Odds API")
    assert out
    assert not out[0].get("book_disabled")


def test_book_prop_rules_include_odds_api():
    assert "Odds API" in config.BOOK_PROP_RULES
    assert config.BOOK_PROP_RULES["Odds API"].get("allow_prop_parlays") is False
