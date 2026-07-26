"""Automatic odds fallback chain (free sources first)."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src.odds_providers.odds_fallback import (
    NO_USABLE_ODDS_MSG,
    fetch_best_available_odds,
    fetch_book_scraper_odds,
    odds_frame_usable,
    _normalize_odds_frame,
)
from src.predictor import merge_predictions_with_odds


def _odds_df(book: str, n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "fighter_1": f"Fighter A{i}",
                "fighter_2": f"Fighter B{i}",
                "f1_odds": 1.80 + i * 0.1,
                "f2_odds": 2.10 - i * 0.05,
                "bookmaker": book,
                "bookmaker_count": 1,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_rejects_bad_odds():
    bad = pd.DataFrame(
        [{"fighter_1": "A", "fighter_2": "B", "f1_odds": 0.5, "f2_odds": 2.0}]
    )
    assert _normalize_odds_frame(bad, bookmaker="X").empty
    good = _odds_df("BetNow.eu", 1)
    assert odds_frame_usable(good)


def test_fetch_best_available_odds_api_to_action_network(monkeypatch):
    calls: list[str] = []

    def boom_api(**kwargs):
        calls.append("api")
        raise RuntimeError("401 Unauthorized")

    def ok_an(**kwargs):
        calls.append("an")
        return _odds_df("ActionNetwork", 2)

    monkeypatch.setattr("src.predictor.fetch_ufc_odds", boom_api)
    monkeypatch.setattr(
        "src.odds_providers.action_network.fetch_action_network_odds", ok_an
    )
    monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", True)
    monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", False)
    monkeypatch.setattr(config, "BETNOW_ENABLED", False)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)

    df, meta = fetch_best_available_odds(force_refresh=True)
    assert not df.empty
    assert meta["fail_closed"] is False
    assert meta["source"] == "ActionNetwork"
    assert "OddsAPI" in meta["sources_tried"]
    assert calls == ["api", "an"]


def test_fetch_best_available_optional_betnow(monkeypatch):
    calls: list[str] = []

    def boom_api(**kwargs):
        calls.append("api")
        raise RuntimeError("401 Unauthorized")

    def boom_an(**kwargs):
        calls.append("an")
        return pd.DataFrame()

    def ok_betnow(**kwargs):
        calls.append("betnow")
        return _odds_df("BetNow.eu", 2)

    monkeypatch.setattr("src.predictor.fetch_ufc_odds", boom_api)
    monkeypatch.setattr(
        "src.odds_providers.action_network.fetch_action_network_odds", boom_an
    )
    monkeypatch.setattr(
        "src.odds_providers.betnow_scraper.fetch_betnow_odds", ok_betnow
    )
    monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", True)
    monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", False)
    monkeypatch.setattr(config, "BETNOW_ENABLED", True)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)

    df, meta = fetch_best_available_odds(force_refresh=True)
    assert not df.empty
    assert meta["source"] == "BetNow.eu"
    assert "betnow" in calls


def test_fail_closed_when_all_empty(monkeypatch):
    monkeypatch.setattr(
        "src.predictor.fetch_ufc_odds",
        lambda **k: (_ for _ in ()).throw(RuntimeError("401")),
    )
    monkeypatch.setattr(
        "src.odds_providers.action_network.fetch_action_network_odds",
        lambda **k: pd.DataFrame(),
    )
    monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", True)
    monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", False)
    monkeypatch.setattr(config, "BETNOW_ENABLED", False)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)

    df, meta = fetch_best_available_odds(force_refresh=True)
    assert df.empty
    assert meta["fail_closed"] is True
    assert meta["n_rows"] == 0
    assert NO_USABLE_ODDS_MSG in str(meta.get("warning") or "")
    assert meta.get("no_usable_odds") is True


def test_merge_predictions_uses_fallback(monkeypatch):
    preds = pd.DataFrame(
        [
            {
                "fighter_1": "Fighter A0",
                "fighter_2": "Fighter B0",
                "prob_f1_win": 0.62,
                "prob_f2_win": 0.38,
                "predicted_winner": "Fighter A0",
                "predicted_prob": 0.62,
            }
        ]
    )

    def fake_best(**kwargs):
        return _odds_df("ActionNetwork", 1), {
            "source": "ActionNetwork",
            "sources_tried": ["OddsAPI", "ActionNetwork"],
            "warning": "Using Action Network",
            "fail_closed": False,
            "n_rows": 1,
        }

    monkeypatch.setattr(
        "src.odds_providers.odds_fallback.fetch_best_available_odds",
        fake_best,
    )
    out = merge_predictions_with_odds(preds, odds=None, fetch_if_missing=True)
    assert bool(out.iloc[0]["odds_matched"]) is True
    assert float(out.iloc[0]["f1_odds"]) == pytest.approx(1.80)
    assert float(out.iloc[0]["edge_f1"]) == pytest.approx(0.62 - 1 / 1.80, rel=1e-3)


def test_merge_fail_closed_no_books(monkeypatch):
    preds = pd.DataFrame(
        [
            {
                "fighter_1": "A",
                "fighter_2": "B",
                "prob_f1_win": 0.55,
                "prob_f2_win": 0.45,
            }
        ]
    )
    monkeypatch.setattr(
        "src.odds_providers.odds_fallback.fetch_best_available_odds",
        lambda **k: (
            pd.DataFrame(),
            {
                "source": "",
                "fail_closed": True,
                "warning": NO_USABLE_ODDS_MSG,
                "sources_tried": [],
                "n_rows": 0,
                "no_usable_odds": True,
            },
        ),
    )
    out = merge_predictions_with_odds(preds, fetch_if_missing=True)
    assert bool(out.iloc[0]["odds_matched"]) is False
    assert bool(out.iloc[0].get("no_usable_odds")) is True


def test_scraper_priority_betnow_before_mybookie(monkeypatch):
    order: list[str] = []

    monkeypatch.setattr(
        "src.odds_providers.betnow_scraper.fetch_betnow_odds",
        lambda **k: (order.append("betnow") or _odds_df("BetNow.eu", 1)),
    )
    monkeypatch.setattr(
        "src.odds_providers.mybookie_scraper.fetch_mybookie_odds",
        lambda **k: (order.append("mybookie") or _odds_df("MyBookie", 1)),
    )
    monkeypatch.setattr(config, "BETNOW_ENABLED", True)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", True)
    df = fetch_book_scraper_odds(force_refresh=True)
    assert not df.empty
    assert order == ["betnow"]  # stops at first success


def test_book_scrapers_off_by_default(monkeypatch):
    monkeypatch.setattr(config, "BETNOW_ENABLED", False)
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)
    df = fetch_book_scraper_odds(force_refresh=True)
    assert df.empty
