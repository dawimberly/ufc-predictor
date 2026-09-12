"""Execution-book overlay (DraftKings / MyBookie) + totals point-safety."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src.high_accuracy_strategy import ALLOWED_PROP_KEYS
from src.odds_providers.prop_odds_common import totals_prop_key
from src.predictor import attach_execution_book_odds


def test_over_2_5_is_not_over_1_5():
    # 1.5 line is the only HA-actionable total.
    assert totals_prop_key(1.5, "over") == ("over_1_5_rounds", "Over 1.5")
    assert totals_prop_key(1.5, "under") == ("round_1_finish", "Under 1.5")
    assert totals_prop_key(1.53, "over")[0] == "over_1_5_rounds"  # tolerance 0.05

    pk, sel = totals_prop_key(2.5, "over")
    assert pk == "over_2.5_rounds"
    assert sel == "Over 2.5"
    assert pk not in ALLOWED_PROP_KEYS  # a 2.5 total is a different bet, not HA-sizable

    pk_u, sel_u = totals_prop_key(2.5, "under")
    assert pk_u == "under_2.5_rounds"
    assert pk_u not in ALLOWED_PROP_KEYS


def _preds():
    return pd.DataFrame(
        [
            {
                "fighter_1": "Alpha",
                "fighter_2": "Zulu",
                "prob_f1_win": 0.70,
                "prob_f2_win": 0.30,
                "f1_odds": 1.80,
                "f2_odds": 2.10,
                "pick": "Alpha",
            }
        ]
    )


def test_overlay_edge_dk_and_no_mutation():
    dk = pd.DataFrame(
        [
            {
                "fighter_1": "Alpha",
                "fighter_2": "Zulu",
                "f1_odds": 2.00,
                "f2_odds": 1.90,
                "implied_prob_f1": 0.50,
                "implied_prob_f2": 0.5263,
                "bookmaker": "draftkings",
            }
        ]
    )
    out = attach_execution_book_odds(_preds(), dk_odds=dk, fetch_if_missing=False)
    row = out.iloc[0]
    # Pick side = f1 (p=0.70). edge_dk = 0.70 * 2.00 - 1 = 0.40
    assert row["odds_dk"] == 2.00
    assert row["edge_dk"] == pytest.approx(0.40, abs=1e-6)
    # Consensus edge_api = 0.70 * 1.80 - 1 = 0.26
    assert row["odds_api"] == 1.80
    assert row["edge_api"] == pytest.approx(0.26, abs=1e-6)
    # Never mutates pick / model prob / consensus price
    assert row["prob_f1_win"] == 0.70
    assert row["pick"] == "Alpha"
    assert row["f1_odds"] == 1.80


def test_overlay_missing_book_is_blank_not_failclosed():
    out = attach_execution_book_odds(_preds(), dk_odds=None, mybookie_odds=None, fetch_if_missing=False)
    row = out.iloc[0]
    assert pd.isna(row["odds_dk"]) and pd.isna(row["edge_dk"])
    assert pd.isna(row["odds_mb"]) and pd.isna(row["edge_mb"])
    # Consensus edge still attached (row not fail-closed just because a book is missing)
    assert row["edge_api"] == pytest.approx(0.26, abs=1e-6)


def test_dk_enabled_fetch_once_uses_cache_no_network(tmp_path, monkeypatch):
    from src.odds_providers import draftkings

    cache = tmp_path / "draftkings_odds.csv"
    pd.DataFrame(
        [
            {
                "fighter_1": "Alpha",
                "fighter_2": "Zulu",
                "f1_odds": 2.0,
                "f2_odds": 1.9,
                "implied_prob_f1": 0.50,
                "implied_prob_f2": 0.526,
                "bookmaker": "draftkings",
                "commence_time": "2026-09-12T00:00:00Z",
            }
        ]
    ).to_csv(cache, index=False)

    monkeypatch.setattr(draftkings, "DK_CACHE_PATH", cache, raising=False)
    monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True, raising=False)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key", raising=False)

    def _boom(*a, **k):
        raise AssertionError("network call not allowed when fetch-once cache is fresh")

    monkeypatch.setattr(draftkings.requests, "get", _boom, raising=True)

    out = draftkings.fetch_draftkings_odds()
    assert not out.empty
    assert list(out["fighter_1"]) == ["Alpha"]
