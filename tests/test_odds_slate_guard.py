"""Card-slate fingerprint invalidates stale Odds API caches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.odds_providers.odds_slate_guard import (
    combined_slate_fingerprint,
    invalidate_odds_caches_for_slate_change,
    min_match_for_card,
    save_odds_slate_fingerprint,
)


def test_fingerprint_stable_and_order_independent():
    a = pd.DataFrame(
        {
            "fighter_1": ["Islam Makhachev", "Neil Magny"],
            "fighter_2": ["Ian Machado Garry", "Ramiz Brahimaj"],
            "event_name": ["UFC 330", "UFC 330"],
        }
    )
    b = a.iloc[::-1].reset_index(drop=True)
    assert combined_slate_fingerprint(a) == combined_slate_fingerprint(b)
    assert combined_slate_fingerprint(a)


def test_fingerprint_changes_when_roster_changes():
    a = pd.DataFrame(
        {"fighter_1": ["A"], "fighter_2": ["B"], "event_name": ["UFC 1"]}
    )
    b = pd.DataFrame(
        {"fighter_1": ["A"], "fighter_2": ["C"], "event_name": ["UFC 1"]}
    )
    assert combined_slate_fingerprint(a) != combined_slate_fingerprint(b)


def test_invalidate_clears_stale_moneyline_cache(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    ml = cache / "ufc_odds_api.csv"
    ml.write_text("event_id,fighter_1,fighter_2\nx,a,b\n", encoding="utf-8")
    fp = cache / "odds_slate_fingerprint.txt"
    fp.write_text("oldslatehash1234\n", encoding="utf-8")
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "ODDS_CACHE_PATH", ml)

    combined = pd.DataFrame(
        {"fighter_1": ["A"], "fighter_2": ["B"], "event_name": ["UFC 1"]}
    )
    assert invalidate_odds_caches_for_slate_change(combined) is True
    assert not ml.is_file()


def test_min_match_for_card():
    assert min_match_for_card(0) == 0
    assert min_match_for_card(5) == 3
    assert min_match_for_card(12) == 6


def test_save_fingerprint_writes_marker(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    combined = pd.DataFrame(
        {"fighter_1": ["A"], "fighter_2": ["B"], "event_name": ["UFC 1"]}
    )
    digest = save_odds_slate_fingerprint(combined)
    assert digest
    marker = cache / "odds_slate_fingerprint.txt"
    assert marker.read_text(encoding="utf-8").strip() == digest
