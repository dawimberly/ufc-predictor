"""Odds cache reuse with ODDS_FETCH_ONCE."""

from __future__ import annotations

from pathlib import Path

import config
from src.predictor import _odds_cache_fresh


def test_odds_fetch_once_reuses_cache_ignoring_age(tmp_path: Path, monkeypatch):
    cache = tmp_path / "ufc_odds_api.csv"
    cache.write_text("event_id,fighter_1,fighter_2\nx,a,b\n", encoding="utf-8")
    monkeypatch.setattr(config, "ODDS_CACHE_PATH", cache)
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)
    monkeypatch.setattr(config, "ODDS_CACHE_TTL_HOURS", 0.0001)
    # Make file look ancient
    import os
    import time

    old = time.time() - 3600 * 48
    os.utime(cache, (old, old))
    assert _odds_cache_fresh() is True


def test_odds_ttl_applies_when_fetch_once_off(tmp_path: Path, monkeypatch):
    cache = tmp_path / "ufc_odds_api.csv"
    cache.write_text("event_id,fighter_1,fighter_2\nx,a,b\n", encoding="utf-8")
    monkeypatch.setattr(config, "ODDS_CACHE_PATH", cache)
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", False)
    monkeypatch.setattr(config, "ODDS_CACHE_TTL_HOURS", 0.25)
    import os
    import time

    old = time.time() - 3600 * 2
    os.utime(cache, (old, old))
    assert _odds_cache_fresh() is False


def test_prop_fetch_once_marker_counts_as_fresh(tmp_path: Path, monkeypatch):
    import src.odds_providers.the_odds_api as toa

    cache = tmp_path / "the_odds_api_prop_odds.csv"
    marker = tmp_path / "the_odds_api_prop_odds.once"
    marker.write_text("fetched_at=1\nrows=0\n", encoding="utf-8")
    monkeypatch.setattr(toa, "PROP_CACHE_PATH", cache)
    monkeypatch.setattr(toa, "PROP_FETCH_ONCE_MARKER", marker)
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)
    assert toa._cache_fresh(cache) is True


def test_prop_cache_does_not_block_moneyline_fetch(tmp_path: Path, monkeypatch):
    """Prop once-marker must not forbid the first moneyline download."""
    from src.odds_providers.odds_api_client import odds_fetch_once_blocks_live

    cache = tmp_path / "cache"
    cache.mkdir()
    prop_csv = cache / "the_odds_api_prop_odds.csv"
    prop_csv.write_text("fighter,market,line\nA,ko,1.5\n", encoding="utf-8")
    (cache / "the_odds_api_prop_odds.once").write_text("rows=1\n", encoding="utf-8")
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "ODDS_CACHE_PATH", cache / "ufc_odds_api.csv")
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)

    assert odds_fetch_once_blocks_live(context="/sports/mma_mixed_martial_arts/odds") is False
    assert odds_fetch_once_blocks_live(context="/sports/mma_mixed_martial_arts/events") is True


def test_moneyline_cache_blocks_moneyline_not_props(tmp_path: Path, monkeypatch):
    from src.odds_providers.odds_api_client import odds_fetch_once_blocks_live

    cache = tmp_path / "cache"
    cache.mkdir()
    ml = cache / "ufc_odds_api.csv"
    ml.write_text("event_id,fighter_1,fighter_2\nx,a,b\n", encoding="utf-8")
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "ODDS_CACHE_PATH", cache / "ufc_odds_api.csv")
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)

    assert odds_fetch_once_blocks_live(context="/sports/mma_mixed_martial_arts/odds") is True
    assert odds_fetch_once_blocks_live(context="/sports/mma_mixed_martial_arts/events") is False


def test_prop_empty_cache_write_sets_marker(tmp_path: Path, monkeypatch):
    import pandas as pd
    import src.odds_providers.the_odds_api as toa
    from src.odds_providers.prop_odds_common import empty_prop_odds_df

    cache = tmp_path / "the_odds_api_prop_odds.csv"
    marker = tmp_path / "the_odds_api_prop_odds.once"
    monkeypatch.setattr(toa, "PROP_CACHE_PATH", cache)
    monkeypatch.setattr(toa, "PROP_FETCH_ONCE_MARKER", marker)
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)
    toa._write_prop_cache(empty_prop_odds_df())
    assert cache.is_file()
    assert marker.is_file()
    assert toa._cache_fresh(cache) is True
    cached = pd.read_csv(cache)
    assert cached.empty
