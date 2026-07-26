"""Odds API client diagnostics + 401 messaging (no real network)."""

from __future__ import annotations

import time
from pathlib import Path

from src.odds_providers.odds_api_client import (
    DEFAULT_REGIONS,
    FORCED_SPORT,
    LAST_REQUEST_META,
    MSG_KEY_REJECTED,
    MSG_MISSING_KEY,
    MSG_QUOTA_EXHAUSTED,
    clear_odds_api_session,
    key_last4,
    normalize_odds_api_key,
    odds_api_fail_closed_message,
    refresh_odds_api_runtime,
    resolve_odds_api_key_source,
)


def test_normalize_strips_quotes_and_ws():
    assert normalize_odds_api_key('  "abc123"  ') == "abc123"
    assert normalize_odds_api_key("'xyz'") == "xyz"
    assert key_last4("abcdefgh") == "efgh"


def test_refresh_forces_sport_and_default_regions(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("THE_ODDS_API_KEY=testkey1234567890\n", encoding="utf-8")
    monkeypatch.setenv("THE_ODDS_API_KEY", "testkey1234567890")
    monkeypatch.delenv("ODDS_API_REGIONS", raising=False)
    monkeypatch.setattr(
        "src.odds_providers.odds_api_client.resolve_odds_api_key_source",
        lambda root=None: ("testkey1234567890", str(env_file)),
    )
    meta = refresh_odds_api_runtime(root=tmp_path)
    assert meta["sport"] == FORCED_SPORT
    assert meta["regions"] == DEFAULT_REGIONS
    assert meta["markets"] == "h2h"
    assert meta["odds_format"] == "decimal"
    assert meta["key_loaded"] is True
    assert meta["key_length"] == len("testkey1234567890")
    assert meta["key_last4"] == "7890"
    assert "testkey1234567890" not in str(meta.values())


def test_prefer_newest_env_between_project_and_dist(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.project_paths.resolve_root",
        lambda entry_file=None: tmp_path,
    )
    older = tmp_path / ".env"
    newer_dir = tmp_path / "dist"
    newer_dir.mkdir()
    newer = newer_dir / ".env"
    older.write_text("THE_ODDS_API_KEY=oldkeyaaaaaaaaaaaaaaaaaaaaaa\n", encoding="utf-8")
    newer.write_text("THE_ODDS_API_KEY=newkeybbbbbbbbbbbbbbbbbbbb\n", encoding="utf-8")
    # Make dist/.env strictly newer
    now = time.time()
    import os

    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    key, source = resolve_odds_api_key_source(tmp_path)
    assert key.endswith("bbbb")
    assert source.endswith(str(Path("dist") / ".env")) or "dist" in source.replace("\\", "/")


def test_401_quota_message_not_unauthorized():
    LAST_REQUEST_META.update(
        {
            "key_loaded": True,
            "key_length": 32,
            "key_last4": "d616",
            "key_source": r"C:\UFC-Predictor\.env",
            "sport": FORCED_SPORT,
            "regions": DEFAULT_REGIONS,
        }
    )
    msg = odds_api_fail_closed_message(
        status_code=401,
        error_code="OUT_OF_USAGE_CREDITS",
    )
    assert "NO BET" in msg
    assert MSG_QUOTA_EXHAUSTED in msg
    assert "unauthorized" not in msg.lower()
    assert "last4=d616" in msg


def test_401_auth_message_key_rejected():
    LAST_REQUEST_META.update(
        {
            "key_loaded": True,
            "key_length": 32,
            "key_last4": "abcd",
            "sport": FORCED_SPORT,
            "regions": DEFAULT_REGIONS,
        }
    )
    msg = odds_api_fail_closed_message(status_code=401, error_code="")
    assert MSG_KEY_REJECTED in msg
    assert "unauthorized" not in msg.lower()


def test_missing_key_message():
    LAST_REQUEST_META.update(
        {
            "key_loaded": False,
            "key_length": 0,
            "key_last4": "",
            "sport": FORCED_SPORT,
            "regions": DEFAULT_REGIONS,
        }
    )
    msg = odds_api_fail_closed_message(detail="THE_ODDS_API_KEY missing")
    assert MSG_MISSING_KEY in msg


def test_clear_session_resets_module_session():
    clear_odds_api_session()
    from src.odds_providers import odds_api_client as mod

    assert mod._SESSION is None
