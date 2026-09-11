"""Dashboard open uses overnight snapshot; never auto-runs Refresh Next Two."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.background_runner import (
    describe_startup_cache_status,
    load_startup_background_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def test_empty_snapshot_status_is_ready_not_refresh() -> None:
    assert describe_startup_cache_status(None) == "Ready — click Refresh Next Two"


def test_fresh_snapshot_status_names_overnight() -> None:
    snap = {
        "event_label": "UFC 320 + Fight Night",
        "_manifest": {"run_type": "full", "trigger": "scheduled"},
    }
    text = describe_startup_cache_status(snap, stale=False)
    assert "overnight snapshot" in text
    assert "full/scheduled" in text
    assert "UFC 320" in text
    assert "STALE" not in text


def test_stale_snapshot_status_warns_without_rescore() -> None:
    text = describe_startup_cache_status(
        {"event_label": "UFC Vegas"},
        stale=True,
    )
    assert text.startswith("STALE CACHE")
    assert "Refresh Next Two if the card changed" in text


def test_past_event_uses_stale_warning() -> None:
    text = describe_startup_cache_status(
        {"event_label": "Last night's card"},
        past_event=True,
    )
    assert "STALE CACHE" in text


def test_startup_load_skips_live_event_name_match(monkeypatch) -> None:
    """Name drift must not drop a usable overnight snapshot."""
    calls: list[float | None] = []

    def fake_load(*, max_age_hours=None):
        calls.append(max_age_hours)
        return {
            "event_label": "UFC Cached Card",
            "combined": pd.DataFrame({"fighter_1": ["A"], "fighter_2": ["B"]}),
            "cards": [],
            "_manifest": {"run_type": "full", "trigger": "scheduled"},
        }

    monkeypatch.setattr("src.background_runner.load_background_snapshot", fake_load)
    monkeypatch.setattr(
        "src.background_runner._startup_snapshot_is_past", lambda snap: False
    )

    data, meta = load_startup_background_snapshot()
    assert data is not None
    assert data["event_label"] == "UFC Cached Card"
    assert meta["stale"] is False
    assert "overnight snapshot" in meta["status"]
    assert calls == [24]


def test_startup_load_allows_72h_stale(monkeypatch) -> None:
    def fake_load(*, max_age_hours=None):
        if max_age_hours == 24:
            return None
        return {
            "event_label": "Older card",
            "combined": pd.DataFrame({"fighter_1": ["A"], "fighter_2": ["B"]}),
            "cards": [],
            "_manifest": {"run_type": "full", "trigger": "scheduled"},
        }

    monkeypatch.setattr("src.background_runner.load_background_snapshot", fake_load)
    monkeypatch.setattr(
        "src.background_runner._startup_snapshot_is_past", lambda snap: False
    )
    data, meta = load_startup_background_snapshot()
    assert data is not None
    assert meta["stale"] is True
    assert meta["status"].startswith("STALE CACHE")


def test_startup_load_empty_is_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.background_runner.load_background_snapshot",
        lambda **kwargs: None,
    )
    data, meta = load_startup_background_snapshot()
    assert data is None
    assert meta["status"] == "Ready — click Refresh Next Two"


def test_past_event_dates_mark_snapshot_stale(monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

    def fake_load(*, max_age_hours=None):
        return {
            "event_label": "UFC last night",
            "combined": pd.DataFrame(
                {"fighter_1": ["A"], "fighter_2": ["B"], "event_date": [yesterday]}
            ),
            "cards": [],
            "_manifest": {"run_type": "full", "trigger": "scheduled"},
        }

    monkeypatch.setattr("src.background_runner.load_background_snapshot", fake_load)
    data, meta = load_startup_background_snapshot()
    assert data is not None
    assert meta["past_event"] is True
    assert meta["status"].startswith("STALE CACHE")


def test_dashboard_startup_never_auto_refreshes() -> None:
    src = (ROOT / "src" / "ufc_dashboard.py").read_text(encoding="utf-8")
    assert "load_startup_background_snapshot" in src
    assert "self.after(800, self._auto_refresh_if_empty)" not in src
    assert "def _on_refresh(self)" in src  # still used by the Refresh Next Two button
    assert "not auto-running Refresh Next Two" in src
    leftover = src.replace("not auto-running Refresh Next Two", "")
    assert "auto-running Refresh Next Two" not in leftover
    assert "_snapshot_matches_live_events" not in src


def test_dashboard_pick_line_includes_odds() -> None:
    from src.bet_tiers import format_odds_line, format_top_pick_line

    bet = {
        "bet_tier": "blue",
        "display_label": "Jones ML",
        "book": "Odds API",
        "american_odds": "+110",
        "odds_display": "2.10",
        "prob": 0.55,
        "edge_pct": 7.5,
        "suggested_stake": 5.0,
    }
    line, color = format_top_pick_line(bet, 2)
    assert "#2" in line
    assert "Jones ML" in line
    assert "Odds API" in line
    assert "+110" in line
    assert "2.10" in line
    assert color.startswith("#")
    odds = format_odds_line(bet)
    assert odds == "Odds API  +110 (2.10)"
    dash = (ROOT / "src" / "ufc_dashboard.py").read_text(encoding="utf-8")
    assert "def _pick_line_text" in dash
    assert "format_top_pick_line" in dash
    assert "self._odds_line(bet)" in dash


def test_nightly_scripts_are_5am_full_and_logon_auto() -> None:
    bat = (ROOT / "scripts" / "setup_background.bat").read_text(
        encoding="utf-8", errors="replace"
    )
    ps1 = (ROOT / "scripts" / "register_background_tasks.ps1").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "/ST 05:00" in bat
    assert "/ST 04:00" not in bat
    assert "full scheduled" in bat
    assert "auto startup" in bat
    assert "/ST 05:00" in ps1
    assert "full scheduled" in ps1
    assert "auto startup" in ps1
    vbs = (ROOT / "scripts" / "run_background_hidden.vbs").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "run_background.bat" in vbs
