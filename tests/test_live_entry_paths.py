"""CLI start/record and scheduler/run_cli Live entry paths (temp state, mocked deps)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import config
from src import circuit_breaker as cb
from src.circuit_breaker import (
    initialize_live_session,
    live_safety_block_reason,
    record_bankroll_cli,
    start_live_session_cli,
)

DAY = date(2026, 6, 1)
NOW = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
KEY = f"live:{DAY.isoformat()}"


@pytest.fixture(autouse=True)
def _clock(monkeypatch, _isolated_safety_state):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_BANKROLL_MAX_AGE_MINUTES", 120)
    monkeypatch.setattr(cb, "_today", lambda: DAY)
    monkeypatch.setattr(cb, "_now", lambda: NOW)
    return _isolated_safety_state


def test_main_dispatches_start_live_session_and_record_bankroll(monkeypatch, _clock, capsys):
    import main

    monkeypatch.setattr(config, "CIRCUIT_BREAKER_STATE_PATH", _clock)
    assert main.main(["start-live-session", "--bankroll", "1000"]) == 0
    out = capsys.readouterr().out
    assert "started" in out.lower()
    state = json.loads(_clock.read_text(encoding="utf-8"))
    assert state[KEY]["bankroll_open"] == 1000.0
    assert state[KEY]["current_bankroll"] == 1000.0

    assert main.main(["start-live-session", "--bankroll", "500"]) == 1
    assert "Not changed" in capsys.readouterr().out
    assert json.loads(_clock.read_text(encoding="utf-8"))[KEY]["bankroll_open"] == 1000.0

    assert main.main(["record-bankroll", "--bankroll", "990"]) == 0
    assert "recorded" in capsys.readouterr().out.lower()
    state = json.loads(_clock.read_text(encoding="utf-8"))
    assert state[KEY]["bankroll_open"] == 1000.0
    assert state[KEY]["current_bankroll"] == 990.0
    assert state[KEY]["current_bankroll_source"] == "record-bankroll"


def test_cli_entry_source_dispatches_before_argparse():
    """cli_entry.main handles start-live-session / record-bankroll before build_parser."""
    from pathlib import Path

    text = Path("/workspace/src/cli_entry.py").read_text(encoding="utf-8")
    assert 'raw[0] == "start-live-session"' in text
    assert 'raw[0] == "record-bankroll"' in text
    assert "start_live_session_cli" in text
    assert "record_bankroll_cli" in text
    # Same handlers main.py uses — exercised above with temp state.
    assert start_live_session_cli(["--bankroll", "750"]) == 0
    assert record_bankroll_cli(["--bankroll", "740"]) == 0


def test_start_and_record_cli_helpers_refuse_bad_input(_clock, capsys):
    assert start_live_session_cli(["--bankroll", "0"]) == 1
    assert "Not changed" in capsys.readouterr().out
    assert not _clock.exists()
    assert start_live_session_cli(["--bankroll", "1000"]) == 0
    assert record_bankroll_cli(["--bankroll", "-5"]) == 1
    assert json.loads(_clock.read_text(encoding="utf-8"))[KEY]["current_bankroll"] == 1000.0


def test_watch_cycle_blocks_without_recorded_bankroll(monkeypatch, _clock):
    from src.scheduler import watch_cycle

    monkeypatch.setattr(config, "INITIAL_BANKROLL", 1000.0)
    preds = pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "fighter_1": "A",
                "fighter_2": "B",
                "predicted_winner": "A",
                "predicted_prob": 0.7,
                "prob_f1_win": 0.7,
                "prob_f2_win": 0.3,
                "edge_f1": 0.12,
                "edge_f2": -0.05,
                "best_edge": 0.12,
                "confidence_label": "high",
                "f1_odds": 1.55,
                "f2_odds": 2.5,
            }
        ]
    )
    alert, fp, notify, reason = watch_cycle(
        last_fingerprint=None,
        event_name="UFC Test",
        preds=preds,
        bankroll=1000.0,
    )
    assert not notify
    assert alert["available"] is False
    assert "missing" in reason or "start-live-session" in reason
    assert not _clock.exists()

    initialize_live_session(1000.0, now=NOW)
    # Loss beyond Live limit (0.012 → need ≤ ~988)
    from src.circuit_breaker import record_bankroll_observation

    record_bankroll_observation(980.0, now=NOW)
    alert2, _, notify2, reason2 = watch_cycle(
        last_fingerprint=None,
        event_name="UFC Test",
        preds=preds,
        bankroll=1000.0,
    )
    assert not notify2
    assert "daily loss" in reason2
    assert alert2["available"] is False


def test_watch_cycle_allows_with_fresh_recorded_bankroll(monkeypatch, _clock):
    from src.scheduler import watch_cycle

    initialize_live_session(1000.0, now=NOW)
    monkeypatch.setattr(
        "src.scheduler.generate_alerts",
        lambda preds, **kwargs: {
            "available": True,
            "event_name": "UFC Test",
            "singles": [{"pick": "A", "suggested_stake": 4.0}],
            "parlays": [],
            "singles_count": 1,
            "parlays_count": 0,
            "bankroll": 1000.0,
        },
    )
    monkeypatch.setattr("src.scheduler.should_send_alert", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("src.scheduler.alert_fingerprint", lambda *_a, **_k: "fp-1")
    preds = pd.DataFrame([{"fight_id": "f1"}])
    alert, fp, notify, reason = watch_cycle(
        last_fingerprint=None,
        event_name="UFC Test",
        preds=preds,
        bankroll=None,
    )
    assert notify
    assert reason == ""
    assert fp == "fp-1"
    assert alert["available"] is True


def test_run_cli_safety_gate_ignores_configured_bankroll(monkeypatch, _clock):
    import main

    monkeypatch.setattr(config, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "apply_profile_overrides", lambda: None)
    monkeypatch.setattr(main, "run_preflight", lambda **_k: 0)
    monkeypatch.setattr(main, "ensure_data_dirs", lambda: None)
    monkeypatch.setattr(main, "load_or_refresh_data", lambda **_k: pd.DataFrame())
    monkeypatch.setattr(main, "resolve_event_targets", lambda *_a, **_k: [(0, "UFC Test")])
    monkeypatch.setattr(main, "_model_exists", lambda: True)
    monkeypatch.setattr(main, "fetch_event_card", lambda *_a, **_k: pd.DataFrame([{"fighter_1": "A"}]))
    monkeypatch.setattr(
        main,
        "generate_predictions",
        lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "fight_id": "f1",
                    "fighter_1": "A",
                    "fighter_2": "B",
                    "event_name": "UFC Test",
                    "predicted_winner": "A",
                    "predicted_prob": 0.7,
                    "prob_f1_win": 0.7,
                    "prob_f2_win": 0.3,
                    "edge_f1": 0.12,
                    "edge_f2": -0.05,
                    "best_edge": 0.12,
                    "confidence_label": "high",
                    "f1_odds": 1.55,
                    "f2_odds": 2.5,
                }
            ]
        ),
    )
    # No Live session → safety gate must block even with a healthy INITIAL_BANKROLL.
    monkeypatch.setattr(
        main,
        "generate_alerts",
        lambda *_a, **_k: {
            "available": True,
            "bankroll": 1000.0,
            "singles": [{"suggested_stake": 5.0}],
            "parlays": [],
            "warnings": [],
        },
    )
    result = main.run_cli(
        "UFC Test",
        profile="live",
        skip_preflight=True,
        apply_safety_gates=True,
        use_odds=False,
        include_adjacent_week=False,
    )
    assert result["safety_blocked"] is True
    assert "missing" in result["safety_reason"] or "start-live-session" in result["safety_reason"]
    assert not _clock.exists()

    initialize_live_session(1000.0, now=NOW)
    from src.circuit_breaker import record_bankroll_observation

    record_bankroll_observation(980.0, now=NOW)
    cb.set_entry_block(None)
    result2 = main.run_cli(
        "UFC Test",
        profile="live",
        skip_preflight=True,
        apply_safety_gates=True,
        use_odds=False,
        include_adjacent_week=False,
    )
    assert result2["safety_blocked"] is True
    assert "daily loss" in result2["safety_reason"]


def test_watch_loop_exits_when_preflight_fails(monkeypatch, _clock):
    from src import scheduler

    monkeypatch.setattr("src.preflight.run_preflight", lambda **_k: 1)
    with pytest.raises(SystemExit) as exc:
        scheduler.watch_loop(max_iterations=1, bankroll=1000.0)
    assert exc.value.code == 1
    assert not _clock.exists()


def test_disabled_breaker_blocks_watch_cycle(monkeypatch, _clock):
    from src.scheduler import watch_cycle

    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)
    initialize_live_session(1000.0, now=NOW)
    alert, _, notify, reason = watch_cycle(
        last_fingerprint=None,
        event_name="UFC Test",
        preds=pd.DataFrame([{"fight_id": "f1"}]),
        bankroll=1000.0,
    )
    assert not notify
    assert "disabled" in reason
    assert alert["available"] is False
