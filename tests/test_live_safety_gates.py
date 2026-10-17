"""Live safety: recorded bankroll, disabled breaker, narrative revalidation, book provenance."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
import requests

import config
from src import circuit_breaker as cb
from src.bet_tiers import TIER_BLUE
from src.circuit_breaker import (
    LiveSessionExists,
    NoLiveSession,
    SafetyStateUnreadable,
    check_alerts_allowed,
    evaluate_live_daily_loss,
    initialize_live_session,
    live_observed_bankroll,
    live_safety_block_reason,
    record_bankroll_cli,
    record_bankroll_observation,
    start_live_session_cli,
    update_session_bankroll_anchor,
)
from src.grok_analysis import apply_grok_kelly_adjustments, attach_narrative_tilts_to_alerts
from src.quote_integrity import live_quote_block_reason
from src.strategy import (
    allocate_card_budget_pct,
    budget_aware_alerts,
    collect_dashboard_risk_warnings,
)
from tests.test_audit_fail_closed import _fresh_quote, _strong_ticket

DAY = date(2026, 6, 1)
NOW = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
KEY = f"live:{DAY.isoformat()}"
TILT_UP = {
    "ok": True,
    "picks": [{"id": "ab", "kelly_adjustment": 1.20, "conviction": "high", "narrative_edge": "x"}],
}
NO_OP = {"ok": True, "picks": []}
FAILED = {"ok": False, "picks": []}
NARRATIVES = [None, FAILED, NO_OP, TILT_UP]


@pytest.fixture(autouse=True)
def _live_clock_and_no_network(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.25)
    monkeypatch.setattr(config, "ODDS_CACHE_TTL_MINUTES", 20)
    monkeypatch.setattr(config, "LIVE_BANKROLL_MAX_AGE_MINUTES", 120)
    monkeypatch.setattr(cb, "_today", lambda: DAY)
    monkeypatch.setattr(cb, "_now", lambda: NOW)

    def _no_network(*_args, **_kwargs):
        raise AssertionError("network access in a safety test")

    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(requests, "post", _no_network)
    monkeypatch.setattr(requests.Session, "request", _no_network)


@pytest.fixture
def state_path(_isolated_safety_state):
    return _isolated_safety_state


@pytest.fixture
def live_session_today(state_path):
    initialize_live_session(1000.0, now=NOW)
    return state_path


def _write_state(path, state) -> None:
    path.write_text(json.dumps(state), encoding="utf-8")


def _session(anchor=1000.0, **extra) -> dict:
    stamp = NOW.isoformat()
    row = {
        "session_date": DAY.isoformat(),
        "bankroll_open": anchor,
        "profile": "live",
        "initialized_at": stamp,
        "initialized_by": "explicit",
        "current_bankroll": anchor,
        "current_bankroll_source": "start-live-session",
        "current_bankroll_observed_at": stamp,
    }
    row.update(extra)
    return row


def _snapshot(path):
    if not path.exists():
        return None
    return path.read_bytes(), os.stat(path).st_mtime_ns


def _ticket(**extra) -> dict:
    fields = _fresh_quote(
        quote_timestamp=(NOW - timedelta(minutes=5)).isoformat(),
        commence_time=(NOW + timedelta(days=2)).isoformat(),
    )
    fields.update(extra)
    for key, value in (("kelly_stake_usd", 4.0), ("suggested_stake", 4.0)):
        fields.setdefault(key, value)
    return _strong_ticket(**fields)


def _funded(ticket) -> bool:
    return float(ticket.get("suggested_stake") or 0.0) > 0.0


def _assert_zero(ticket) -> None:
    for field in ("suggested_stake", "stake_usd", "stake_pct", "kelly_stake_usd", "max_safe_bet_usd"):
        assert float(ticket.get(field) or 0.0) == 0.0, field
    assert ticket["bet_action"] == "NO BET"
    assert ticket["quote_actionable"] is False


def _allocate(tickets, *, as_of=NOW):
    return allocate_card_budget_pct(tickets, 100.0, profile="live", inplace=False, as_of=as_of)


# --- Item 1: recorded current bankroll -------------------------------------------


def test_valid_session_and_recorded_bankroll_allow_valid_ticket(live_session_today):
    before = _snapshot(live_session_today)
    decision = evaluate_live_daily_loss(now=NOW)
    assert decision.allowed
    assert decision.current is not None
    assert decision.current.value == 1000.0
    assert decision.current.source == "start-live-session"
    out = _allocate([_ticket()])
    assert 0.0 < float(out[0]["suggested_stake"]) <= 4.0 + 1e-9
    assert _snapshot(live_session_today) == before


def test_caller_or_config_bankroll_cannot_authorize_or_hide_loss(state_path, monkeypatch):
    initialize_live_session(1000.0, now=NOW)
    record_bankroll_observation(980.0, now=NOW)
    monkeypatch.setattr(config, "INITIAL_BANKROLL", 1000.0)
    before = _snapshot(state_path)
    # A "healthy" caller bankroll must not override the recorded 980 loss.
    decision = evaluate_live_daily_loss(now=NOW)
    assert not decision.allowed
    assert decision.computed_trip
    _assert_zero(_allocate([_ticket(bankroll=1000.0)])[0])
    _assert_zero(
        apply_grok_kelly_adjustments([_ticket(bankroll=1000.0)], TILT_UP, now=NOW)[0]
    )
    # Read-only paths leave the file alone; the alert gate may then persist the trip.
    assert _snapshot(state_path) == before
    assert not check_alerts_allowed(1000.0)[0]
    stored = json.loads(state_path.read_text(encoding="utf-8"))[KEY]
    assert stored["circuit_tripped"] is True
    assert stored["bankroll_open"] == 1000.0
    assert stored["current_bankroll"] == 980.0


def test_computed_loss_over_limit_blocks_before_any_trip_flag(state_path):
    initialize_live_session(1000.0, now=NOW)
    record_bankroll_observation(985.0, now=NOW)
    before = _snapshot(state_path)
    decision = evaluate_live_daily_loss(now=NOW)
    assert not decision.allowed
    assert decision.computed_trip
    assert "daily loss circuit breaker" in decision.reason
    out = _allocate([_ticket()])
    _assert_zero(out[0])
    assert "daily loss" in out[0]["safety_block_reason"]
    stored = json.loads(state_path.read_text(encoding="utf-8"))[KEY]
    assert "circuit_tripped" not in stored
    assert stored["current_bankroll"] == 985.0
    assert _snapshot(state_path) == before


def test_missing_file_blocks_everywhere_and_is_never_recreated(state_path):
    assert not state_path.exists()
    assert "live safety state missing" in live_safety_block_reason(now=NOW)
    assert "start-live-session" in live_safety_block_reason(now=NOW)
    _assert_zero(_allocate([_ticket()])[0])
    for narrative in NARRATIVES:
        _assert_zero(apply_grok_kelly_adjustments([_ticket()], narrative, now=NOW)[0])
    cached = budget_aware_alerts({"singles": [_ticket()], "bankroll": 1000.0}, None, "DraftKings")
    _assert_zero(cached["singles"][0])
    allowed, reason = check_alerts_allowed(1000.0)
    assert not allowed and "missing" in reason
    warnings = collect_dashboard_risk_warnings({"bankroll": 1000.0}, None)
    assert any(sev == "critical" and "Live stakes are $0" in msg for sev, msg in warnings)
    assert not state_path.exists()


def test_deleted_file_blocks_and_is_not_recreated(state_path):
    initialize_live_session(1000.0, now=NOW)
    assert _funded(_allocate([_ticket()])[0])
    state_path.unlink()
    _assert_zero(_allocate([_ticket()])[0])
    _assert_zero(apply_grok_kelly_adjustments([_ticket()], None, now=NOW)[0])
    assert not check_alerts_allowed(1000.0)[0]
    assert not state_path.exists()


@pytest.mark.parametrize(
    "state, needle",
    [
        ({KEY: _session(anchor=None)}, "anchor missing or invalid"),
        ({KEY: _session(anchor="abc")}, "anchor missing or invalid"),
        ({KEY: _session(anchor=0)}, "anchor missing or invalid"),
        ({KEY: _session(anchor=-50.0)}, "anchor missing or invalid"),
        ({KEY: _session(anchor=True)}, "anchor missing or invalid"),
        ({KEY: {"session_date": DAY.isoformat(), "initialized_at": NOW.isoformat()}}, "anchor"),
        ({"live:2026-05-31": _session(session_date="2026-05-31")}, "last Live session 2026-05-31"),
        ({KEY: _session(session_date="2026-05-31")}, "does not match"),
        ({KEY: "not-a-record"}, "not an object"),
        ({DAY.isoformat(): {"session_date": DAY.isoformat(), "bankroll_open": 1000.0}}, "no Live session"),
        ({}, "no Live session"),
    ],
)
def test_bad_or_stale_anchor_blocks_without_reanchoring(state_path, state, needle):
    _write_state(state_path, state)
    before = _snapshot(state_path)
    reason = live_safety_block_reason(now=NOW)
    assert needle in reason
    _assert_zero(_allocate([_ticket()])[0])
    _assert_zero(apply_grok_kelly_adjustments([_ticket()], TILT_UP, now=NOW)[0])
    assert not check_alerts_allowed(1000.0)[0]
    assert _snapshot(state_path) == before


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_anchor_blocks(state_path, raw):
    state_path.write_text(
        '{"%s": {"session_date": "%s", "bankroll_open": %s, "initialized_at": "%s"}}'
        % (KEY, DAY.isoformat(), raw, NOW.isoformat()),
        encoding="utf-8",
    )
    before = _snapshot(state_path)
    assert "anchor missing or invalid" in live_safety_block_reason(now=NOW)
    _assert_zero(_allocate([_ticket()])[0])
    assert _snapshot(state_path) == before


@pytest.mark.parametrize("text", ["{not-json", "[1, 2]", "", "\x00\x01"])
def test_unreadable_file_blocks_and_is_not_replaced(state_path, text):
    state_path.write_text(text, encoding="utf-8")
    before = _snapshot(state_path)
    reason = live_safety_block_reason(now=NOW)
    assert "unreadable" in reason or "not an object" in reason
    _assert_zero(_allocate([_ticket()])[0])
    with pytest.raises(SafetyStateUnreadable):
        initialize_live_session(1000.0, now=NOW)
    assert start_live_session_cli(["--bankroll", "1000"]) == 1
    assert _snapshot(state_path) == before


@pytest.mark.parametrize(
    "patch",
    [
        {"current_bankroll": None},
        {"current_bankroll": 0},
        {"current_bankroll": -1},
        {"current_bankroll": "abc"},
        {"current_bankroll": True},
        {"current_bankroll_source": ""},
        {"current_bankroll_source": "config_default"},
        {"current_bankroll_source": "INITIAL_BANKROLL"},
        {"current_bankroll_source": "caller"},
        {"current_bankroll_source": "budget.json"},
        {"current_bankroll_observed_at": None},
        {"current_bankroll_observed_at": ""},
        {"current_bankroll_observed_at": "not-a-time"},
        {"current_bankroll_observed_at": "2026-06-01T18:00:00"},  # naive
        {"current_bankroll_observed_at": (NOW - timedelta(hours=3)).isoformat()},
        {"current_bankroll_observed_at": (NOW + timedelta(hours=1)).isoformat()},
        {"initialized_at": None},
        {"initialized_at": "2026-06-01T19:00:00+00:00"},  # after observation
    ],
)
def test_missing_invalid_stale_or_untrusted_current_bankroll_blocks(state_path, patch):
    row = _session()
    row.update(patch)
    if "current_bankroll" in patch and patch["current_bankroll"] is None:
        row.pop("current_bankroll", None)
    if patch.get("current_bankroll_source") == "":
        row["current_bankroll_source"] = ""
    _write_state(state_path, {KEY: row})
    before = _snapshot(state_path)
    reason = live_safety_block_reason(now=NOW)
    assert reason
    assert "daily loss circuit breaker (" not in reason
    assert live_observed_bankroll(now=NOW) is None
    _assert_zero(_allocate([_ticket(bankroll=1000.0)])[0])
    assert not check_alerts_allowed(1000.0)[0]
    assert _snapshot(state_path) == before


def test_fresh_record_bankroll_allows_then_stale_blocks(state_path, monkeypatch):
    initialize_live_session(1000.0, now=NOW - timedelta(minutes=30))
    assert _funded(_allocate([_ticket()])[0])
    later = NOW + timedelta(hours=3)
    monkeypatch.setattr(cb, "_now", lambda: later)
    assert "recorded" in live_safety_block_reason()
    _assert_zero(_allocate([_ticket()])[0])
    record_bankroll_observation(990.0, now=later)
    assert live_safety_block_reason() == ""
    assert _funded(_allocate([_ticket()])[0])


def test_stored_trip_blocks_even_after_bankroll_recovers(state_path):
    _write_state(state_path, {KEY: _session(circuit_tripped=True, loss_pct=0.02)})
    before = _snapshot(state_path)
    assert "stored trip" in live_safety_block_reason(now=NOW)
    _assert_zero(_allocate([_ticket()])[0])
    assert _snapshot(state_path) == before


def test_legacy_plain_date_trip_still_blocks_live(state_path):
    _write_state(
        state_path,
        {
            DAY.isoformat(): {"session_date": DAY.isoformat(), "circuit_tripped": True, "loss_pct": 0.05},
            KEY: _session(),
        },
    )
    assert "stored trip" in live_safety_block_reason(now=NOW)
    _assert_zero(_allocate([_ticket()])[0])


def test_explicit_initialization_never_overwrites_anchor_or_trip(state_path):
    with pytest.raises(ValueError):
        initialize_live_session(float("nan"))
    with pytest.raises(ValueError):
        initialize_live_session(0)
    assert not state_path.exists()

    session = initialize_live_session(1000.0, now=NOW)
    assert session["bankroll_open"] == 1000.0
    assert session["current_bankroll"] == 1000.0
    assert session["current_bankroll_source"] == "start-live-session"
    assert session["initialized_by"] == "explicit"
    first = _snapshot(state_path)
    with pytest.raises(LiveSessionExists):
        initialize_live_session(500.0, now=NOW)
    update_session_bankroll_anchor(500.0)
    assert start_live_session_cli(["--bankroll", "500"]) == 1
    assert _snapshot(state_path) == first

    _write_state(state_path, {KEY: _session(circuit_tripped=True, loss_pct=0.03)})
    tripped = _snapshot(state_path)
    with pytest.raises(LiveSessionExists):
        initialize_live_session(2000.0, now=NOW)
    assert _snapshot(state_path) == tripped
    assert "stored trip" in live_safety_block_reason(now=NOW)


def test_cli_starts_session_and_records_opening_as_current(state_path, capsys):
    _write_state(state_path, {"live:2026-05-31": _session(session_date="2026-05-31", bankroll_open=800.0)})
    assert start_live_session_cli(["--bankroll", "1000"]) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state[KEY]["bankroll_open"] == 1000.0
    assert state[KEY]["current_bankroll"] == 1000.0
    assert state[KEY]["current_bankroll_source"] == "start-live-session"
    assert state["live:2026-05-31"]["bankroll_open"] == 800.0
    assert "started" in capsys.readouterr().out
    assert live_safety_block_reason(now=NOW) == ""


def test_record_bankroll_cli_updates_observation_not_anchor(state_path, capsys):
    initialize_live_session(1000.0, now=NOW - timedelta(minutes=10))
    assert record_bankroll_cli(["--bankroll", "990"]) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))[KEY]
    assert state["bankroll_open"] == 1000.0
    assert state["current_bankroll"] == 990.0
    assert state["current_bankroll_source"] == "record-bankroll"
    assert "recorded" in capsys.readouterr().out.lower()
    assert live_safety_block_reason(now=NOW) == ""

    with pytest.raises(NoLiveSession):
        record_bankroll_observation(900.0, today=date(2026, 6, 2), now=NOW)
    assert record_bankroll_cli(["--bankroll", "0"]) == 1
    assert state_path.read_text(encoding="utf-8")  # still present
    assert json.loads(state_path.read_text(encoding="utf-8"))[KEY]["bankroll_open"] == 1000.0


def test_alert_gate_records_computed_trip_only_on_existing_session(state_path):
    initialize_live_session(1000.0, now=NOW)
    record_bankroll_observation(980.0, now=NOW)
    allowed, reason = check_alerts_allowed(None)
    assert not allowed and "daily loss" in reason
    stored = json.loads(state_path.read_text(encoding="utf-8"))[KEY]
    assert stored["circuit_tripped"] is True
    assert stored["bankroll_open"] == 1000.0
    assert stored["current_bankroll"] == 980.0
    cb.set_entry_block(None)
    assert "stored trip" in live_safety_block_reason(now=NOW)


def test_preflight_fails_without_live_session_or_current_bankroll(state_path, monkeypatch):
    from src import preflight

    monkeypatch.setattr(config, "ODDS_API_KEY", "")
    monkeypatch.setattr(config, "LOG_DIR", state_path.parent / "logs")
    monkeypatch.setattr(config, "CACHE_DIR", state_path.parent / "cache")
    monkeypatch.setattr(config, "apply_profile_overrides", lambda: None)
    monkeypatch.setattr(preflight, "model_needs_retrain", lambda: (False, ""))
    monkeypatch.setattr(preflight, "stale_model_warning", lambda: "")
    monkeypatch.setattr(preflight, "read_heartbeat", lambda: None)
    monkeypatch.setattr(preflight, "heartbeat_age_minutes", lambda: None)
    lines: list[str] = []
    assert preflight.run_preflight(printer=lines.append) == 1
    assert any("[FAIL] Live stakes blocked" in line for line in lines)
    assert not state_path.exists()

    initialize_live_session(1000.0, now=NOW)
    lines.clear()
    preflight.run_preflight(printer=lines.append)
    assert any("current bankroll recorded" in line for line in lines)

    # Drop the observation; preflight must fail again.
    row = json.loads(state_path.read_text(encoding="utf-8"))[KEY]
    row.pop("current_bankroll")
    _write_state(state_path, {KEY: row})
    lines.clear()
    assert preflight.run_preflight(printer=lines.append) == 1
    assert any("current bankroll" in line for line in lines)


def test_authorization_paths_do_not_touch_safety_file(live_session_today):
    before = _snapshot(live_session_today)
    evaluate_live_daily_loss(now=NOW)
    live_safety_block_reason(now=NOW)
    live_observed_bankroll(now=NOW)
    cb.get_circuit_status()
    first = _allocate([_ticket()])
    for narrative in NARRATIVES:
        apply_grok_kelly_adjustments(first, narrative, now=NOW)
    attach_narrative_tilts_to_alerts({"singles": [_ticket()], "parlays": []}, TILT_UP, now=NOW)
    budget_aware_alerts({"singles": [_ticket()]}, None, "DraftKings")
    collect_dashboard_risk_warnings(None, None)
    assert _snapshot(live_session_today) == before


# --- Item 2: disabled breaker ----------------------------------------------------


def test_live_disabled_breaker_returns_zero(state_path, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)
    initialize_live_session(1000.0, now=NOW)
    reason = live_safety_block_reason(now=NOW)
    assert "disabled" in reason
    _assert_zero(_allocate([_ticket()])[0])
    _assert_zero(apply_grok_kelly_adjustments([_ticket()], TILT_UP, now=NOW)[0])
    allowed, block = check_alerts_allowed(1000.0)
    assert not allowed and "disabled" in block
    warnings = collect_dashboard_risk_warnings(None, None)
    assert any("disabled" in msg for _, msg in warnings)
    # Authorization still must not rewrite the session.
    stored = json.loads(state_path.read_text(encoding="utf-8"))[KEY]
    assert stored["bankroll_open"] == 1000.0
    assert "circuit_tripped" not in stored


def test_paper_disabled_breaker_still_allows(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)
    tripped, reason, _ = cb.daily_loss_circuit_tripped(1000.0)
    assert not tripped and reason == ""
    allowed, _ = check_alerts_allowed(1000.0)
    assert allowed
    paper = allocate_card_budget_pct([_strong_ticket(kelly_stake_usd=8.0)], 100.0, profile="paper")
    assert float(paper[0]["suggested_stake"]) > 0.0


# --- Item 3: every narrative return path is revalidated --------------------------


@pytest.mark.parametrize("narrative", NARRATIVES, ids=["none", "failed", "noop", "tilt"])
def test_quote_expiring_during_narrative_returns_zero(live_session_today, narrative):
    first = _allocate([_ticket()])
    assert _funded(first[0])
    later = NOW + timedelta(minutes=30)
    out = apply_grok_kelly_adjustments(first, narrative, now=later)
    _assert_zero(out[0])
    assert out[0]["quote_block_reason"] == "expired_quote"


@pytest.mark.parametrize("narrative", NARRATIVES, ids=["none", "failed", "noop", "tilt"])
def test_event_start_during_narrative_returns_zero(live_session_today, narrative):
    ticket = _ticket(commence_time=(NOW + timedelta(minutes=10)).isoformat())
    first = _allocate([ticket])
    assert _funded(first[0])
    out = apply_grok_kelly_adjustments(first, narrative, now=NOW + timedelta(minutes=11))
    _assert_zero(out[0])
    assert out[0]["quote_block_reason"] == "already_started"


@pytest.mark.parametrize("narrative", NARRATIVES, ids=["none", "failed", "noop", "tilt"])
def test_valid_blue_ticket_stays_funded_through_narrative(live_session_today, narrative):
    first = _allocate([_ticket(bet_tier=TIER_BLUE)])
    out = apply_grok_kelly_adjustments(first, narrative, now=NOW)
    assert 0.0 < float(out[0]["suggested_stake"]) <= 4.0 + 1e-9
    assert float(out[0]["kelly_stake_usd"]) == 4.0
    assert out[0]["bet_tier"] == TIER_BLUE


@pytest.mark.parametrize("narrative", NARRATIVES, ids=["none", "failed", "noop", "tilt"])
def test_dict_payload_lists_are_all_revalidated(live_session_today, narrative):
    later = NOW + timedelta(minutes=30)
    payload = {
        "singles": [_ticket()],
        "parlays": [_ticket(fight_id="p1", is_parlay=True)],
        "prop_singles": [_ticket(fight_id="pr1", market_type="prop")],
        "model_parlays": [_ticket(fight_id="mp1", is_parlay=True)],
        "bankroll": 1000.0,
    }
    out = attach_narrative_tilts_to_alerts(payload, narrative, now=later)
    for key in ("singles", "parlays", "prop_singles", "model_parlays"):
        assert out[key], key
        for ticket in out[key]:
            _assert_zero(ticket)
    direct = apply_grok_kelly_adjustments(payload, narrative, now=later)
    for key in ("singles", "parlays", "prop_singles", "model_parlays"):
        for ticket in direct[key]:
            _assert_zero(ticket)
    for ticket in direct["items"]:
        _assert_zero(ticket)


def test_dict_payload_keeps_valid_singles_funded(live_session_today):
    payload = {"singles": [_ticket()], "parlays": []}
    out = attach_narrative_tilts_to_alerts(payload, TILT_UP, now=NOW)
    assert 0.0 < float(out["singles"][0]["suggested_stake"]) <= 4.0 + 1e-9


@pytest.mark.parametrize("narrative", NARRATIVES, ids=["none", "failed", "noop", "tilt"])
@pytest.mark.parametrize("ceiling", [None, "", "abc", float("nan"), float("inf"), -1.0, 0.0, "missing"])
def test_invalid_kelly_ceiling_cannot_survive_narrative(live_session_today, narrative, ceiling):
    cached = _ticket(suggested_stake=12.0, stake_usd=12.0, card_pool_usd=100.0, stake_pct=12.0)
    if ceiling == "missing":
        cached.pop("kelly_stake_usd")
    else:
        cached["kelly_stake_usd"] = ceiling
    out = apply_grok_kelly_adjustments([cached], narrative, now=NOW)
    _assert_zero(out[0])


def test_narrative_never_raises_stored_ceiling(live_session_today):
    first = _allocate([_ticket(max_safe_bet_usd=4.0, kelly_pct=0.4)])
    out = apply_grok_kelly_adjustments(first, TILT_UP, now=NOW)
    assert float(out[0]["kelly_stake_usd"]) == 4.0
    assert float(out[0]["max_safe_bet_usd"]) <= 4.0 + 1e-9
    assert float(out[0]["kelly_pct"]) <= 0.4 + 1e-9
    assert float(out[0]["suggested_stake"]) <= 4.0 + 1e-9


def test_cached_oversized_stake_is_capped_on_return(live_session_today):
    cached = _ticket(kelly_stake_usd=3.0, suggested_stake=30.0, stake_usd=30.0, card_pool_usd=100.0)
    out = apply_grok_kelly_adjustments([cached], None, now=NOW)
    assert float(out[0]["suggested_stake"]) == 3.0
    assert float(out[0]["stake_usd"]) == 3.0
    assert float(out[0]["stake_pct"]) == 3.0


def test_second_allocation_cannot_revive_rejected_stakes(live_session_today):
    expired = _ticket(quote_timestamp=(NOW - timedelta(hours=2)).isoformat(), kelly_stake_usd=9.0)
    no_ceiling = _ticket(fight_id="cd", kelly_stake_usd=None, suggested_stake=9.0)
    valid = _ticket(fight_id="ef")
    first = _allocate([expired, no_ceiling, valid])
    tilted = apply_grok_kelly_adjustments(first, TILT_UP, now=NOW)
    second = _allocate(tilted)
    _assert_zero(second[0])
    _assert_zero(second[1])
    assert 0.0 < float(second[2]["suggested_stake"]) <= 4.0 + 1e-9


def test_safety_block_during_narrative_returns_zero(state_path):
    initialize_live_session(1000.0, now=NOW)
    first = _allocate([_ticket()])
    assert _funded(first[0])
    record_bankroll_observation(980.0, now=NOW)
    out = apply_grok_kelly_adjustments(first, NO_OP, now=NOW)
    _assert_zero(out[0])
    assert "daily loss" in out[0]["safety_block_reason"]


def test_paper_override_is_dropped_in_live_narrative(live_session_today):
    override = _ticket(uncertainty_action="tighten", uncertainty_reason="paper_wide_override")
    out = apply_grok_kelly_adjustments([override], None, now=NOW)
    _assert_zero(out[0])
    assert out[0]["quote_block_reason"] == "paper_wide_override"


def test_paper_narrative_and_overrides_are_unchanged(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    bets = [_strong_ticket(kelly_stake_usd=4.0, suggested_stake=4.0)]
    assert apply_grok_kelly_adjustments(bets, None) is bets
    override = _strong_ticket(
        uncertainty_action="tighten",
        uncertainty_reason="paper_wide_override",
        kelly_stake_usd=8.0,
        suggested_stake=5.0,
    )
    out = apply_grok_kelly_adjustments([override], NO_OP)
    assert float(out[0]["suggested_stake"]) == 5.0
    paper = allocate_card_budget_pct([_strong_ticket(kelly_stake_usd=8.0)], 100.0, profile="paper")
    assert float(paper[0]["suggested_stake"]) > 0.0
    assert not cb.daily_loss_circuit_tripped(1000.0)[0]


def test_generate_alerts_uses_recorded_bankroll_not_config(live_session_today, monkeypatch):
    from src.alerts import generate_alerts
    from tests.test_alerts import _preds_with_edge

    monkeypatch.setattr(config, "INITIAL_BANKROLL", 50.0)
    monkeypatch.setattr("src.skip_scorecard.rollup_skip_reasons", lambda **_k: None)
    out = generate_alerts(
        _preds_with_edge(),
        min_edge=0.05,
        risk_metrics={"available": False},
        bankroll=50.0,
    )
    assert out["bankroll"] == 1000.0
    assert out["bankroll_source"] == "start-live-session"
    assert out["bankroll_observed_at"]


def test_generate_alerts_missing_recorded_bankroll_returns_zero(state_path, monkeypatch):
    from src.alerts import generate_alerts
    from tests.test_alerts import _preds_with_edge

    monkeypatch.setattr(config, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr("src.skip_scorecard.rollup_skip_reasons", lambda **_k: None)
    out = generate_alerts(
        _preds_with_edge(),
        min_edge=0.05,
        risk_metrics={"available": False},
        bankroll=1000.0,
    )
    assert out["bankroll"] is None
    assert out["bankroll_source"] == "missing"
    for key in ("singles", "parlays", "prop_singles", "model_parlays"):
        for ticket in out.get(key) or []:
            assert float(ticket.get("suggested_stake") or 0.0) == 0.0
    assert not state_path.exists()


# --- Item 4: explicit book provenance ---------------------------------------------


@pytest.mark.parametrize(
    "kind, reason",
    [
        ("__missing__", "price_kind_not_book"),
        (None, "price_kind_not_book"),
        ("", "price_kind_not_book"),
        ("   ", "price_kind_not_book"),
        ("unknown", "price_kind_not_book"),
        ("synthetic", "price_kind_not_book"),
        ("books", "price_kind_not_book"),
        ("consensus", "consensus_not_executable"),
        ("mean", "consensus_not_executable"),
        ("median", "consensus_not_executable"),
        ("advisory", "consensus_not_executable"),
        ("Consensus ", "consensus_not_executable"),
    ],
)
def test_live_requires_explicit_book_price_kind(live_session_today, kind, reason):
    ticket = _ticket()
    if kind == "__missing__":
        ticket.pop("price_kind")
    else:
        ticket["price_kind"] = kind
    assert live_quote_block_reason(ticket, now=NOW) == reason
    out = _allocate([ticket])
    _assert_zero(out[0])
    assert out[0]["quote_block_reason"] == reason


@pytest.mark.parametrize("kind", ["book", "BOOK", " Book "])
def test_genuine_book_quote_passes(live_session_today, kind):
    ticket = _ticket(price_kind=kind)
    assert live_quote_block_reason(ticket, now=NOW) == ""
    out = _allocate([ticket])
    assert 0.0 < float(out[0]["suggested_stake"]) <= 4.0 + 1e-9


def test_book_label_does_not_bypass_single_book_rules(live_session_today):
    multi = _ticket(bookmaker_count=3)
    no_book = _ticket(bookmaker="")
    aggregate = _ticket(bookmaker="consensus")
    assert live_quote_block_reason(multi, now=NOW) == "consensus_not_executable"
    assert live_quote_block_reason(no_book, now=NOW) == "missing_bookmaker"
    assert live_quote_block_reason(aggregate, now=NOW) == "consensus_not_executable"


def test_single_book_scrapers_label_book_and_fallback_does_not_invent_it():
    from src.odds_providers.betnow_scraper import _parse_moneyline_rows
    from src.odds_providers.odds_fallback import _normalize_odds_frame

    rows = _parse_moneyline_rows(
        [{"fighters": [{"name": "Jon Jones", "moneyline": "-200"}, {"name": "Stipe Miocic", "moneyline": "+170"}]}]
    )
    assert rows[0]["price_kind"] == "book"
    assert rows[0]["bookmaker_count"] == 1
    unlabeled = _normalize_odds_frame(
        pd.DataFrame([{"fighter_1": "A", "fighter_2": "B", "f1_odds": 1.8, "f2_odds": 2.1}]),
        bookmaker="SomeBook",
    )
    assert "price_kind" not in unlabeled.columns


# --- Item 5: quote names never fall back to card names ----------------------------


def test_quote_fighter_names_never_copied_from_card():
    from src.predictor import merge_predictions_with_odds

    card = pd.DataFrame(
        [
            {
                "fighter_1": "Card One",
                "fighter_2": "Card Two",
                "prob_f1_win": 0.62,
                "prob_f2_win": 0.38,
                "event_id": "evt-card",
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "fighter_1": "Quote One",
                "fighter_2": "Quote Two",
                "f1_odds": 1.80,
                "f2_odds": 2.10,
                "event_id": "evt-card",
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
                "quote_timestamp": (NOW - timedelta(minutes=5)).isoformat(),
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
            }
        ]
    )
    # Exact match path uses quote names from the market row.
    matched = merge_predictions_with_odds(card, market, fetch_if_missing=False)
    # Names don't match, so no attach — but when they do via alias, quote fields stay quote-side.
    assert matched.loc[0, "odds_matched"] in (False, True)

    # Direct helper: blank quote fields stay blank.
    from src.predictor import _quote_side_name

    empty = pd.Series({"fighter_1": "", "fighter_2": None, "f1_odds": 1.8, "f2_odds": 2.1})
    assert _quote_side_name(empty, 1) == ""
    assert _quote_side_name(empty, 2) == ""
    named = pd.Series({"fighter_1": "Quote One", "fighter_2": "Quote Two"})
    assert _quote_side_name(named, 1) == "Quote One"
    assert _quote_side_name(named, 2) == "Quote Two"

    ticket = _ticket(quote_fighter_1="", quote_fighter_2="")
    assert live_quote_block_reason(ticket, now=NOW) == "missing_fighters"
    out = _allocate([ticket])
    _assert_zero(out[0])


def test_merge_keeps_blank_quote_names_when_market_row_has_none(live_session_today, monkeypatch):
    import src.predictor as predictor

    card = pd.DataFrame(
        [
            {
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "prob_f1_win": 0.70,
                "prob_f2_win": 0.30,
                "event_id": "evt-card",
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "f1_odds": 1.55,
                "f2_odds": 2.50,
                "event_id": "evt-card",
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
                "quote_timestamp": (NOW - timedelta(minutes=5)).isoformat(),
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
            }
        ]
    )
    out = predictor.merge_predictions_with_odds(card, market, fetch_if_missing=False)
    assert bool(out.loc[0, "odds_matched"])
    assert out.loc[0, "quote_fighter_1"] == "Jon Jones"
    assert out.loc[0, "quote_fighter_2"] == "Stipe Miocic"

    # If a lookup somehow returns prices without quote names, card names must not fill in.
    monkeypatch.setattr(
        predictor,
        "_lookup_odds_row",
        lambda *_a, **_k: pd.Series(
            {
                "fighter_1": "",
                "fighter_2": None,
                "f1_odds": 1.55,
                "f2_odds": 2.50,
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
                "event_id": "evt-card",
                "quote_timestamp": (NOW - timedelta(minutes=5)).isoformat(),
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
            }
        ),
    )
    blank_out = predictor.merge_predictions_with_odds(card, market, fetch_if_missing=False)
    assert bool(blank_out.loc[0, "odds_matched"])
    assert blank_out.loc[0, "quote_fighter_1"] == ""
    assert blank_out.loc[0, "quote_fighter_2"] == ""
    ticket = _ticket(
        quote_fighter_1=blank_out.loc[0, "quote_fighter_1"],
        quote_fighter_2=blank_out.loc[0, "quote_fighter_2"],
    )
    assert live_quote_block_reason(ticket, now=NOW) == "missing_fighters"
    _assert_zero(_allocate([ticket])[0])
