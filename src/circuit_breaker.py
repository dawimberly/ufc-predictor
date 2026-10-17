"""Session safety: daily loss circuit breaker for alerts and watch mode.

Live keeps three operations apart:

* ``initialize_live_session`` writes today's opening bankroll. Only the
  ``start-live-session`` command calls it, and it never overwrites a session.
* ``record_bankroll_observation`` stores the operator's current bankroll with
  its source and UTC observation time (``record-bankroll`` command). It never
  touches the opening bankroll or a trip.
* ``evaluate_live_daily_loss`` / ``live_safety_block_reason`` authorize Live
  stakes. They only read the file. Missing, stale, or unusable state blocks,
  and so does a disabled breaker.

Live never uses a bankroll number handed in by a caller: configured or
budget-file values carry no observation time, so they can neither authorize
stakes nor hide a loss.

Paper keeps its original behavior: the first alert check of the day anchors
the Paper session under the plain date key, and the breaker switch applies.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

import config
from src.safe_io import write_json_atomic

logger = logging.getLogger(__name__)

_entry_block_reason: str | None = None

LIVE_SESSION_PREFIX = "live:"
START_LIVE_SESSION_HINT = "python main.py start-live-session --bankroll <amount>"
RECORD_BANKROLL_HINT = "python main.py record-bankroll --bankroll <amount>"
SOURCE_SESSION_START = "start-live-session"
SOURCE_RECORD_BANKROLL = "record-bankroll"
OBSERVATION_SOURCES = (SOURCE_SESSION_START, SOURCE_RECORD_BANKROLL)
_CLOCK_SKEW = timedelta(minutes=2)


class SafetyStateUnreadable(Exception):
    """The daily-loss file exists but cannot be trusted."""


class LiveSessionExists(Exception):
    """Today's Live session is already recorded and will not be overwritten."""


class NoLiveSession(Exception):
    """There is no Live session for today to record a bankroll against."""


class BankrollObservation(NamedTuple):
    value: float
    source: str
    observed_at: datetime


class LiveSafetyDecision(NamedTuple):
    allowed: bool
    reason: str
    loss_pct: float | None
    # True only when a valid session and a valid recorded bankroll exceed the limit.
    computed_trip: bool = False
    current: BankrollObservation | None = None


def _state_path() -> str:
    return str(config.CIRCUIT_BREAKER_STATE_PATH)


def _today() -> date:
    return date.today()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _session_day(today: date | None = None) -> str:
    return (today or _today()).isoformat()


def live_session_key(today: date | None = None) -> str:
    return f"{LIVE_SESSION_PREFIX}{_session_day(today)}"


def load_safety_state() -> dict[str, Any]:
    """Read the circuit-breaker file. A missing file is an empty state.

    A present file that cannot be parsed is a failure. Callers must not treat
    that as "no loss recorded".
    """
    path = Path(_state_path())
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError, TypeError) as exc:
        raise SafetyStateUnreadable(f"safety state unreadable: {path.name}") from exc
    if not isinstance(data, dict):
        raise SafetyStateUnreadable("safety state is not an object")
    return data


def daily_loss_limit_pct() -> float:
    return config.profile_value("daily_loss_limit_fraction")


def _breaker_enabled() -> bool:
    return bool(getattr(config, "CIRCUIT_BREAKER_ENABLED", True))


def _live_loss_limit() -> float | None:
    """Live limit regardless of which profile the menu shows."""
    try:
        if config.is_live_profile():
            raw = config.profile_value("daily_loss_limit_fraction")
        else:
            raw = config._PROFILE_LIVE["daily_loss_limit_fraction"]
        limit = float(raw)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(limit) or limit <= 0 or limit >= 1:
        return None
    return limit


def _bankroll_max_age() -> timedelta | None:
    try:
        minutes = float(getattr(config, "LIVE_BANKROLL_MAX_AGE_MINUTES", 0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(minutes) or minutes <= 0:
        return None
    return timedelta(minutes=minutes)


def _finite_positive(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _parse_utc(value: Any) -> datetime | None:
    """Timezone-aware ISO time, or None. A naive time has no known zone and is refused."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_tripped(session: dict[str, Any]) -> bool:
    flag = session.get("circuit_tripped")
    return flag not in (None, False, 0)


def _latest_live_day(state: dict[str, Any]) -> str:
    days = [
        key[len(LIVE_SESSION_PREFIX):]
        for key in state
        if isinstance(key, str) and key.startswith(LIVE_SESSION_PREFIX)
    ]
    return max(days) if days else ""


def _disabled_decision() -> LiveSafetyDecision:
    return LiveSafetyDecision(
        False,
        "live daily-loss protection disabled (CIRCUIT_BREAKER_ENABLED=false) - live stakes blocked",
        None,
    )


def _live_session_record(
    today: date | None = None,
) -> tuple[dict[str, Any] | None, float | None, LiveSafetyDecision | None]:
    """Today's untripped Live session and its anchor, or the blocking decision. Read-only."""
    path = Path(_state_path())
    if not path.exists():
        return None, None, LiveSafetyDecision(
            False,
            f"live safety state missing - start today's Live session: {START_LIVE_SESSION_HINT}",
            None,
        )
    try:
        state = load_safety_state()
    except SafetyStateUnreadable as exc:
        return None, None, LiveSafetyDecision(False, f"{exc} - live stakes blocked", None)

    day = _session_day(today)
    legacy = state.get(day)
    if isinstance(legacy, dict) and _is_tripped(legacy):
        loss = _finite_positive(legacy.get("loss_pct"))
        return None, None, LiveSafetyDecision(
            False,
            f"daily loss circuit breaker (stored trip {loss or 0.0:.2%} for {day})",
            loss,
        )

    session = state.get(live_session_key(today))
    if session is None:
        last = _latest_live_day(state)
        stale = f" (last Live session {last})" if last else ""
        return None, None, LiveSafetyDecision(
            False,
            f"no Live session for {day}{stale} - start one: {START_LIVE_SESSION_HINT}",
            None,
        )
    if not isinstance(session, dict):
        return None, None, LiveSafetyDecision(False, "live session record is not an object", None)
    if str(session.get("session_date") or "") != day:
        return None, None, LiveSafetyDecision(
            False,
            f"live session date {session.get('session_date')!r} does not match {day}",
            None,
        )
    if _is_tripped(session):
        loss = _finite_positive(session.get("loss_pct"))
        return None, None, LiveSafetyDecision(
            False,
            f"daily loss circuit breaker (stored trip {loss or 0.0:.2%} for {day})",
            loss,
        )
    anchor = _finite_positive(session.get("bankroll_open"))
    if anchor is None:
        return None, None, LiveSafetyDecision(False, "live session anchor missing or invalid", None)
    return session, anchor, None


def _session_observation(
    session: dict[str, Any],
    now: datetime,
) -> tuple[BankrollObservation | None, str]:
    """The session's recorded current bankroll, or why it cannot be used."""
    value = _finite_positive(session.get("current_bankroll"))
    if value is None:
        return None, f"current bankroll missing or invalid - record it: {RECORD_BANKROLL_HINT}"
    source = str(session.get("current_bankroll_source") or "").strip()
    if source not in OBSERVATION_SOURCES:
        return None, f"current bankroll source {source or 'missing'!r} is not an operator record"
    observed = _parse_utc(session.get("current_bankroll_observed_at"))
    if observed is None:
        return None, "current bankroll observation time missing or invalid"
    started = _parse_utc(session.get("initialized_at"))
    if started is None:
        return None, "live session start time missing or invalid"
    if observed < started - _CLOCK_SKEW:
        return None, "current bankroll was recorded before today's Live session started"
    if observed - now > _CLOCK_SKEW:
        return None, "current bankroll observation time is in the future"
    max_age = _bankroll_max_age()
    if max_age is None:
        return None, "live bankroll max age unreadable"
    age = now - observed
    if age > max_age:
        return None, (
            f"current bankroll recorded {age.total_seconds() / 60:.0f} min ago "
            f"(max {max_age.total_seconds() / 60:.0f} min) - record it: {RECORD_BANKROLL_HINT}"
        )
    return BankrollObservation(value, source, observed), ""


def evaluate_live_daily_loss(
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> LiveSafetyDecision:
    """Read-only Live authorization. Never creates, resets, or trips a session."""
    if not _breaker_enabled():
        return _disabled_decision()
    session, anchor, blocked = _live_session_record(today)
    if blocked is not None:
        return blocked
    assert session is not None and anchor is not None
    current, problem = _session_observation(session, now or _now())
    if current is None:
        return LiveSafetyDecision(False, problem, None)
    limit = _live_loss_limit()
    if limit is None:
        return LiveSafetyDecision(False, "live daily loss limit unreadable", None, current=current)

    loss_pct = (anchor - current.value) / anchor
    if loss_pct >= limit - 1e-9:
        return LiveSafetyDecision(
            False,
            f"daily loss circuit breaker ({loss_pct:.2%} >= {limit:.2%} limit, profile=live)",
            loss_pct,
            computed_trip=True,
            current=current,
        )
    return LiveSafetyDecision(True, "", loss_pct, current=current)


def live_safety_block_reason(
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> str:
    """Empty when Live stakes may stand. Read-only."""
    try:
        decision = evaluate_live_daily_loss(today=today, now=now)
    except Exception:
        return "safety state unreadable - live stakes blocked"
    return "" if decision.allowed else (decision.reason or "live safety state blocked")


def live_observed_bankroll(
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> BankrollObservation | None:
    """Today's usable recorded bankroll for Live sizing, or None. Read-only."""
    try:
        session, _, blocked = _live_session_record(today)
    except Exception:
        return None
    if blocked is not None or session is None:
        return None
    current, _ = _session_observation(session, now or _now())
    return current


def initialize_live_session(
    bankroll: float | None,
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record today's opening Live bankroll. Explicit use only.

    Refuses a missing or invalid bankroll, an unreadable file, and an existing
    session for today (including a tripped one). The opening bankroll is also
    the first recorded current bankroll.
    """
    opening = _finite_positive(bankroll)
    if opening is None:
        raise ValueError("Live session needs a finite bankroll greater than 0")
    state = load_safety_state()
    day = _session_day(today)
    key = live_session_key(today)
    existing = state.get(key)
    if existing is not None:
        detail = ""
        if isinstance(existing, dict):
            detail = (
                f" (anchor {existing.get('bankroll_open')!r}, "
                f"tripped={bool(_is_tripped(existing))})"
            )
        raise LiveSessionExists(f"Live session for {day} already exists{detail}; not overwritten")
    stamp = (now or _now()).astimezone(timezone.utc).isoformat()
    session = {
        "session_date": day,
        "bankroll_open": round(opening, 4),
        "profile": "live",
        "initialized_at": stamp,
        "initialized_by": "explicit",
        "current_bankroll": round(opening, 4),
        "current_bankroll_source": SOURCE_SESSION_START,
        "current_bankroll_observed_at": stamp,
    }
    state[key] = session
    if not write_json_atomic(_state_path(), state):
        raise OSError(f"could not write {Path(_state_path()).name}")
    logger.info("Live session started for %s with opening bankroll %.2f", day, opening)
    return dict(session)


def record_bankroll_observation(
    bankroll: float | None,
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> BankrollObservation:
    """Store the operator's current bankroll on today's session. Explicit use only.

    The opening bankroll and any trip are left exactly as they are.
    """
    value = _finite_positive(bankroll)
    if value is None:
        raise ValueError("current bankroll must be a finite amount greater than 0")
    state = load_safety_state()
    day = _session_day(today)
    key = live_session_key(today)
    session = state.get(key)
    if not isinstance(session, dict) or str(session.get("session_date") or "") != day:
        raise NoLiveSession(f"no Live session for {day} - start one: {START_LIVE_SESSION_HINT}")
    observed = (now or _now()).astimezone(timezone.utc)
    session["current_bankroll"] = round(value, 4)
    session["current_bankroll_source"] = SOURCE_RECORD_BANKROLL
    session["current_bankroll_observed_at"] = observed.isoformat()
    state[key] = session
    if not write_json_atomic(_state_path(), state):
        raise OSError(f"could not write {Path(_state_path()).name}")
    logger.info("Live bankroll recorded for %s: %.2f", day, value)
    return BankrollObservation(round(value, 4), SOURCE_RECORD_BANKROLL, observed)


def _record_live_trip(decision: LiveSafetyDecision, *, today: date | None = None) -> None:
    """Persist a trip the alert gate already computed against a valid session."""
    try:
        state = load_safety_state()
    except SafetyStateUnreadable:
        return
    session = state.get(live_session_key(today))
    if not isinstance(session, dict) or _is_tripped(session):
        return
    session["circuit_tripped"] = True
    session["circuit_tripped_at"] = _now().isoformat()
    session["loss_pct"] = round(float(decision.loss_pct or 0.0), 6)
    state[live_session_key(today)] = session
    write_json_atomic(_state_path(), state)
    logger.warning(decision.reason)


def _anchor_paper_session(bankroll: float, *, session_key: str | None = None) -> None:
    """Paper's original first-check anchor under the plain date key."""
    today = _session_day()
    key = session_key or today
    state = load_safety_state()
    session = state.setdefault(key, {})
    if session.get("session_date") != today:
        session["session_date"] = today
        session["bankroll_open"] = round(float(bankroll), 4)
        session.pop("circuit_tripped", None)
        session.pop("loss_pct", None)
        state[key] = session
        write_json_atomic(_state_path(), state)


def update_session_bankroll_anchor(bankroll: float | None, *, session_key: str | None = None) -> None:
    """Explicitly anchor today's session for the active profile.

    Live goes through ``initialize_live_session`` and leaves an existing
    session alone. Paper keeps the plain date-key anchor.
    """
    if bankroll is None or bankroll <= 0:
        return
    if config.is_live_profile():
        try:
            initialize_live_session(bankroll)
        except LiveSessionExists:
            logger.info("Live session already started today; anchor unchanged")
        return
    _anchor_paper_session(float(bankroll), session_key=session_key)


def _paper_daily_loss(
    bankroll: float | None,
    *,
    session_key: str | None = None,
) -> tuple[bool, str, float]:
    try:
        state = load_safety_state()
    except SafetyStateUnreadable as exc:
        return True, str(exc), 0.0

    today = _session_day()
    key = session_key or today
    session = state.get(key) if isinstance(state.get(key), dict) else {}
    if session.get("circuit_tripped"):
        loss_pct = float(session.get("loss_pct") or 0.0)
        reason = (
            f"daily loss circuit breaker (stored trip {loss_pct:.2%}, "
            f"profile={config.UFC_PROFILE})"
        )
        return True, reason, loss_pct
    if bankroll is None or bankroll <= 0:
        return False, "", 0.0

    try:
        _anchor_paper_session(float(bankroll), session_key=key)
        state = load_safety_state()
    except SafetyStateUnreadable as exc:
        return True, str(exc), 0.0
    session = state.get(key, {})
    open_br = float(session.get("bankroll_open") or bankroll)
    if open_br <= 0:
        return False, "", 0.0

    loss_pct = (open_br - float(bankroll)) / open_br
    limit = daily_loss_limit_pct()
    if loss_pct < limit - 1e-9:
        return False, "", loss_pct

    reason = (
        f"daily loss circuit breaker ({loss_pct:.2%} >= {limit:.2%} limit, "
        f"profile={config.UFC_PROFILE})"
    )
    session["circuit_tripped"] = True
    session["circuit_tripped_at"] = _now().isoformat()
    session["loss_pct"] = round(loss_pct, 6)
    state[key] = session
    write_json_atomic(_state_path(), state)
    logger.warning(reason)
    return True, reason, loss_pct


def daily_loss_circuit_tripped(
    bankroll: float | None,
    *,
    session_key: str | None = None,
) -> tuple[bool, str, float]:
    """Alert-gate check. Returns (tripped, reason, loss_pct).

    Live reads today's session and its recorded bankroll; ``bankroll`` is not
    used. A computed trip is recorded on the existing session so it stays
    blocking for the day, and a disabled breaker blocks. Paper keeps its
    first-check anchor and its on/off switch.
    """
    if config.is_live_profile():
        decision = evaluate_live_daily_loss()
        if decision.allowed:
            return False, "", float(decision.loss_pct or 0.0)
        if decision.computed_trip:
            _record_live_trip(decision)
        return True, decision.reason, float(decision.loss_pct or 0.0)
    if not _breaker_enabled():
        return False, "", 0.0
    return _paper_daily_loss(bankroll, session_key=session_key)


def set_entry_block(reason: str | None) -> None:
    global _entry_block_reason
    _entry_block_reason = reason


def entry_block_active() -> bool:
    return bool(_entry_block_reason)


def entry_block_reason() -> str:
    return _entry_block_reason or ""


def _sticky_block_reason(reason: str) -> bool:
    """Only a real loss trip sticks for the process. Missing/stale state can recover."""
    text = str(reason or "")
    return "stored trip" in text or text.startswith("daily loss circuit breaker (")


def check_alerts_allowed(
    bankroll: float | None = None,
    *,
    drawdown_halted: bool = False,
    drawdown_reason: str = "",
) -> tuple[bool, str]:
    """Combined gate for new alerts: circuit breaker + drawdown halt + manual block."""
    if entry_block_active():
        return False, entry_block_reason()
    if drawdown_halted:
        return False, drawdown_reason or "peak drawdown halt active"
    tripped, reason, _ = daily_loss_circuit_tripped(bankroll)
    if tripped:
        if _sticky_block_reason(reason):
            set_entry_block(reason)
        return False, reason
    return True, ""


def apply_live_safety_gate(
    tickets: list[dict[str, Any]],
    *,
    live: bool,
) -> str:
    """Zero every Live stake when the daily-loss state blocks or cannot be read."""
    if not live or not tickets:
        return ""
    reason = live_safety_block_reason()
    if not reason:
        return ""
    from src.quote_integrity import clear_actionable_stakes

    for ticket in tickets:
        if not isinstance(ticket, dict):
            continue
        earlier = str(ticket.get("quote_block_reason") or "").strip()
        keep = earlier if earlier and ticket.get("quote_actionable") is False else reason
        clear_actionable_stakes(ticket, reason=keep)
        ticket["safety_block_reason"] = reason
    return reason


def get_circuit_status() -> dict[str, Any]:
    try:
        state = load_safety_state()
    except SafetyStateUnreadable:
        return {
            "profile": config.UFC_PROFILE,
            "limit_pct": None,
            "bankroll_open": None,
            "loss_pct": None,
            "tripped": True,
            "date": None,
            "entry_block": "safety state unreadable",
            "unreadable": True,
            "live_block_reason": "safety state unreadable - live stakes blocked",
        }
    today = _session_day()
    session = state.get(today, {})
    limit = daily_loss_limit_pct()
    loss_pct = session.get("loss_pct")
    live_session = state.get(live_session_key())
    return {
        "profile": config.UFC_PROFILE,
        "limit_pct": round(limit * 100, 1),
        "bankroll_open": session.get("bankroll_open"),
        "loss_pct": round(float(loss_pct) * 100, 2) if loss_pct is not None else None,
        "tripped": bool(session.get("circuit_tripped")),
        "date": session.get("session_date"),
        "entry_block": entry_block_reason(),
        "live_session": dict(live_session) if isinstance(live_session, dict) else None,
        "live_block_reason": live_safety_block_reason(),
    }


def _bankroll_parser(prog: str, description: str, help_text: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--bankroll", type=float, required=True, help=help_text)
    return parser


def start_live_session_cli(argv: list[str] | None = None) -> int:
    """``python main.py start-live-session --bankroll N``."""
    parser = _bankroll_parser(
        "main.py start-live-session",
        "Record today's opening Live bankroll for the daily-loss breaker.",
        "Opening bankroll in USD.",
    )
    args = parser.parse_args(argv or [])
    try:
        session = initialize_live_session(args.bankroll)
    except LiveSessionExists as exc:
        print(f"Not changed: {exc}")
        return 1
    except SafetyStateUnreadable as exc:
        print(f"Not changed: {exc}. Fix or move the file by hand; it was not overwritten.")
        return 1
    except (ValueError, OSError) as exc:
        print(f"Not changed: {exc}")
        return 1
    print(
        f"Live session {session['session_date']} started with opening bankroll "
        f"${session['bankroll_open']:,.2f}."
    )
    return 0


def record_bankroll_cli(argv: list[str] | None = None) -> int:
    """``python main.py record-bankroll --bankroll N``."""
    parser = _bankroll_parser(
        "main.py record-bankroll",
        "Record your current Live bankroll (with the time) for the daily-loss breaker.",
        "Current bankroll in USD, as it stands now.",
    )
    args = parser.parse_args(argv or [])
    try:
        observation = record_bankroll_observation(args.bankroll)
    except SafetyStateUnreadable as exc:
        print(f"Not changed: {exc}. Fix or move the file by hand; it was not overwritten.")
        return 1
    except (NoLiveSession, ValueError, OSError) as exc:
        print(f"Not changed: {exc}")
        return 1
    print(
        f"Live bankroll ${observation.value:,.2f} recorded at "
        f"{observation.observed_at.strftime('%Y-%m-%d %H:%M UTC')}."
    )
    reason = live_safety_block_reason()
    if reason:
        print(f"Live stakes still blocked: {reason}")
    return 0
