"""Session safety: daily loss circuit breaker for alerts and watch mode."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import config
from src.safe_io import write_json_atomic

logger = logging.getLogger(__name__)

_entry_block_reason: str | None = None


class SafetyStateUnreadable(Exception):
    """The daily-loss file exists but cannot be trusted."""


def _state_path() -> str:
    return str(config.CIRCUIT_BREAKER_STATE_PATH)


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


def update_session_bankroll_anchor(bankroll: float | None, *, session_key: str | None = None) -> None:
    """Record opening bankroll for the current card session (resets daily)."""
    if bankroll is None or bankroll <= 0:
        return
    today = date.today().isoformat()
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


def daily_loss_circuit_tripped(
    bankroll: float | None,
    *,
    session_key: str | None = None,
) -> tuple[bool, str, float]:
    """
    True when intraday/session loss exceeds profile limit.
    Returns (tripped, reason, loss_pct).
    """
    if not config.CIRCUIT_BREAKER_ENABLED:
        return False, "", 0.0
    try:
        state = load_safety_state()
    except SafetyStateUnreadable as exc:
        return True, str(exc), 0.0

    today = date.today().isoformat()
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
        update_session_bankroll_anchor(bankroll, session_key=key)
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
    session["circuit_tripped_at"] = datetime.now(timezone.utc).isoformat()
    session["loss_pct"] = round(loss_pct, 6)
    state[key] = session
    write_json_atomic(_state_path(), state)
    logger.warning(reason)
    return True, reason, loss_pct


def set_entry_block(reason: str | None) -> None:
    global _entry_block_reason
    _entry_block_reason = reason


def entry_block_active() -> bool:
    return bool(_entry_block_reason)


def entry_block_reason() -> str:
    return _entry_block_reason or ""


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
        set_entry_block(reason)
        return False, reason
    return True, ""


def live_safety_block_reason(*, bankroll: float | None = None) -> str:
    """Empty when Live stakes may stand. A tripped breaker or a bad file blocks."""
    if not bool(getattr(config, "CIRCUIT_BREAKER_ENABLED", True)):
        return ""
    try:
        tripped, reason, _ = daily_loss_circuit_tripped(bankroll)
    except SafetyStateUnreadable as exc:
        return str(exc)
    except Exception:
        return "safety state unreadable — live stakes blocked"
    if tripped:
        return reason or "daily_loss_circuit"
    # Stake clearing follows the safety file. The in-memory entry flag is only
    # for check_alerts_allowed and is not cleared when a new session file is clean.
    return ""


def apply_live_safety_gate(
    tickets: list[dict[str, Any]],
    *,
    live: bool,
    bankroll: float | None = None,
) -> str:
    """Zero every Live stake when the daily-loss state blocks or cannot be read."""
    if not live or not tickets:
        return ""
    reason = live_safety_block_reason(bankroll=bankroll)
    if not reason:
        return ""
    from src.quote_integrity import clear_actionable_stakes

    for ticket in tickets:
        clear_actionable_stakes(ticket, reason=reason)
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
        }
    today = date.today().isoformat()
    session = state.get(today, {})
    limit = daily_loss_limit_pct()
    loss_pct = session.get("loss_pct")
    return {
        "profile": config.UFC_PROFILE,
        "limit_pct": round(limit * 100, 1),
        "bankroll_open": session.get("bankroll_open"),
        "loss_pct": round(float(loss_pct) * 100, 2) if loss_pct is not None else None,
        "tripped": bool(session.get("circuit_tripped")),
        "date": session.get("session_date"),
        "entry_block": entry_block_reason(),
    }
