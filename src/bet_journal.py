"""Structured CSV journal for UFC bet signals, alerts, and settlements."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

JOURNAL_FIELDS = [
    "timestamp",
    "event",
    "event_type",
    "fight",
    "pick",
    "edge_pct",
    "model_prob",
    "stake",
    "bankroll",
    "profile",
    "notes",
    "prediction_id",
    "pnl",
    "opening_odds",
    "closing_odds",
    "clv",
    "weight_class",
    "odds_bucket",
    "prop_type",
    "confidence",
    "correct",
    "settlement_complete",
    "skip_reason",
]


def _journal_path(path: Path | str | None = None) -> Path:
    return Path(path) if path else config.BET_JOURNAL_CSV


def _ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=JOURNAL_FIELDS).writeheader()
        return
    # Migrate older journals that lack settlement columns
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        if existing == JOURNAL_FIELDS:
            return
        rows = list(reader)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in JOURNAL_FIELDS})


def log_journal_row(
    event_type: str,
    *,
    event: str = "",
    fight: str = "",
    pick: str = "",
    edge_pct: float | str = "",
    model_prob: float | str = "",
    stake: float | str = "",
    bankroll: float | str = "",
    profile: str = "",
    notes: str = "",
    prediction_id: str = "",
    pnl: float | str = "",
    opening_odds: float | str = "",
    closing_odds: float | str = "",
    clv: float | str = "",
    weight_class: str = "",
    odds_bucket: str = "",
    prop_type: str = "",
    confidence: str = "",
    correct: str | int | bool = "",
    settlement_complete: str | int | bool = "",
    skip_reason: str = "",
    journal_path: Path | str | None = None,
) -> None:
    path = _journal_path(journal_path)
    _ensure_header(path)

    def _flag(v: str | int | bool) -> str:
        if v == "" or v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        return str(v)

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "event": event,
        "event_type": event_type,
        "fight": fight,
        "pick": pick,
        "edge_pct": edge_pct,
        "model_prob": model_prob,
        "stake": stake,
        "bankroll": bankroll,
        "profile": profile or config.UFC_PROFILE,
        "notes": notes,
        "prediction_id": prediction_id,
        "pnl": pnl,
        "opening_odds": opening_odds,
        "closing_odds": closing_odds,
        "clv": clv,
        "weight_class": weight_class,
        "odds_bucket": odds_bucket,
        "prop_type": prop_type,
        "confidence": confidence,
        "correct": _flag(correct),
        "settlement_complete": _flag(settlement_complete),
        "skip_reason": skip_reason,
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore").writerow(row)


def log_signal(
    fight: str,
    pick: str,
    *,
    event: str = "",
    edge_pct: float = 0.0,
    model_prob: float | None = None,
    stake: float = 0.0,
    bankroll: float | None = None,
    notes: str = "",
) -> None:
    log_journal_row(
        "signal",
        event=event,
        fight=fight,
        pick=pick,
        edge_pct=f"{edge_pct:+.1f}" if edge_pct else "",
        model_prob=f"{model_prob:.1%}" if model_prob is not None else "",
        stake=f"{stake:.2f}" if stake else "",
        bankroll=f"{bankroll:.2f}" if bankroll is not None else "",
        notes=notes,
    )


def log_settlement(
    *,
    prediction_id: str,
    event: str = "",
    fight: str = "",
    pick: str = "",
    correct: bool,
    stake: float | None = None,
    opening_odds: float | None = None,
    closing_odds: float | None = None,
    pnl: float | None = None,
    clv: float | None = None,
    weight_class: str = "",
    odds_bucket: str = "",
    prop_type: str = "",
    confidence: str = "",
    profile: str = "",
    settlement_complete: bool = False,
    notes: str = "",
    journal_path: Path | str | None = None,
) -> None:
    """Record a unified settlement event (bank + CLV + segment tags)."""
    note = notes or (
        "settlement_complete" if settlement_complete else "settlement_incomplete_fail_closed"
    )
    log_journal_row(
        "settle",
        event=event,
        fight=fight,
        pick=pick,
        stake=f"{stake:.2f}" if stake is not None else "",
        profile=profile,
        notes=note,
        prediction_id=prediction_id,
        pnl=f"{pnl:.4f}" if pnl is not None else "",
        opening_odds=f"{opening_odds:.4f}" if opening_odds is not None else "",
        closing_odds=f"{closing_odds:.4f}" if closing_odds is not None else "",
        clv=f"{clv:.6f}" if clv is not None else "",
        weight_class=weight_class,
        odds_bucket=odds_bucket,
        prop_type=prop_type,
        confidence=confidence,
        correct=correct,
        settlement_complete=settlement_complete,
        journal_path=journal_path,
    )


def log_alert_dispatch(
    alert_data: dict[str, Any],
    *,
    status: dict[str, Any] | None = None,
) -> None:
    status = status or {}
    notes = (
        f"singles={alert_data.get('singles_count', 0)} "
        f"parlays={alert_data.get('parlays_count', 0)} "
        f"sent={status.get('sent', False)} "
        f"skip={status.get('skip_reason', '')}"
    )
    log_journal_row(
        "alert_dispatch",
        event=str(alert_data.get("event_name", "")),
        bankroll=alert_data.get("bankroll", ""),
        notes=notes[:500],
    )
    for s in alert_data.get("singles", []):
        log_signal(
            s.get("fight", ""),
            s.get("pick", ""),
            event=str(alert_data.get("event_name", "")),
            edge_pct=float(s.get("edge_pct", 0)),
            model_prob=s.get("prob"),
            stake=float(s.get("suggested_stake", 0)),
            bankroll=alert_data.get("bankroll"),
            notes=s.get("brief") or s.get("reasoning", "")[:200],
        )


def log_watch_tick(
    *,
    iteration: int,
    event_name: str,
    singles: int,
    parlays: int,
    notified: bool,
    block_reason: str = "",
) -> None:
    log_journal_row(
        "watch_tick",
        event=event_name,
        notes=f"iter={iteration} singles={singles} parlays={parlays} notify={notified} {block_reason}",
    )
