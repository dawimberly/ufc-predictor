"""Prediction bank — log picks, settle results, track accuracy."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sports_bot.core import config

BANK_FIELDS = [
    "prediction_id",
    "logged_at",
    "sport",
    "event",
    "selection",
    "opponent",
    "market",
    "prob",
    "odds",
    "edge",
    "confidence",
    "stake",
    "reasons",
    "status",
    "actual",
    "correct",
    "settled_at",
    "lesson",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _pid(*parts: str) -> str:
    raw = "|".join(p.strip().lower() for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def ensure_bank(path: Path | None = None) -> Path:
    config.ensure_dirs()
    path = path or config.PREDICTION_BANK_CSV
    if not path.is_file() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=BANK_FIELDS).writeheader()
    return path


def log_prediction(
    *,
    sport: str,
    event: str,
    selection: str,
    opponent: str = "",
    market: str = "moneyline",
    prob: float,
    odds: float | None = None,
    edge: float | None = None,
    confidence: str = "",
    stake: float = 0.0,
    reasons: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append (or upsert open) prediction with free-text reasons."""
    path = ensure_bank(path)
    pid = _pid(sport, event, selection, market)
    row = {
        "prediction_id": pid,
        "logged_at": _utc_now(),
        "sport": sport,
        "event": event,
        "selection": selection,
        "opponent": opponent,
        "market": market,
        "prob": f"{prob:.4f}",
        "odds": f"{odds:.2f}" if odds else "",
        "edge": f"{edge:+.3f}" if edge is not None else "",
        "confidence": confidence,
        "stake": f"{stake:.2f}" if stake else "",
        "reasons": reasons[:800],
        "status": "open",
        "actual": "",
        "correct": "",
        "settled_at": "",
        "lesson": "",
    }
    # Simple append (dedupe later in settle/load)
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=BANK_FIELDS).writerow(row)
    return row


def load_rows(path: Path | None = None) -> list[dict[str, str]]:
    path = ensure_bank(path)
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def settle_prediction(
    prediction_id: str,
    *,
    actual: str,
    path: Path | None = None,
) -> bool:
    """Mark a row settled; correct if actual matches selection (case-insensitive)."""
    path = ensure_bank(path)
    rows = load_rows(path)
    changed = False
    for row in rows:
        if row.get("prediction_id") != prediction_id:
            continue
        if row.get("status") == "settled":
            return False
        row["status"] = "settled"
        row["actual"] = actual
        row["correct"] = (
            "1" if actual.strip().lower() == str(row.get("selection") or "").strip().lower() else "0"
        )
        row["settled_at"] = _utc_now()
        changed = True
    if changed:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BANK_FIELDS)
            w.writeheader()
            w.writerows(rows)
    return changed


def accuracy_stats(path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(path)
    settled = [r for r in rows if r.get("status") == "settled"]
    correct = sum(1 for r in settled if r.get("correct") == "1")
    return {
        "total": len(rows),
        "open": sum(1 for r in rows if r.get("status") != "settled"),
        "settled": len(settled),
        "correct": correct,
        "accuracy": (correct / len(settled)) if settled else None,
    }
