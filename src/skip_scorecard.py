"""Skip-reason scorecard — tokenize non-bets, log, weekly rollup.

Fail-closed: unknown / empty reasons collapse to ``unknown`` (never silent).
Helps answer: are we skipping noise or leaving edge on the table?
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

# Canonical reason codes (tokenize targets)
SKIP_MIN_EDGE = "min_edge"
SKIP_BELOW_TIGHTENED = "below_tightened_min_edge"
SKIP_HIGH_DISAGREEMENT = "high_disagreement"
SKIP_WIDE_INTERVAL = "wide_interval"
SKIP_MISSING_UNCERTAINTY = "missing_uncertainty"
SKIP_STRATEGY_RATING = "strategy_rating"
SKIP_CIRCUIT = "circuit"
SKIP_COOLDOWN = "cooldown"
SKIP_NO_ODDS = "no_odds"
SKIP_NO_PICK = "no_pick"
SKIP_EDGE_NOT_ACTIONABLE = "edge_not_actionable"
SKIP_STALE_MODEL = "stale_model"
SKIP_NARRATIVE_REJECT = "narrative_reject"
SKIP_DRAWDOWN = "drawdown_halt"
SKIP_LOW_MODEL_PROB = "low_model_prob"
SKIP_LOW_CONFIDENCE = "low_confidence"
SKIP_CARD_BET_CAP = "card_bet_cap"
SKIP_UNKNOWN = "unknown"

CANONICAL_CODES: frozenset[str] = frozenset(
    {
        SKIP_MIN_EDGE,
        SKIP_BELOW_TIGHTENED,
        SKIP_HIGH_DISAGREEMENT,
        SKIP_WIDE_INTERVAL,
        SKIP_MISSING_UNCERTAINTY,
        SKIP_STRATEGY_RATING,
        SKIP_CIRCUIT,
        SKIP_COOLDOWN,
        SKIP_NO_ODDS,
        SKIP_NO_PICK,
        SKIP_EDGE_NOT_ACTIONABLE,
        SKIP_STALE_MODEL,
        SKIP_NARRATIVE_REJECT,
        SKIP_DRAWDOWN,
        SKIP_LOW_MODEL_PROB,
        SKIP_LOW_CONFIDENCE,
        SKIP_CARD_BET_CAP,
        SKIP_UNKNOWN,
        # Tighten labels sometimes appear on skip rows — keep as codes
        "elevated_disagreement",
        "elevated_interval_width",
    }
)

# Heuristic buckets for the noise-vs-edge question
NOISE_FILTER_CODES: frozenset[str] = frozenset(
    {
        SKIP_HIGH_DISAGREEMENT,
        SKIP_WIDE_INTERVAL,
        SKIP_MISSING_UNCERTAINTY,
        SKIP_EDGE_NOT_ACTIONABLE,
        SKIP_NO_ODDS,
        SKIP_NO_PICK,
        SKIP_STALE_MODEL,
        SKIP_CIRCUIT,
        SKIP_DRAWDOWN,
        SKIP_COOLDOWN,
        SKIP_LOW_MODEL_PROB,
        SKIP_LOW_CONFIDENCE,
        SKIP_CARD_BET_CAP,
        "elevated_disagreement",
        "elevated_interval_width",
    }
)

EDGE_LEFT_CODES: frozenset[str] = frozenset(
    {
        SKIP_MIN_EDGE,
        SKIP_BELOW_TIGHTENED,
        SKIP_STRATEGY_RATING,
        SKIP_NARRATIVE_REJECT,
        SKIP_LOW_MODEL_PROB,
        SKIP_LOW_CONFIDENCE,
        SKIP_CARD_BET_CAP,
    }
)

# Map free-text / aliases → canonical
_ALIASES: dict[str, str] = {
    "uncertainty": SKIP_MISSING_UNCERTAINTY,
    "missing": SKIP_MISSING_UNCERTAINTY,
    "disagreement": SKIP_HIGH_DISAGREEMENT,
    "interval": SKIP_WIDE_INTERVAL,
    "wide_ci": SKIP_WIDE_INTERVAL,
    "minedge": SKIP_MIN_EDGE,
    "below_edge": SKIP_MIN_EDGE,
    "tightened": SKIP_BELOW_TIGHTENED,
    "below_tightened_min_edge": SKIP_BELOW_TIGHTENED,
    "circuit_breaker": SKIP_CIRCUIT,
    "daily loss": SKIP_CIRCUIT,
    "drawdown": SKIP_DRAWDOWN,
    "peak drawdown": SKIP_DRAWDOWN,
    "cooldown": SKIP_COOLDOWN,
    "dedup": SKIP_COOLDOWN,
    "no odds": SKIP_NO_ODDS,
    "stale": SKIP_STALE_MODEL,
    "narrative": SKIP_NARRATIVE_REJECT,
    "low_prob": SKIP_LOW_MODEL_PROB,
    "low model": SKIP_LOW_MODEL_PROB,
    "low_model_prob": SKIP_LOW_MODEL_PROB,
    "low_confidence": SKIP_LOW_CONFIDENCE,
    "low confidence": SKIP_LOW_CONFIDENCE,
    "card_bet_cap": SKIP_CARD_BET_CAP,
    "card cap": SKIP_CARD_BET_CAP,
    "max_bets": SKIP_CARD_BET_CAP,
}


def skip_scorecard_jsonl_path() -> Path:
    path = Path(getattr(config, "SKIP_SCORECARD_JSONL", config.LOG_DIR / "skip_scorecard.jsonl"))
    if not path.is_absolute():
        path = config.ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def skip_scorecard_json_path() -> Path:
    path = Path(getattr(config, "SKIP_SCORECARD_JSON", config.DATA_DIR / "skip_scorecard.json"))
    if not path.is_absolute():
        path = config.ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tokenize_skip_reason(raw: Any) -> str:
    """Normalize any skip label to a canonical code (fail-closed → unknown)."""
    text = str(raw or "").strip().lower()
    if not text:
        return SKIP_UNKNOWN
    # Prefer first token if comma-joined gate reasons
    primary = text.split(",")[0].strip()
    primary = primary.replace(" ", "_").replace("-", "_")
    if primary in CANONICAL_CODES:
        return primary
    if text in CANONICAL_CODES:
        return text
    for alias, code in _ALIASES.items():
        if alias in text or alias.replace(" ", "_") in primary:
            return code
    # Partial matches on known codes
    for code in CANONICAL_CODES:
        if code in primary or code in text:
            return code
    return SKIP_UNKNOWN


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log_skip(
    reason: str,
    *,
    event: str = "",
    fight: str = "",
    pick: str = "",
    edge_pct: float | str = "",
    model_prob: float | str = "",
    profile: str = "",
    context: str = "",
    notes: str = "",
    disagreement: float | None = None,
    interval_width: float | None = None,
    journal: bool = True,
    scorecard: bool = True,
) -> str:
    """
    Tokenize + persist one skip to journal and JSONL scorecard.

    Returns the canonical reason code used.
    """
    code = tokenize_skip_reason(reason)
    profile_s = profile or getattr(config, "UFC_PROFILE", "paper")
    note_bits = [notes] if notes else []
    if context:
        note_bits.append(f"context={context}")
    if disagreement is not None:
        note_bits.append(f"disagreement={disagreement}")
    if interval_width is not None:
        note_bits.append(f"interval_width={interval_width}")
    note = " | ".join(x for x in note_bits if x)

    if journal:
        try:
            from src.bet_journal import log_journal_row

            log_journal_row(
                "skip",
                event=event,
                fight=fight,
                pick=pick,
                edge_pct=edge_pct,
                model_prob=model_prob,
                stake=0,
                profile=profile_s,
                notes=note or f"skip_reason={code}",
                skip_reason=code,
            )
        except Exception as exc:
            logger.debug("skip journal failed: %s", exc)

    if scorecard:
        try:
            row = {
                "ts": _utc_now(),
                "skip_reason": code,
                "event": event,
                "fight": fight,
                "pick": pick,
                "edge_pct": edge_pct,
                "profile": profile_s,
                "context": context,
                "disagreement": disagreement,
                "interval_width": interval_width,
                "notes": note,
            }
            path = skip_scorecard_jsonl_path()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:
            logger.debug("skip scorecard jsonl failed: %s", exc)

    return code


def record_skip_dict(item: dict[str, Any], *, event: str = "", context: str = "alert") -> str:
    """Log a skipped[] row from generate_alerts."""
    return log_skip(
        str(item.get("skip_reason") or item.get("reason") or SKIP_UNKNOWN),
        event=event or str(item.get("event") or ""),
        fight=str(item.get("fight") or ""),
        pick=str(item.get("pick") or ""),
        edge_pct=item.get("edge_pct", ""),
        profile=str(item.get("profile") or ""),
        context=context,
        disagreement=item.get("disagreement"),
        interval_width=item.get("interval_width"),
    )


def ingest_alert_skips(alert_data: dict[str, Any] | None, *, context: str = "alert") -> int:
    """Persist every row in alert_data['skipped']. Returns count logged."""
    if not alert_data:
        return 0
    skipped = alert_data.get("skipped") or []
    event = str(alert_data.get("event_name") or "")
    n = 0
    for item in skipped:
        if not isinstance(item, dict):
            continue
        # Avoid double-logging rows already marked
        if item.get("_scorecard_logged"):
            continue
        record_skip_dict(item, event=event, context=context)
        item["_scorecard_logged"] = True
        n += 1
    return n


def _parse_ts(raw: str) -> datetime | None:
    text = str(raw or "").replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _load_jsonl_rows(*, days: int | None = 7) -> list[dict[str, Any]]:
    path = skip_scorecard_jsonl_path()
    if not path.is_file():
        return []
    cutoff = None
    if days is not None and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff is not None:
                    ts = _parse_ts(str(obj.get("ts") or ""))
                    if ts is None or ts < cutoff:
                        continue
                rows.append(obj)
    except OSError as exc:
        logger.debug("skip scorecard read failed: %s", exc)
    return rows


def _load_journal_skips(*, days: int | None = 7) -> list[dict[str, Any]]:
    """Fallback / supplement from bet_journal skip + uncertainty_skip rows."""
    path = Path(getattr(config, "BET_JOURNAL_CSV", config.DATA_DIR / "bet_journal.csv"))
    if not path.is_file():
        return []
    import csv

    cutoff = None
    if days is not None and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                et = str(row.get("event_type") or "").strip().lower()
                if et not in ("skip", "uncertainty_skip"):
                    continue
                if cutoff is not None:
                    ts = _parse_ts(str(row.get("timestamp") or ""))
                    if ts is None or ts < cutoff:
                        continue
                reason = row.get("skip_reason") or ""
                if not reason:
                    notes = str(row.get("notes") or "")
                    if "skip_reason=" in notes:
                        reason = notes.split("skip_reason=", 1)[1].split("|", 1)[0].strip()
                out.append(
                    {
                        "ts": row.get("timestamp"),
                        "skip_reason": tokenize_skip_reason(reason),
                        "event": row.get("event"),
                        "fight": row.get("fight"),
                        "pick": row.get("pick"),
                        "profile": row.get("profile"),
                        "context": "journal",
                    }
                )
    except OSError as exc:
        logger.debug("journal skip read failed: %s", exc)
    return out


def rollup_skip_reasons(
    *,
    days: int = 7,
    include_journal: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    """
    Weekly (or custom) rollup: count + % per skip reason.

    Fail-closed: empty history → complete=False, empty counts.
    """
    rows = _load_jsonl_rows(days=days)
    if include_journal:
        # Prefer JSONL; journal fills gaps for older uncertainty_skip-only logs
        seen = {
            (
                str(r.get("ts")),
                str(r.get("fight")),
                str(r.get("skip_reason")),
            )
            for r in rows
        }
        for jr in _load_journal_skips(days=days):
            key = (str(jr.get("ts")), str(jr.get("fight")), str(jr.get("skip_reason")))
            if key not in seen:
                rows.append(jr)
                seen.add(key)

    counts: Counter[str] = Counter()
    for r in rows:
        counts[tokenize_skip_reason(r.get("skip_reason"))] += 1

    total = sum(counts.values())
    by_reason: list[dict[str, Any]] = []
    for code, n in counts.most_common():
        pct = (100.0 * n / total) if total else 0.0
        bucket = (
            "noise_filter"
            if code in NOISE_FILTER_CODES
            else ("edge_left" if code in EDGE_LEFT_CODES else "other")
        )
        by_reason.append(
            {
                "skip_reason": code,
                "count": n,
                "pct": round(pct, 1),
                "bucket": bucket,
            }
        )

    noise_n = sum(c["count"] for c in by_reason if c["bucket"] == "noise_filter")
    edge_n = sum(c["count"] for c in by_reason if c["bucket"] == "edge_left")
    other_n = total - noise_n - edge_n

    if total == 0:
        interpretation = "No skip history in window (fail-closed: nothing to judge)."
    elif noise_n >= edge_n * 1.5:
        interpretation = (
            "Mostly skipping noise (uncertainty / odds / safety). "
            "Filters look protective, not leaving much clear edge on the table."
        )
    elif edge_n >= noise_n * 1.5:
        interpretation = (
            "Many skips are min-edge / sizing related — possible edge left on the table. "
            "Review thresholds before loosening."
        )
    else:
        interpretation = (
            "Mixed: both noise filters and edge-floor skips. "
            "Inspect top reasons before changing gates."
        )

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": int(days),
        "complete": total > 0,
        "fail_closed": total == 0,
        "total_skips": total,
        "by_reason": by_reason,
        "top_reasons": by_reason[:5],
        "noise_filter_count": noise_n,
        "edge_left_count": edge_n,
        "other_count": other_n,
        "noise_filter_pct": round(100.0 * noise_n / total, 1) if total else 0.0,
        "edge_left_pct": round(100.0 * edge_n / total, 1) if total else 0.0,
        "interpretation": interpretation,
    }
    if write_json:
        try:
            skip_scorecard_json_path().write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.debug("skip scorecard json write failed: %s", exc)
    return payload


def format_rollup_text(rollup: dict[str, Any] | None = None, *, days: int = 7) -> str:
    """CLI / dashboard friendly multi-line summary."""
    data = rollup if rollup is not None else rollup_skip_reasons(days=days, write_json=False)
    lines = [
        f"Skip scorecard ({data.get('window_days', days)}d) - "
        f"{data.get('total_skips', 0)} skips"
    ]
    top = data.get("top_reasons") or data.get("by_reason") or []
    if not top:
        lines.append("  (no skips logged - fail-closed)")
    else:
        for row in top[:8]:
            lines.append(
                f"  {row['skip_reason']}: {row['count']} ({row['pct']:.0f}%) [{row.get('bucket', '')}]"
            )
        lines.append(
            f"  noise filters {data.get('noise_filter_pct', 0):.0f}% | "
            f"edge-floor {data.get('edge_left_pct', 0):.0f}%"
        )
        lines.append(f"  -> {data.get('interpretation', '')}")
    return "\n".join(lines)


def top_skip_lines(rollup: dict[str, Any] | None = None, *, days: int = 7, limit: int = 5) -> list[str]:
    data = rollup if rollup is not None else rollup_skip_reasons(days=days, write_json=False)
    out: list[str] = []
    for row in (data.get("top_reasons") or [])[:limit]:
        out.append(f"{row['skip_reason']} {row['count']} ({row['pct']:.0f}%)")
    return out
