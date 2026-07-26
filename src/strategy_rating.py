"""Segment rating feedback — clamp Kelly fraction from rolling performance.

Paper on by default (``STRATEGY_RATING_ENABLED``). Live stays off unless
``STRATEGY_RATING_LIVE_ENABLED=true``. Fail-closed: any error / thin sample /
disabled → multiplier 1.0.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {
    "ts": 0.0,
    "days": 0,
    "segments": {},
    "stack_mult": 1.0,
}


def _clip_mult(value: float) -> float:
    lo = float(getattr(config, "STRATEGY_RATING_MULT_MIN", 0.8) or 0.8)
    hi = float(getattr(config, "STRATEGY_RATING_MULT_MAX", 1.2) or 1.2)
    if lo > hi:
        lo, hi = hi, lo
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not (v == v):  # NaN
        return 1.0
    return round(max(lo, min(hi, v)), 4)


def score_to_kelly_multiplier(score: float, *, trade_count: int = 0) -> float:
    """Map 0–100 risk-adjusted score → Kelly mult.

    Score 50 → 1.0; Excellent (~85+) → near max; Weak → near min.
    Below ``STRATEGY_RATING_MIN_TRADES`` → neutral 1.0.
    """
    min_trades = int(getattr(config, "STRATEGY_RATING_MIN_TRADES", 8) or 8)
    if int(trade_count) < min_trades:
        return 1.0
    lo = float(getattr(config, "STRATEGY_RATING_MULT_MIN", 0.8) or 0.8)
    hi = float(getattr(config, "STRATEGY_RATING_MULT_MAX", 1.2) or 1.2)
    s = max(0.0, min(100.0, float(score)))
    if s <= 50.0:
        mult = lo + (1.0 - lo) * (s / 50.0)
    else:
        mult = 1.0 + (hi - 1.0) * ((s - 50.0) / 50.0)
    return _clip_mult(mult)


def _lookback_days() -> int:
    return max(7, int(getattr(config, "STRATEGY_RATING_LOOKBACK_DAYS", 365) or 365))


def _cache_ttl_sec() -> float:
    return max(60.0, float(getattr(config, "STRATEGY_RATING_CACHE_SEC", 3600) or 3600))


def clear_rating_cache() -> None:
    _cache["ts"] = 0.0
    _cache["days"] = 0
    _cache["segments"] = {}
    _cache["stack_mult"] = 1.0


def refresh_ratings(*, days: int | None = None, force: bool = False) -> dict[str, Any]:
    """Pull rolling segment metrics and cache Kelly multipliers."""
    days = int(days) if days is not None else _lookback_days()
    now = time.time()
    if (
        not force
        and _cache["segments"]
        and _cache["days"] == days
        and (now - float(_cache["ts"])) < _cache_ttl_sec()
    ):
        return snapshot()

    try:
        from src.strategy_performance import get_segment_ratings

        ratings = get_segment_ratings(days=days)
    except Exception as exc:
        logger.warning("strategy_rating refresh failed (fail-closed → 1.0): %s", exc)
        _cache["ts"] = now
        _cache["days"] = days
        _cache["segments"] = {}
        _cache["stack_mult"] = 1.0
        return snapshot()

    segments: dict[str, dict[str, Any]] = {}
    active_mults: list[float] = []
    min_trades = int(getattr(config, "STRATEGY_RATING_MIN_TRADES", 8) or 8)
    for key, row in (ratings.get("segments") or {}).items():
        score = float(row.get("risk_adjusted_score") or 0.0)
        trades = int(row.get("trade_count") or 0)
        mult = score_to_kelly_multiplier(score, trade_count=trades)
        segments[str(key)] = {
            "segment_key": key,
            "dimension": row.get("dimension"),
            "value": row.get("value"),
            "label": row.get("label") or key,
            "score": round(score, 1),
            "rating": row.get("rating") or "No data",
            "trade_count": trades,
            "return_pct": row.get("return_pct", 0.0),
            "sharpe": row.get("sharpe", 0.0),
            "win_rate_pct": row.get("win_rate_pct", 0.0),
            "pnl_contribution": row.get("pnl_contribution", 0.0),
            "kelly_mult": mult,
        }
        if trades >= min_trades:
            active_mults.append(mult)

    if active_mults:
        geo = 1.0
        for m in active_mults:
            geo *= float(m)
        geo = geo ** (1.0 / len(active_mults))
        blend = float(getattr(config, "STRATEGY_RATING_STACK_BLEND", 0.5) or 0.5)
        blend = max(0.0, min(1.0, blend))
        stack = _clip_mult(1.0 + blend * (geo - 1.0))
    else:
        stack = 1.0

    _cache["ts"] = now
    _cache["days"] = days
    _cache["segments"] = segments
    _cache["stack_mult"] = stack
    return snapshot()


def snapshot() -> dict[str, Any]:
    enabled = False
    try:
        enabled = bool(config.effective_strategy_rating_enabled())
    except Exception:
        enabled = False
    return {
        "as_of_ts": _cache["ts"],
        "lookback_days": _cache["days"] or _lookback_days(),
        "enabled": enabled,
        "stack_mult": float(_cache["stack_mult"] or 1.0),
        "segments": dict(_cache["segments"] or {}),
    }


def segment_kelly_multiplier(segment_key: str | None) -> float:
    """Per-segment Kelly mult (1.0 when disabled / unknown / thin data)."""
    try:
        if not config.effective_strategy_rating_enabled():
            return 1.0
    except Exception:
        return 1.0
    if not segment_key:
        return 1.0
    if not _cache["segments"]:
        try:
            refresh_ratings()
        except Exception as exc:
            logger.debug("strategy_rating refresh failed: %s", exc)
            return 1.0
    row = (_cache["segments"] or {}).get(str(segment_key))
    if not row:
        return 1.0
    return float(row.get("kelly_mult") or 1.0)


def combine_segment_multipliers(segments: dict[str, str] | None) -> float:
    """Geometric mean of qualifying segment mults; fail-closed to 1.0."""
    try:
        if not config.effective_strategy_rating_enabled():
            return 1.0
    except Exception:
        return 1.0
    if not segments:
        return 1.0
    try:
        if not _cache["segments"]:
            refresh_ratings()
    except Exception as exc:
        logger.debug("strategy_rating combine refresh failed: %s", exc)
        return 1.0

    from src.strategy_performance import SEGMENT_DIMENSIONS, segment_key

    mults: list[float] = []
    min_trades = int(getattr(config, "STRATEGY_RATING_MIN_TRADES", 8) or 8)
    for dim in SEGMENT_DIMENSIONS:
        val = segments.get(dim)
        if not val:
            continue
        key = segment_key(dim, val)
        row = (_cache["segments"] or {}).get(key)
        if not row:
            continue
        if int(row.get("trade_count") or 0) < min_trades:
            continue
        mults.append(float(row.get("kelly_mult") or 1.0))

    if not mults:
        return 1.0
    geo = 1.0
    for m in mults:
        geo *= float(m)
    return _clip_mult(geo ** (1.0 / len(mults)))


def kelly_multiplier_for_context(
    *,
    weight_class: Any = None,
    decimal_odds: Any = None,
    confidence_label: Any = None,
    prop_type: Any = None,
    market_type: str = "moneyline",
    row: pd.Series | dict[str, Any] | None = None,
) -> float:
    """Resolve combined Kelly clamp for a bet context. Fail-closed → 1.0."""
    try:
        if not config.effective_strategy_rating_enabled():
            return 1.0
        from src.strategy_performance import classify_bet_segments

        segs = classify_bet_segments(
            weight_class=weight_class,
            decimal_odds=decimal_odds,
            confidence_label=confidence_label,
            prop_type=prop_type,
            market_type=market_type,
            row=row,
        )
        return combine_segment_multipliers(segs)
    except Exception as exc:
        logger.debug("kelly_multiplier_for_context fail-closed: %s", exc)
        return 1.0


def apply_rating_to_kelly_fraction(
    kelly_fraction: float,
    *,
    rating_mult: float | None = None,
    row: pd.Series | dict[str, Any] | None = None,
    weight_class: Any = None,
    decimal_odds: Any = None,
    confidence_label: Any = None,
    prop_type: Any = None,
    market_type: str = "moneyline",
) -> float:
    """Multiply profile Kelly fraction by clipped segment rating mult."""
    try:
        base = float(kelly_fraction)
    except (TypeError, ValueError):
        return 0.0
    if base <= 0:
        return 0.0
    try:
        if not config.effective_strategy_rating_enabled():
            return base
    except Exception:
        return base

    if rating_mult is None:
        rating_mult = kelly_multiplier_for_context(
            weight_class=weight_class,
            decimal_odds=decimal_odds,
            confidence_label=confidence_label,
            prop_type=prop_type,
            market_type=market_type,
            row=row,
        )
    return base * _clip_mult(rating_mult if rating_mult is not None else 1.0)


def apply_rating_to_stake(
    stake: float,
    *,
    rating_mult: float | None = None,
    row: pd.Series | dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Scale a USD stake by segment rating (same fail-closed rules)."""
    try:
        s = float(stake)
    except (TypeError, ValueError):
        return 0.0
    if s <= 0:
        return 0.0
    try:
        if not config.effective_strategy_rating_enabled():
            return s
    except Exception:
        return s
    if rating_mult is None:
        rating_mult = kelly_multiplier_for_context(row=row, **kwargs)
    return round(s * _clip_mult(rating_mult if rating_mult is not None else 1.0), 4)


def format_strategy_rating_banner() -> str | None:
    try:
        if not config.effective_strategy_rating_enabled():
            return None
    except Exception:
        return None
    try:
        snap = refresh_ratings()
    except Exception as exc:
        logger.debug("strategy rating banner unavailable: %s", exc)
        return ">>> Strategy Rating: ON (warming up / fail-closed) <<<"
    segments = snap.get("segments") or {}
    ranked = sorted(
        [v for v in segments.values() if int(v.get("trade_count") or 0) > 0],
        key=lambda x: float(x.get("score") or 0),
        reverse=True,
    )
    days = snap.get("lookback_days") or _lookback_days()
    stack = float(snap.get("stack_mult") or 1.0)
    profile = config.profile_label()
    if not ranked:
        return (
            f">>> Strategy Rating: ON ({profile}) | {days}d | "
            f"stack x{stack:.2f} | collecting settled bets <<<"
        )
    parts = []
    for row in ranked[:4]:
        label = str(row.get("value") or row.get("segment_key") or "?")
        parts.append(f"{label} x{float(row.get('kelly_mult') or 1):.2f}")
    return (
        f">>> Strategy Rating: ON ({profile}) | {days}d | stack x{stack:.2f} | "
        f"{', '.join(parts)} <<<"
    )


def rankings_table(*, days: int | None = None) -> list[dict[str, Any]]:
    snap = refresh_ratings(days=days, force=True)
    rows = list((snap.get("segments") or {}).values())
    rows.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return rows
