"""Unified bet settlement: PnL, closing odds (CLV), segment tags, ledger sync.

Fail-closed: missing stake/opening odds → outcome may settle, but PnL/CLV stay
empty and incomplete rows are excluded from threshold health feedback.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compute_pnl(*, correct: bool, stake: float | None, opening_odds: float | None) -> float | None:
    """Cash PnL at taken (opening) odds. None if stake/odds missing (fail-closed)."""
    stake_f = _safe_float(stake)
    odds_f = _safe_float(opening_odds)
    if stake_f is None or stake_f <= 0 or odds_f is None or odds_f <= 1.0:
        return None
    return stake_f * (odds_f - 1.0) if correct else -stake_f


def compute_clv(*, opening_odds: float | None, closing_odds: float | None) -> float | None:
    """
    Closing-line value in probability points for the bet side.

    clv = implied_close - implied_open = 1/O_close - 1/O_open
    Positive → beat the close (got a better price than closing line).
    None if either price missing (fail-closed).
    """
    o_open = _safe_float(opening_odds)
    o_close = _safe_float(closing_odds)
    if o_open is None or o_close is None or o_open <= 1.0 or o_close <= 1.0:
        return None
    return (1.0 / o_close) - (1.0 / o_open)


def closing_odds_from_fight_row(
    hit: pd.Series | dict[str, Any],
    *,
    pick: str,
    fighter_1: str,
    fighter_2: str,
) -> float | None:
    """Extract closing decimal odds for the picked side from a completed fight row."""
    if isinstance(hit, dict):
        hit = pd.Series(hit)

    def _clean(name: Any) -> str:
        return str(name or "").strip().lower()

    pick_c = _clean(pick)
    f1_c = _clean(fighter_1)
    f2_c = _clean(fighter_2)

    # Explicit closing columns
    for key in ("closing_odds", "close_odds", "closing_decimal"):
        val = _safe_float(hit.get(key))
        if val is not None and val > 1.0:
            return val

    # Side-specific close / last known market
    side_keys_f1 = ("f1_closing_odds", "f1_odds", "fighter1_odds", "r_odds")
    side_keys_f2 = ("f2_closing_odds", "f2_odds", "fighter2_odds", "b_odds")
    if pick_c and f1_c and (pick_c == f1_c or pick_c in f1_c or f1_c in pick_c):
        for key in side_keys_f1:
            val = _safe_float(hit.get(key))
            if val is not None and val > 1.0:
                return val
    if pick_c and f2_c and (pick_c == f2_c or pick_c in f2_c or f2_c in pick_c):
        for key in side_keys_f2:
            val = _safe_float(hit.get(key))
            if val is not None and val > 1.0:
                return val
    return None


def settlement_complete(
    *,
    stake: float | None,
    opening_odds: float | None,
    pnl: float | None,
) -> bool:
    """True when we have enough data for health / ROI feedback."""
    return (
        _safe_float(stake) is not None
        and float(stake) > 0
        and _safe_float(opening_odds) is not None
        and float(opening_odds) > 1.0
        and pnl is not None
    )


def health_lookback_days(profile: str | None = None) -> int:
    import config

    prof = config.normalize_profile(profile or config.UFC_PROFILE)
    if prof == "live":
        return max(14, int(getattr(config, "LIVE_HEALTH_LOOKBACK_DAYS", 180) or 180))
    return max(14, int(getattr(config, "PAPER_HEALTH_LOOKBACK_DAYS", 90) or 90))
