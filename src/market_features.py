"""Market features for UFC fights — leakage-safe when odds are pre-fight.

Gated via ``config.ENABLE_MARKET_FEATURES`` + ``MARKET_FEATURE_COLUMNS``.
``line_move`` is NaN unless opening odds exist without leakage.
``model_minus_mkt`` is computed at evaluation time (needs model probs).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "mkt_implied_prob",
    "line_move",
)

# Eval-only residual name (not trained unless nested OOF is added later).
MARKET_RESIDUAL_COLUMN = "model_minus_mkt"


def decimal_odds_to_implied_pair(
    f1_odds: Any, f2_odds: Any
) -> tuple[float, float]:
    """De-vig 2-way implied probs from decimal (or American) odds."""
    try:
        o1 = float(f1_odds)
        o2 = float(f2_odds)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not (o1 > 1.0 and o2 > 1.0):
        # American
        if abs(o1) >= 100 or abs(o2) >= 100:
            def _am_to_dec(odds: float) -> float:
                if odds >= 100:
                    return 1.0 + odds / 100.0
                if odds <= -100:
                    return 1.0 + 100.0 / abs(odds)
                return float("nan")

            o1, o2 = _am_to_dec(o1), _am_to_dec(o2)
        else:
            return float("nan"), float("nan")
    if not (o1 > 1.0 and o2 > 1.0):
        return float("nan"), float("nan")
    p1 = 1.0 / o1
    p2 = 1.0 / o2
    denom = p1 + p2
    if denom <= 0:
        return float("nan"), float("nan")
    return float(p1 / denom), float(p2 / denom)


def attach_market_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ``mkt_implied_prob`` and ``line_move`` columns.

    Uses closing/available ``f1_odds``/``f2_odds`` (historical merge is pre-fight
    closing-style). Opening→close line move requires ``f1_open_odds`` /
    ``f2_open_odds``; absent those, ``line_move`` is NaN (no leakage fill).
    """
    if features is None or features.empty:
        return features
    out = features.copy()

    if "implied_prob_f1" in out.columns:
        out["mkt_implied_prob"] = pd.to_numeric(out["implied_prob_f1"], errors="coerce")
    elif {"f1_odds", "f2_odds"}.issubset(out.columns):
        from src.feature_engineering import decimal_odds_to_implied

        out["mkt_implied_prob"] = decimal_odds_to_implied(out["f1_odds"], out["f2_odds"])
    else:
        out["mkt_implied_prob"] = np.nan

    # Opening odds not in current UFC sources — leave NaN (documented coverage 0%).
    if {"f1_open_odds", "f2_open_odds", "f1_odds", "f2_odds"}.issubset(out.columns):
        open_imp = []
        close_imp = []
        for _, row in out.iterrows():
            o_p, _ = decimal_odds_to_implied_pair(row.get("f1_open_odds"), row.get("f2_open_odds"))
            c_p, _ = decimal_odds_to_implied_pair(row.get("f1_odds"), row.get("f2_odds"))
            open_imp.append(o_p)
            close_imp.append(c_p)
        open_s = pd.Series(open_imp, index=out.index, dtype=float)
        close_s = pd.Series(close_imp, index=out.index, dtype=float)
        out["line_move"] = close_s - open_s  # + = f1 shortened (favorite more)
    else:
        out["line_move"] = np.nan

    return out


def model_minus_market(
    proba: np.ndarray | pd.Series,
    mkt_implied: np.ndarray | pd.Series,
) -> np.ndarray:
    """Residual p_model − mkt_implied_prob (f1)."""
    p = np.asarray(proba, dtype=float)
    m = np.asarray(mkt_implied, dtype=float)
    return p - m


def shrink_proba_toward_market(
    proba: np.ndarray,
    mkt_implied: np.ndarray,
    ci_width: np.ndarray,
    *,
    width_threshold: float = 0.40,
    shrink: float = 0.35,
) -> np.ndarray:
    """
    Research-only CAL: when CI is wide, blend model p toward market.

    ``p' = (1-w)*p + w*mkt`` with ``w = shrink`` only when width ≥ threshold
    and market is present; otherwise leave p unchanged.
    """
    p = np.asarray(proba, dtype=float).copy()
    m = np.asarray(mkt_implied, dtype=float)
    w = np.asarray(ci_width, dtype=float)
    mask = (w >= float(width_threshold)) & np.isfinite(m) & np.isfinite(p)
    alpha = float(np.clip(shrink, 0.0, 1.0))
    p[mask] = (1.0 - alpha) * p[mask] + alpha * m[mask]
    return p


def log_market_coverage(
    features: pd.DataFrame, *, year: int = 2025, label: str = ""
) -> dict[str, float]:
    coverage: dict[str, float] = {}
    if features is None or features.empty:
        return coverage
    date_col = "event_date" if "event_date" in features.columns else "date"
    sample = features
    if date_col in features.columns:
        dts = pd.to_datetime(features[date_col], errors="coerce")
        sample = features.loc[dts.dt.year == year]
    if sample.empty:
        sample = features
    n = len(sample)
    for col in MARKET_FEATURE_COLUMNS:
        if col not in sample.columns:
            coverage[col] = 0.0
            continue
        coverage[col] = float(sample[col].notna().mean() * 100.0)
    logger.info(
        "Market coverage [%s]: year=%s n=%s mkt_implied=%.1f%% line_move=%.1f%%",
        label or "n/a",
        year,
        n,
        coverage.get("mkt_implied_prob", 0.0),
        coverage.get("line_move", 0.0),
    )
    return coverage
