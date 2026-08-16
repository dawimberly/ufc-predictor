"""Shared odds reliability helpers — placeholders, freshness, suspect edges.

UFC-only. Soft-fail friendly: one book down must not invent edges.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AUTH_PLACEHOLDERS = frozenset(
    {
        "",
        "your_cookie",
        "cookie",
        "changeme",
        "none",
        "null",
        "your_session",
        "session",
        "token",
        "placeholder",
    }
)

# Soft cap for actionable tickets; harder display blank (UI also uses 0.30).
SUSPECT_EDGE = 0.25
HARD_BLANK_EDGE = 0.30

_EDGE_BLANK_COLS = (
    "f1_odds",
    "f2_odds",
    "implied_prob_f1",
    "implied_prob_f2",
    "edge_f1",
    "edge_f2",
    "edge_pct",
    "best_edge",
)


def is_placeholder_auth(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return text.lower() in AUTH_PLACEHOLDERS


def usable_cookie(value: Any, *, min_len: int = 20) -> str:
    """Return cookie string only when real (not placeholder / too short)."""
    text = str(value or "").strip()
    if is_placeholder_auth(text) or len(text) < int(min_len):
        return ""
    return text


def usable_session_token(value: Any, *, min_len: int = 8) -> str:
    """Return session token only when real (not placeholder / too short)."""
    text = str(value or "").strip()
    if is_placeholder_auth(text) or len(text) < int(min_len):
        return ""
    return text


def cache_freshness_meta(path: Path | str | None) -> dict[str, Any]:
    """mtime / age for odds cache files (UI + logs)."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return {
            "path": str(path or ""),
            "exists": False,
            "mtime": None,
            "age_min": None,
            "bytes": 0,
        }
    st = p.stat()
    age_min = (time.time() - st.st_mtime) / 60.0
    return {
        "path": str(p),
        "exists": True,
        "mtime": float(st.st_mtime),
        "age_min": float(age_min),
        "bytes": int(st.st_size),
    }


def log_book_status(
    book: str,
    *,
    mode: str,
    matched: int | None = None,
    total: int | None = None,
    warning: str = "",
    detail: str = "",
) -> None:
    """Consistent per-book status line for Refresh / Quick Odds / props."""
    bits = [f"odds_status book={book}", f"mode={mode}"]
    if matched is not None and total is not None:
        bits.append(f"matched={matched}/{total}")
    elif matched is not None:
        bits.append(f"matched={matched}")
    if detail:
        bits.append(detail)
    if warning:
        bits.append(f"warn={warning[:160]}")
    logger.info(" | ".join(bits))


def edge_is_suspect(edge: float | None, *, hard: bool = False) -> bool:
    """True when |edge| exceeds soft (25%) or hard (30%) threshold."""
    if edge is None:
        return False
    try:
        e = float(edge)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(e):
        return False
    lim = HARD_BLANK_EDGE if hard else SUSPECT_EDGE
    return abs(e) > lim


def blank_unmatched_odds_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unmatched fights: blank odds + edges (never leave stale / invented numbers).

    Keeps fight rows so one book down does not blank the card.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    if "odds_matched" not in out.columns:
        return out
    matched = out["odds_matched"].fillna(False).astype(bool)
    unmatched = ~matched
    if not unmatched.any():
        if "edge_suspect" not in out.columns:
            out["edge_suspect"] = False
        return out

    for col in _EDGE_BLANK_COLS:
        if col in out.columns:
            out.loc[unmatched, col] = np.nan
    if "best_edge_side" in out.columns:
        out.loc[unmatched, "best_edge_side"] = ""
    if "odds_source" in out.columns:
        out.loc[unmatched, "odds_source"] = ""
    if "odds_book" in out.columns:
        out.loc[unmatched, "odds_book"] = ""
    out["edge_suspect"] = False
    return out


def apply_suspect_edge_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag |edge| > 25%; blank display edge_pct when |edge| > 30%."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    if "edge_suspect" not in out.columns:
        out["edge_suspect"] = False

    if "odds_matched" in out.columns:
        matched = out["odds_matched"].fillna(False).astype(bool)
    else:
        matched = pd.Series(True, index=out.index)

    for idx in out.index:
        if not bool(matched.loc[idx]):
            out.at[idx, "edge_suspect"] = False
            continue
        edge = out.at[idx, "edge_pct"] if "edge_pct" in out.columns else np.nan
        try:
            # edge_pct is percent points; also accept fractional edge_f*
            e_pct = float(edge) if pd.notna(edge) else float("nan")
        except (TypeError, ValueError):
            e_pct = float("nan")
        e_frac = e_pct / 100.0 if np.isfinite(e_pct) else float("nan")
        if not np.isfinite(e_frac) and "best_edge" in out.columns:
            try:
                e_frac = float(out.at[idx, "best_edge"])
            except (TypeError, ValueError):
                e_frac = float("nan")
        if not np.isfinite(e_frac):
            continue
        if edge_is_suspect(e_frac, hard=False):
            out.at[idx, "edge_suspect"] = True
        if edge_is_suspect(e_frac, hard=True):
            # Do not show invented extreme edges in tables / tickets
            if "edge_pct" in out.columns:
                out.at[idx, "edge_pct"] = np.nan
            if "best_edge" in out.columns:
                out.at[idx, "best_edge"] = np.nan
            logger.info(
                "odds_reliability: blanked suspect edge=%.1f%% fight_idx=%s",
                e_frac * 100.0,
                idx,
            )
    return out


def harden_merged_odds_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Blank unmatched + flag/blank extreme edges after merge."""
    out = blank_unmatched_odds_rows(df)
    return apply_suspect_edge_flags(out)
