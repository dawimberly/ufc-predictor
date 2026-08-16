"""Detect card slate changes and invalidate stale odds caches (ODDS_FETCH_ONCE).

When Refresh Next Two loads a new roster, moneyline caches from the prior slate
cause name_mismatch 0/N even with a valid THE_ODDS_API_KEY.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

_SLATE_FP_FILE = "odds_slate_fingerprint.txt"


def _marker_path() -> Path:
    return Path(getattr(config, "CACHE_DIR", Path("data/cache"))) / _SLATE_FP_FILE


def combined_slate_fingerprint(combined: pd.DataFrame) -> str:
    """Stable hash from loaded card fight pairs (+ event names when present)."""
    if combined is None or not isinstance(combined, pd.DataFrame) or combined.empty:
        return ""
    f1 = combined.get("fighter_1", combined.get("fighter1", pd.Series(dtype=str)))
    f2 = combined.get("fighter_2", combined.get("fighter2", pd.Series(dtype=str)))
    ev = combined.get("event_name", combined.get("event", pd.Series(dtype=str)))
    bits: list[str] = []
    for i in range(len(combined)):
        a = str(f1.iloc[i] if i < len(f1) else "").strip().lower()
        b = str(f2.iloc[i] if i < len(f2) else "").strip().lower()
        if not a or not b:
            continue
        pair = tuple(sorted((a, b)))
        e = str(ev.iloc[i] if i < len(ev) else "").strip().lower()
        bits.append(f"{e}|{pair[0]}|{pair[1]}")
    if not bits:
        return ""
    bits.sort()
    digest = hashlib.sha256("\n".join(bits).encode("utf-8")).hexdigest()[:16]
    return digest


def read_odds_slate_fingerprint() -> str:
    p = _marker_path()
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def clear_mybookie_odds_caches() -> list[str]:
    """Remove MyBookie moneyline/prop caches so a new scrape is allowed."""
    cache_dir = Path(getattr(config, "CACHE_DIR", "") or ".")
    candidates = [
        cache_dir / "mybookie_odds.csv",
        cache_dir / "mybookie_prop_odds.csv",
    ]
    removed: list[str] = []
    for p in candidates:
        try:
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except OSError as exc:
            logger.debug("Could not remove MyBookie cache %s: %s", p, exc)
    if removed:
        logger.info("Cleared MyBookie odds caches: %s", ", ".join(removed))
    return removed


def invalidate_odds_caches_for_slate_change(
    combined: pd.DataFrame,
    *,
    reason: str = "card_slate_changed",
) -> bool:
    """
    If the loaded card roster differs from the last odds download slate,
    delete Odds API + MyBookie once-caches before merge.

    Returns True when any cache file was removed.
    """
    new_fp = combined_slate_fingerprint(combined)
    if not new_fp:
        return False
    old_fp = read_odds_slate_fingerprint()
    if not old_fp or old_fp == new_fp:
        return False

    from src.odds_providers.odds_api_client import clear_odds_api_fetch_once_caches

    cleared: list[str] = []
    cleared.extend(clear_odds_api_fetch_once_caches())
    cleared.extend(clear_mybookie_odds_caches())
    logger.warning(
        "Odds slate changed (%s → %s, reason=%s) — cleared %d cache file(s)",
        old_fp,
        new_fp,
        reason,
        len(cleared),
    )
    return bool(cleared)


def save_odds_slate_fingerprint(combined: pd.DataFrame) -> str:
    """
    Persist slate fingerprint after a successful card load.

    Clears stale odds caches when the roster changed since the last save.
    """
    new_fp = combined_slate_fingerprint(combined)
    if not new_fp:
        return ""
    invalidate_odds_caches_for_slate_change(combined, reason="refresh_next_two")
    p = _marker_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new_fp + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write odds slate fingerprint: %s", exc)
    return new_fp


def min_match_for_card(total_fights: int) -> int:
    """Minimum matched lines before we treat odds cache as healthy."""
    n = int(total_fights or 0)
    if n <= 0:
        return 0
    return max(3, (n + 1) // 2)
