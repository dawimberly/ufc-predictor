"""Automatic odds fallback — free sources first.

Priority:
  1. The Odds API (free-tier key from env) — cached 15–30 min
  2. Action Network UFC scoreboard scrape (optional, fail-soft)
  3. DraftKings / BetNow / MyBookie — optional only, off by default

Fail-closed only when every enabled source returns no usable lines:
  ``NO BET — no usable odds (fail-closed)``
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Last attempt metadata for callers / dashboard warnings
LAST_ODDS_META: dict[str, Any] = {
    "source": "",
    "sources_tried": [],
    "warning": "",
    "fail_closed": True,
    "n_rows": 0,
}

REQUIRED_ODDS_COLS = ("fighter_1", "fighter_2", "f1_odds", "f2_odds")
NO_USABLE_ODDS_MSG = "NO BET — no usable odds (fail-closed)"


def _normalize_odds_frame(df: pd.DataFrame, *, bookmaker: str) -> pd.DataFrame:
    """Ensure canonical columns for merge_predictions_with_odds."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Alias columns
    if "fighter_1" not in out.columns and "fighter1" in out.columns:
        out = out.rename(columns={"fighter1": "fighter_1", "fighter2": "fighter_2"})
    for col in REQUIRED_ODDS_COLS:
        if col not in out.columns:
            return pd.DataFrame()
    out["f1_odds"] = pd.to_numeric(out["f1_odds"], errors="coerce")
    out["f2_odds"] = pd.to_numeric(out["f2_odds"], errors="coerce")
    usable = out[(out["f1_odds"] > 1.0) & (out["f2_odds"] > 1.0)].copy()
    if usable.empty:
        return pd.DataFrame()
    if "bookmaker" not in usable.columns or usable["bookmaker"].isna().all():
        usable["bookmaker"] = bookmaker
    if "bookmaker_count" not in usable.columns:
        usable["bookmaker_count"] = 1
    if "odds_source" not in usable.columns:
        usable["odds_source"] = bookmaker
    return usable.reset_index(drop=True)


def odds_frame_usable(df: pd.DataFrame | None) -> bool:
    if df is None or getattr(df, "empty", True):
        return False
    try:
        return not _normalize_odds_frame(df, bookmaker="check").empty
    except Exception:
        return False


def _try_fetch(
    label: str,
    fn: Callable[..., pd.DataFrame],
    *,
    force_refresh: bool,
) -> tuple[pd.DataFrame, str | None]:
    """Return (normalized_df, error_message)."""
    try:
        raw = fn(force_refresh=force_refresh)
    except Exception as exc:
        logger.info("Odds source %s failed: %s", label, exc)
        return pd.DataFrame(), str(exc)
    norm = _normalize_odds_frame(raw, bookmaker=label)
    if norm.empty:
        return pd.DataFrame(), f"{label}: no usable lines"
    if "odds_source" not in norm.columns:
        norm["odds_source"] = label
    else:
        norm["odds_source"] = norm["odds_source"].fillna(label)
    return norm, None


def fetch_book_scraper_odds(
    *,
    force_refresh: bool = False,
    skip_sources: set[str] | None = None,
) -> pd.DataFrame:
    """
    Optional book scrapers (off by default): BetNow → MyBookie.

    Used only when ``BETNOW_ENABLED`` / ``MYBOOKIE_ENABLED`` are true.
    """
    skip = {s.lower() for s in (skip_sources or set())}

    def _skip_label(label: str) -> bool:
        low = label.lower()
        tokens = {low, low.replace(".eu", "")}
        if "betnow" in low:
            tokens.add("betnow")
        if "mybookie" in low:
            tokens.add("mybookie")
        return bool(tokens & skip)

    tried: list[str] = []
    chain: list[tuple[str, Callable[..., pd.DataFrame]]] = []
    if getattr(config, "BETNOW_ENABLED", False):
        try:
            from src.odds_providers.betnow_scraper import fetch_betnow_odds

            if not _skip_label("BetNow.eu"):
                chain.append(("BetNow.eu", fetch_betnow_odds))
        except Exception as exc:
            logger.debug("BetNow import failed: %s", exc)
    if getattr(config, "MYBOOKIE_ENABLED", False):
        try:
            from src.odds_providers.mybookie_scraper import fetch_mybookie_odds

            if not _skip_label("MyBookie"):
                chain.append(("MyBookie", fetch_mybookie_odds))
        except Exception as exc:
            logger.debug("MyBookie import failed: %s", exc)

    if not chain:
        LAST_ODDS_META.clear()
        LAST_ODDS_META.update(
            {
                "source": "",
                "sources_tried": [],
                "warning": "Book scrapers disabled (BETNOW_ENABLED/MYBOOKIE_ENABLED=false).",
                "fail_closed": True,
                "n_rows": 0,
            }
        )
        return pd.DataFrame()

    for label, fn in chain:
        tried.append(label)
        df, err = _try_fetch(label, fn, force_refresh=force_refresh)
        if not df.empty:
            logger.info("Odds scraper fallback hit: %s (%s rows)", label, len(df))
            meta = {
                "source": label,
                "sources_tried": tried,
                "warning": f"Using {label} odds (free primary sources unavailable).",
                "fail_closed": False,
                "n_rows": len(df),
            }
            LAST_ODDS_META.clear()
            LAST_ODDS_META.update(meta)
            return df
        if err:
            logger.debug("Scraper %s: %s", label, err)

    LAST_ODDS_META.clear()
    LAST_ODDS_META.update(
        {
            "source": "",
            "sources_tried": tried,
            "warning": "All enabled book scrapers returned no usable lines.",
            "fail_closed": True,
            "n_rows": 0,
        }
    )
    return pd.DataFrame()


def fetch_best_available_odds(
    *,
    force_refresh: bool = False,
    skip_sources: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Try Odds API → Action Network → optional DK/BetNow/MyBookie.

    Returns (odds_df, meta). Empty df + fail_closed when nothing usable.
    """
    skip = {s.lower() for s in (skip_sources or set())}
    tried: list[str] = []
    errors: list[str] = []

    def _want(name: str) -> bool:
        return name.lower() not in skip and name.lower().replace(".eu", "") not in skip

    ttl_m = int(getattr(config, "ODDS_CACHE_TTL_MINUTES", 20) or 20)

    # 1) The Odds API (free tier) — primary
    if _want("odds_api") or _want("consensus"):
        tried.append("OddsAPI")
        try:
            from src.predictor import fetch_ufc_odds

            df, err = _try_fetch("OddsAPI", fetch_ufc_odds, force_refresh=force_refresh)
            if not df.empty:
                meta = {
                    "source": "OddsAPI",
                    "sources_tried": list(tried),
                    "warning": "",
                    "fail_closed": False,
                    "n_rows": len(df),
                    "cache_ttl_minutes": ttl_m,
                }
                LAST_ODDS_META.clear()
                LAST_ODDS_META.update(meta)
                logger.info(
                    "Odds fallback: using Odds API consensus (%s rows, cache_ttl=%sm)",
                    len(df),
                    ttl_m,
                )
                return df, meta
            if err:
                errors.append(err)
                low = err.lower()
                if "401" in low or "unauthorized" in low:
                    logger.warning(
                        "Odds API unauthorized — trying Action Network / optional books"
                    )
                    skip.add("draftkings")
        except Exception as exc:
            errors.append(str(exc))
            low = str(exc).lower()
            if "401" in low or "unauthorized" in low or "missing odds api" in low:
                skip.add("draftkings")

    # 2) Action Network (free backup scrape — fail-soft)
    if getattr(config, "ACTION_NETWORK_ENABLED", True) and _want("actionnetwork"):
        tried.append("ActionNetwork")
        try:
            from src.odds_providers.action_network import fetch_action_network_odds

            df, err = _try_fetch(
                "ActionNetwork", fetch_action_network_odds, force_refresh=force_refresh
            )
            if not df.empty:
                meta = {
                    "source": "ActionNetwork",
                    "sources_tried": list(tried),
                    "warning": "Using Action Network UFC odds (Odds API unavailable).",
                    "fail_closed": False,
                    "n_rows": len(df),
                    "cache_ttl_minutes": ttl_m,
                }
                LAST_ODDS_META.clear()
                LAST_ODDS_META.update(meta)
                logger.info(
                    "Odds fallback: using Action Network (%s rows)", len(df)
                )
                return df, meta
            if err:
                errors.append(err)
        except Exception as exc:
            errors.append(str(exc))

    # 3) DraftKings (optional — off by default; shares Odds API quota when used)
    if getattr(config, "DRAFTKINGS_ENABLED", False) and _want("draftkings"):
        tried.append("DraftKings")
        try:
            from src.odds_providers.draftkings import fetch_draftkings_odds

            df, err = _try_fetch(
                "DraftKings", fetch_draftkings_odds, force_refresh=force_refresh
            )
            if not df.empty:
                try:
                    from src.odds_providers import draftkings as dk

                    warn = str(getattr(dk, "LAST_WARNING", "") or "")
                except Exception:
                    warn = ""
                src = "DraftKings"
                if "consensus" in warn.lower():
                    src = "Consensus (via DraftKings fallback)"
                elif "cached" in warn.lower():
                    src = "DraftKings (cached)"
                meta = {
                    "source": src,
                    "sources_tried": list(tried),
                    "warning": warn,
                    "fail_closed": False,
                    "n_rows": len(df),
                    "cache_ttl_minutes": ttl_m,
                }
                LAST_ODDS_META.clear()
                LAST_ODDS_META.update(meta)
                logger.info("Odds fallback: using %s (%s rows)", src, len(df))
                return df, meta
            if err:
                errors.append(err)
        except Exception as exc:
            errors.append(str(exc))

    # 4) Optional book scrapers (BetNow / MyBookie — off by default)
    scraped = fetch_book_scraper_odds(
        force_refresh=force_refresh,
        skip_sources=skip,
    )
    if not scraped.empty:
        meta = dict(LAST_ODDS_META)
        meta["sources_tried"] = tried + list(meta.get("sources_tried") or [])
        meta["cache_ttl_minutes"] = ttl_m
        LAST_ODDS_META.clear()
        LAST_ODDS_META.update(meta)
        return scraped, meta

    tried.extend(list(LAST_ODDS_META.get("sources_tried") or []))
    warning = NO_USABLE_ODDS_MSG
    detail = " Sources tried: " + (", ".join(tried) if tried else "(none)") + "."
    if errors:
        detail += " Last errors: " + "; ".join(errors[-3:])
    warning = f"{NO_USABLE_ODDS_MSG}{detail}"
    meta = {
        "source": "",
        "sources_tried": tried,
        "warning": warning,
        "fail_closed": True,
        "n_rows": 0,
        "no_usable_odds": True,
        "cache_ttl_minutes": ttl_m,
    }
    LAST_ODDS_META.clear()
    LAST_ODDS_META.update(meta)
    logger.warning(warning)
    return pd.DataFrame(), meta


def attach_odds_source_columns(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    default_source: str = "",
) -> pd.DataFrame:
    """Copy odds_source / bookmaker onto matched prediction rows when present on odds."""
    out = predictions
    if "odds_source" not in out.columns:
        out = out.copy()
        out["odds_source"] = ""
    if "odds_book" not in out.columns:
        out = out.copy() if out is predictions else out
        out["odds_book"] = ""
    if default_source and "odds_matched" in out.columns:
        matched = out["odds_matched"].fillna(False).astype(bool)
        out.loc[matched & (out["odds_source"].astype(str).str.strip() == ""), "odds_source"] = (
            default_source
        )
        out.loc[matched & (out["odds_book"].astype(str).str.strip() == ""), "odds_book"] = (
            default_source
        )
    return out
