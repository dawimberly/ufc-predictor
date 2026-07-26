"""DraftKings UFC odds via The Odds API (bookmakers=draftkings)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config
from src.data_loader import ensure_data_dirs
from src.predictor import OddsAPIError, _implied_probs, _names_match, _to_decimal_odds

logger = logging.getLogger(__name__)

DK_CACHE_PATH = config.CACHE_DIR / "draftkings_odds.csv"
BOOKMAKER_KEY = "draftkings"
DK_UNAVAILABLE_MSG = "DraftKings unavailable - check THE_ODDS_API_KEY"

# Set when returning consensus / cache fallback so the dashboard can show a warning.
LAST_WARNING: str = ""


def _cache_fresh(path: Path) -> bool:
    """Reuse DraftKings Odds-API caches; ODDS_FETCH_ONCE ignores TTL once file exists."""
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < 32:
            return False
    except OSError:
        return False
    if bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        return True
    if config.ODDS_CACHE_TTL_HOURS <= 0:
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < config.ODDS_CACHE_TTL_HOURS


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        cached = pd.read_csv(path, parse_dates=["commence_time"])
    except Exception as exc:
        logger.debug("DraftKings cache read failed: %s", exc)
        return None
    if cached is None or cached.empty:
        return None
    return cached


def _is_unauthorized(exc: BaseException, response: requests.Response | None = None) -> bool:
    status = getattr(response, "status_code", None)
    if status == 401:
        return True
    text = str(exc).lower()
    return "401" in text or "unauthorized" in text


def _consensus_odds(*, force_refresh: bool) -> pd.DataFrame:
    """Multi-book consensus from The Odds API (no draftkings-only filter)."""
    from src.predictor import fetch_ufc_odds

    odds_df = fetch_ufc_odds(force_refresh=force_refresh)
    if odds_df.empty:
        raise OddsAPIError("Consensus odds empty")
    out = odds_df.copy()
    if "bookmaker" not in out.columns:
        out["bookmaker"] = "Consensus"
    return out


def _fallback_odds(*, force_refresh: bool, reason: str) -> pd.DataFrame:
    """
    Soft recovery: stale DK cache → Odds API consensus → BetNow → MyBookie.
    Never raises — empty DataFrame when nothing usable remains.
    """
    global LAST_WARNING

    stale = _read_cache(DK_CACHE_PATH)
    if stale is not None:
        LAST_WARNING = f"{DK_UNAVAILABLE_MSG} Using cached DraftKings lines ({reason})."
        logger.warning(LAST_WARNING)
        return stale

    try:
        consensus = _consensus_odds(force_refresh=force_refresh)
        LAST_WARNING = (
            f"{DK_UNAVAILABLE_MSG} Showing consensus odds from other books ({reason})."
        )
        logger.warning(LAST_WARNING)
        return consensus
    except Exception as api_exc:
        logger.warning("%s Consensus fallback failed: %s", DK_UNAVAILABLE_MSG, api_exc)

    # Odds API down (e.g. 401) → automatic BetNow / MyBookie scrapers
    try:
        from src.odds_providers.odds_fallback import fetch_book_scraper_odds, LAST_ODDS_META

        scraped = fetch_book_scraper_odds(force_refresh=force_refresh)
        if scraped is not None and not scraped.empty:
            src = str(LAST_ODDS_META.get("source") or "book scraper")
            LAST_WARNING = (
                f"{DK_UNAVAILABLE_MSG} Using {src} odds instead ({reason})."
            )
            logger.warning(LAST_WARNING)
            return scraped
    except Exception as scrape_exc:
        logger.warning("Book scraper fallback after DraftKings failed: %s", scrape_exc)

    LAST_WARNING = DK_UNAVAILABLE_MSG
    return pd.DataFrame()


def fetch_draftkings_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch DraftKings h2h lines for UFC/MMA.

    Returns columns aligned with merge_predictions_with_odds:
    fighter_1, fighter_2, f1_odds, f2_odds, implied_prob_f1, implied_prob_f2, bookmaker.

    On 401 / auth failure: falls back to stale cache, then consensus odds from other
    books. Returns an empty frame (with LAST_WARNING set) when no fallback works so
    the dashboard never crashes.
    """
    global LAST_WARNING
    LAST_WARNING = ""
    ensure_data_dirs()

    if not config.ODDS_API_KEY:
        return _fallback_odds(force_refresh=force_refresh, reason="missing API key")

    if not force_refresh and _cache_fresh(DK_CACHE_PATH):
        cached = _read_cache(DK_CACHE_PATH)
        if cached is not None:
            logger.info("Using cached DraftKings odds (%s rows)", len(cached))
            return cached

    # Hard block: DraftKings uses The Odds API directly (bypasses odds_api_get).
    try:
        from src.odds_providers.odds_api_client import (
            OddsApiFetchBlocked,
            ensure_live_odds_api_allowed,
        )

        ensure_live_odds_api_allowed(context="draftkings moneylines")
    except OddsApiFetchBlocked:
        cached = _read_cache(DK_CACHE_PATH)
        if cached is not None:
            logger.info("ODDS_FETCH_ONCE: blocked live DK odds — using stale cache (%s rows)", len(cached))
            return cached
        return _fallback_odds(force_refresh=False, reason="ODDS_FETCH_ONCE blocked live pull")

    url = f"{config.ODDS_API_BASE_URL}/sports/{config.ODDS_API_SPORT}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.ODDS_API_REGIONS,
        "markets": config.ODDS_API_MARKETS,
        "oddsFormat": config.ODDS_API_ODDS_FORMAT,
        "bookmakers": BOOKMAKER_KEY,
    }

    response: requests.Response | None = None
    try:
        response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT_SEC)
        if response.status_code == 401:
            return _fallback_odds(force_refresh=False, reason="401 Unauthorized")
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as exc:
        if _is_unauthorized(exc, response):
            return _fallback_odds(force_refresh=False, reason="401 Unauthorized")
        logger.warning("DraftKings odds HTTP error: %s", exc)
        return _fallback_odds(force_refresh=False, reason=str(exc))
    except requests.RequestException as exc:
        if _is_unauthorized(exc, response):
            return _fallback_odds(force_refresh=False, reason="401 Unauthorized")
        logger.warning("DraftKings odds request failed: %s", exc)
        return _fallback_odds(force_refresh=False, reason=str(exc))
    except json.JSONDecodeError as exc:
        logger.warning("DraftKings odds returned invalid JSON: %s", exc)
        return _fallback_odds(force_refresh=False, reason="invalid JSON")

    if not isinstance(payload, list):
        logger.warning("Unexpected DraftKings response: %s", type(payload))
        return _fallback_odds(force_refresh=False, reason="unexpected response")

    rows: list[dict[str, Any]] = []
    for event in payload:
        home = str(event.get("home_team", "")).strip()
        away = str(event.get("away_team", "")).strip()
        if not home or not away:
            continue

        f1_px: float | None = None
        f2_px: float | None = None

        for book in event.get("bookmakers", []):
            if str(book.get("key", "")).lower() != BOOKMAKER_KEY:
                continue
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices: dict[str, float] = {}
                for outcome in market.get("outcomes", []):
                    name = str(outcome.get("name", "")).strip()
                    price = outcome.get("price")
                    if name and price is not None:
                        prices[name] = _to_decimal_odds(
                            float(price), config.ODDS_API_ODDS_FORMAT
                        )
                f1_px = next((px for nm, px in prices.items() if _names_match(nm, home)), None)
                f2_px = next((px for nm, px in prices.items() if _names_match(nm, away)), None)
                break

        if not f1_px or not f2_px or f1_px <= 1 or f2_px <= 1:
            continue

        imp1, imp2 = _implied_probs(f1_px, f2_px)
        rows.append(
            {
                "event_id": event.get("id", ""),
                "commence_time": event.get("commence_time"),
                "fighter_1": home,
                "fighter_2": away,
                "f1_odds": round(f1_px, 3),
                "f2_odds": round(f2_px, 3),
                "implied_prob_f1": imp1,
                "implied_prob_f2": imp2,
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
            }
        )

    odds_df = pd.DataFrame(rows)
    if odds_df.empty:
        logger.warning("No DraftKings UFC h2h odds returned - trying consensus fallback")
        return _fallback_odds(force_refresh=force_refresh, reason="no DraftKings lines")

    if "commence_time" in odds_df.columns:
        odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], errors="coerce")

    odds_df.to_csv(DK_CACHE_PATH, index=False)
    logger.info("Fetched %s DraftKings odds lines", len(odds_df))
    return odds_df
