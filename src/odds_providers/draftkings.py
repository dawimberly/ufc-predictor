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
from src.predictor import _implied_probs, _names_match, _to_decimal_odds

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


def rows_from_odds_api_events(payload: list[Any]) -> pd.DataFrame:
    """DraftKings h2h rows from an Odds API event list. Other books are ignored."""
    rows: list[dict[str, Any]] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
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
    if not odds_df.empty and "commence_time" in odds_df.columns:
        odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], errors="coerce")
    return odds_df


def persist_draftkings_from_payload(payload: list[Any]) -> pd.DataFrame:
    """Save DraftKings rows taken from the main Odds API download. No extra request."""
    global LAST_WARNING
    odds_df = rows_from_odds_api_events(payload)
    if odds_df.empty:
        return odds_df
    ensure_data_dirs()
    odds_df.to_csv(DK_CACHE_PATH, index=False)
    LAST_WARNING = ""
    logger.info("Saved %s DraftKings lines from the main Odds API download", len(odds_df))
    return odds_df


def _fallback_odds(*, force_refresh: bool, reason: str) -> pd.DataFrame:
    """Return a real DraftKings cache only. Never substitute another book's prices."""
    del force_refresh
    global LAST_WARNING

    stale = _read_cache(DK_CACHE_PATH)
    if stale is not None:
        LAST_WARNING = f"{DK_UNAVAILABLE_MSG} Using cached DraftKings lines ({reason})."
        logger.warning(LAST_WARNING)
        return stale

    LAST_WARNING = f"{DK_UNAVAILABLE_MSG} No DraftKings lines ({reason})."
    logger.warning(LAST_WARNING)
    return pd.DataFrame()


def fetch_draftkings_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch DraftKings h2h lines for UFC/MMA.

    Returns columns aligned with merge_predictions_with_odds:
    fighter_1, fighter_2, f1_odds, f2_odds, implied_prob_f1, implied_prob_f2, bookmaker.

    Uses the DraftKings rows saved during the main Odds API download when that
    cache exists. A separate request runs only when the DraftKings cache is missing.
    Other books are never shown as DraftKings prices. An empty frame plus
    LAST_WARNING is returned when no DraftKings lines are available.
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

    # A filled DraftKings cache is enough. Do not spend another credit.
    # A missing cache is allowed one pull even when the consensus file already
    # exists, because that file does not contain per-book prices.
    cached = _read_cache(DK_CACHE_PATH)
    if cached is not None and bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        logger.info("ODDS_FETCH_ONCE: using cached DraftKings odds (%s rows)", len(cached))
        return cached

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

    odds_df = rows_from_odds_api_events(payload)
    if odds_df.empty:
        logger.warning("No DraftKings UFC h2h odds returned")
        return _fallback_odds(force_refresh=False, reason="no DraftKings lines")

    odds_df.to_csv(DK_CACHE_PATH, index=False)
    logger.info("Fetched %s DraftKings odds lines", len(odds_df))
    return odds_df
