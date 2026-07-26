"""The Odds API moneyline + prop markets (free-tier primary source).

``odds_source`` is labeled ``the_odds_api`` for moneylines.
Prop lines use the same label (treated as live by HA gates).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import config
from src.data_loader import ensure_data_dirs
from src.odds_providers.prop_odds_common import empty_prop_odds_df, prop_row
from src.predictor import (
    OddsAPIError,
    _implied_probs,
    _names_match,
    _to_decimal_odds,
    fetch_ufc_odds,
)

logger = logging.getLogger(__name__)

ODDS_SOURCE = "the_odds_api"
BOOK_NAME = "Odds API"
PROP_CACHE_PATH = config.CACHE_DIR / "the_odds_api_prop_odds.csv"
# Marker written after any completed prop API pass (even 0 rows) so ODDS_FETCH_ONCE
# does not re-burn credits when markets are empty for the current slate.
PROP_FETCH_ONCE_MARKER = config.CACHE_DIR / "the_odds_api_prop_odds.once"
LAST_WARNING: str = ""
LAST_ERROR: str = ""


def _cache_fresh(path: Path) -> bool:
    """Reuse Odds API caches; with ODDS_FETCH_ONCE, ignore TTL once file/marker exists."""
    if bool(getattr(config, "ODDS_FETCH_ONCE", True)) and PROP_FETCH_ONCE_MARKER.is_file():
        return True
    if not path.is_file():
        return False
    try:
        # Header-only empty prop CSV is still a valid "already fetched" cache.
        if path.stat().st_size < 16:
            return False
    except OSError:
        return False
    if bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        return True
    ttl_h = float(getattr(config, "ODDS_CACHE_TTL_HOURS", 0) or 0)
    if ttl_h <= 0:
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < ttl_h


def _write_prop_cache(df: pd.DataFrame) -> None:
    """Persist prop rows (or empty header) + once-marker to stop repeat API burns."""
    try:
        PROP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROP_CACHE_PATH, index=False)
        if bool(getattr(config, "ODDS_FETCH_ONCE", True)):
            PROP_FETCH_ONCE_MARKER.write_text(
                f"fetched_at={time.time()}\nrows={len(df)}\n",
                encoding="utf-8",
            )
    except Exception as exc:
        logger.debug("Odds API prop cache write skipped: %s", exc)


def fetch_the_odds_api_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """
    UFC h2h consensus from The Odds API with best-book label.

    Sets ``odds_source=the_odds_api`` and ``bookmaker`` to the best listed book
    (or ``Odds API`` when only consensus is available).
    """
    global LAST_WARNING, LAST_ERROR
    LAST_WARNING = ""
    LAST_ERROR = ""
    from src.odds_providers import odds_api_client as _odds_client
    from src.odds_providers.odds_api_client import (
        LAST_REQUEST_META,
        odds_api_fail_closed_message,
        refresh_odds_api_runtime,
    )

    refresh_odds_api_runtime()
    if not getattr(config, "ODDS_API_KEY", ""):
        LAST_ERROR = odds_api_fail_closed_message(detail="THE_ODDS_API_KEY missing")
        LAST_WARNING = LAST_ERROR
        raise OddsAPIError(LAST_ERROR)

    try:
        df = fetch_ufc_odds(force_refresh=force_refresh)
    except OddsAPIError as exc:
        msg = str(exc)
        if "NO BET" not in msg:
            LAST_ERROR = odds_api_fail_closed_message(
                status_code=LAST_REQUEST_META.get("status_code"),
                error_code=str(LAST_REQUEST_META.get("error_code") or ""),
                detail=msg,
            )
        else:
            LAST_ERROR = msg
        LAST_WARNING = LAST_ERROR
        raise OddsAPIError(LAST_ERROR) from exc

    if df is None or df.empty:
        LAST_ERROR = odds_api_fail_closed_message(detail="Odds API returned no lines")
        LAST_WARNING = LAST_ERROR
        raise OddsAPIError(LAST_ERROR)

    out = df.copy()
    warn = str(getattr(_odds_client, "LAST_FETCH_WARNING", "") or "").strip()
    if not warn:
        try:
            warn = str(getattr(df, "attrs", {}).get("odds_api_warning") or "")
        except Exception:
            warn = ""
    if warn:
        LAST_WARNING = warn
    out["odds_source"] = ODDS_SOURCE
    if "bookmaker" not in out.columns or out["bookmaker"].isna().all():
        out["bookmaker"] = BOOK_NAME
    else:
        out["bookmaker"] = out["bookmaker"].fillna(BOOK_NAME)
    out["odds_source"] = ODDS_SOURCE
    return out


def _pick_best_totals_price(
    outcomes_by_book: list[tuple[str, list[dict[str, Any]]]],
    *,
    side: str,
    point: float = 1.5,
) -> tuple[float, str] | None:
    """Median decimal price across books for Over/Under at point; return (dec, book)."""
    prices: list[tuple[float, str]] = []
    for book_title, outcomes in outcomes_by_book:
        for outcome in outcomes:
            name = str(outcome.get("name") or "").strip().lower()
            if name != side:
                continue
            try:
                pt = float(outcome.get("point"))
                price = outcome.get("price")
            except (TypeError, ValueError):
                continue
            if price is None or abs(pt - point) > 0.01:
                continue
            dec = _to_decimal_odds(float(price), config.ODDS_API_ODDS_FORMAT)
            if dec > 1.0:
                prices.append((dec, book_title))
    if not prices:
        return None
    prices.sort(key=lambda x: x[0])
    mid = prices[len(prices) // 2]
    return mid


# Odds API totals point → internal prop_key (+ display selection).
_TOTALS_POINT_MAP: dict[tuple[float, str], tuple[str, str]] = {
    (1.5, "over"): ("over_1_5_rounds", "Over 1.5"),
    (1.5, "under"): ("under_1_5_rounds", "Under 1.5"),
}


def _collect_totals_points(
    outcomes_by_book: list[tuple[str, list[dict[str, Any]]]],
) -> set[float]:
    points: set[float] = set()
    for _, outcomes in outcomes_by_book:
        for outcome in outcomes:
            try:
                points.add(float(outcome.get("point")))
            except (TypeError, ValueError):
                continue
    return points


def fetch_the_odds_api_prop_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch UFC totals from The Odds API and map known round lines to prop keys.

    Currently maps Over/Under 1.5 → ``over_1_5_rounds`` / ``under_1_5_rounds``.
    Method props are not offered by The Odds API for MMA; those still come from
    optional book scrapers when enabled.
    """
    global LAST_WARNING, LAST_ERROR
    LAST_WARNING = ""
    LAST_ERROR = ""
    ensure_data_dirs()
    from src.odds_providers.odds_api_client import (
        FORCED_SPORT,
        LAST_REQUEST_META,
        odds_api_fail_closed_message,
        odds_api_get,
        refresh_odds_api_runtime,
    )

    refresh_odds_api_runtime()
    if not bool(getattr(config, "ENABLE_PROPS", False)):
        return empty_prop_odds_df()
    if not getattr(config, "ODDS_API_KEY", ""):
        LAST_WARNING = "Odds API props unavailable — set THE_ODDS_API_KEY"
        LAST_ERROR = odds_api_fail_closed_message(detail="THE_ODDS_API_KEY missing")
        logger.warning(LAST_WARNING)
        return empty_prop_odds_df()

    # ODDS_FETCH_ONCE: never re-hit props endpoints while a prior download exists
    # (including empty markets — otherwise Soft Update burns 7+ credits every click).
    if force_refresh and bool(getattr(config, "ODDS_FETCH_ONCE", True)) and _cache_fresh(PROP_CACHE_PATH):
        logger.info("ODDS_FETCH_ONCE: reusing cached prop odds (force_refresh ignored)")
        force_refresh = False

    if not force_refresh and _cache_fresh(PROP_CACHE_PATH):
        try:
            if PROP_CACHE_PATH.is_file():
                cached = pd.read_csv(PROP_CACHE_PATH)
                logger.info(
                    "Using cached Odds API prop odds (%s rows, once=%s ttl=%sm)",
                    len(cached),
                    bool(getattr(config, "ODDS_FETCH_ONCE", True)),
                    getattr(config, "ODDS_CACHE_TTL_MINUTES", 20),
                )
                return cached if not cached.empty else empty_prop_odds_df()
            # Marker-only (prior empty fetch): do not call API again
            logger.info("ODDS_FETCH_ONCE: prop once-marker present — skipping live prop pull")
            return empty_prop_odds_df()
        except Exception as exc:
            logger.debug("Odds API prop cache read failed: %s", exc)

    try:
        events_resp = odds_api_get(
            f"/sports/{FORCED_SPORT}/events",
            include_odds_params=False,
        )
    except Exception as exc:
        LAST_ERROR = odds_api_fail_closed_message(detail=str(exc))
        LAST_WARNING = LAST_ERROR
        logger.warning("Odds API prop events failed: %s", exc)
        return empty_prop_odds_df()

    if events_resp.status_code == 401:
        LAST_ERROR = odds_api_fail_closed_message(
            status_code=401,
            error_code=str(LAST_REQUEST_META.get("error_code") or ""),
        )
        LAST_WARNING = LAST_ERROR
        logger.warning(LAST_WARNING)
        return empty_prop_odds_df()
    try:
        events_resp.raise_for_status()
        events = events_resp.json()
    except requests.RequestException as exc:
        LAST_ERROR = odds_api_fail_closed_message(detail=str(exc))
        LAST_WARNING = LAST_ERROR
        logger.warning("Odds API prop events failed: %s", exc)
        return empty_prop_odds_df()
    except json.JSONDecodeError:
        LAST_WARNING = "Odds API prop events returned invalid JSON"
        return empty_prop_odds_df()

    if not isinstance(events, list) or not events:
        LAST_WARNING = "Odds API returned no events for prop markets"
        empty = empty_prop_odds_df()
        _write_prop_cache(empty)
        return empty

    markets = config.ODDS_API_PROP_MARKETS or "totals"
    rows: list[dict[str, Any]] = []

    # Free-tier guard: at most N event prop pulls, prefer soonest events.
    max_events = int(getattr(config, "ODDS_API_PROP_MAX_EVENTS", 6) or 6)
    max_events = max(1, min(max_events, 12))
    events_sorted = sorted(
        [e for e in events if isinstance(e, dict)],
        key=lambda e: str(e.get("commence_time") or ""),
    )[:max_events]
    logger.info(
        "Odds API props: fetching markets=%s for %s/%s events (cap=%s)",
        markets,
        len(events_sorted),
        len(events),
        max_events,
    )

    for event in events_sorted:
        event_id = event.get("id")
        home = str(event.get("home_team", "")).strip()
        away = str(event.get("away_team", "")).strip()
        if not event_id or not home or not away:
            continue

        try:
            resp = odds_api_get(
                f"/sports/{FORCED_SPORT}/events/{event_id}/odds",
                include_odds_params=False,
                extra_params={
                    "regions": config.ODDS_API_REGIONS,
                    "markets": markets,
                    "oddsFormat": config.ODDS_API_ODDS_FORMAT,
                },
            )
            if resp.status_code == 401:
                LAST_ERROR = odds_api_fail_closed_message(
                    status_code=401,
                    error_code=str(LAST_REQUEST_META.get("error_code") or ""),
                )
                LAST_WARNING = LAST_ERROR
                return empty_prop_odds_df()
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.debug("Odds API props skipped for %s vs %s: %s", home, away, exc)
            continue

        outcomes_by_book: list[tuple[str, list[dict[str, Any]]]] = []
        for book in payload.get("bookmakers") or []:
            title = str(book.get("title") or book.get("key") or BOOK_NAME).strip()
            for market in book.get("markets") or []:
                if str(market.get("key") or "") != "totals":
                    continue
                outcomes_by_book.append((title, list(market.get("outcomes") or [])))

        if not outcomes_by_book:
            continue

        for point in sorted(_collect_totals_points(outcomes_by_book)):
            for side in ("over", "under"):
                mapped = _TOTALS_POINT_MAP.get((round(point, 1), side))
                if mapped is None and abs(point - 1.5) < 0.01:
                    mapped = _TOTALS_POINT_MAP.get((1.5, side))
                if mapped is None:
                    continue
                prop_key, selection = mapped
                priced = _pick_best_totals_price(
                    outcomes_by_book, side=side, point=point
                )
                if priced is None:
                    continue
                dec, book_title = priced
                rows.append(
                    prop_row(
                        fighter_1=home,
                        fighter_2=away,
                        prop_key=prop_key,
                        selection=selection,
                        decimal_odds=dec,
                        bookmaker=book_title or BOOK_NAME,
                        odds_source=ODDS_SOURCE,
                        market_key="totals",
                        point=float(point),
                    )
                )
                # Alias: Under 1.5 == Round 1 finish for scrapers / older keys
                if prop_key == "under_1_5_rounds":
                    rows.append(
                        prop_row(
                            fighter_1=home,
                            fighter_2=away,
                            prop_key="round_1_finish",
                            selection="Under 1.5 / R1 Finish",
                            decimal_odds=dec,
                            bookmaker=book_title or BOOK_NAME,
                            odds_source=ODDS_SOURCE,
                            market_key="totals",
                            point=float(point),
                        )
                    )

    if not rows:
        LAST_WARNING = "Odds API returned no totals prop markets for mapped round lines"
        logger.info(LAST_WARNING)
        empty = empty_prop_odds_df()
        _write_prop_cache(empty)
        return empty

    df = pd.DataFrame(rows)
    _write_prop_cache(df)
    logger.info("Fetched %s Odds API prop lines", len(df))
    return df
