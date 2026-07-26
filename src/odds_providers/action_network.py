"""Action Network UFC odds scrape (free backup, fail-soft).

Public scoreboard: https://api.actionnetwork.com/web/v1/scoreboard/ufc
Never raises into callers — returns empty DataFrame on any failure.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config
from src.odds_providers.prop_odds_common import american_to_decimal
from src.predictor import _implied_probs

logger = logging.getLogger(__name__)

ACTION_NETWORK_CACHE_PATH = config.CACHE_DIR / "action_network_ufc_odds.csv"
DEFAULT_URL = "https://api.actionnetwork.com/web/v1/scoreboard/ufc"


def _cache_fresh(path: Path | None = None) -> bool:
    p = Path(path or ACTION_NETWORK_CACHE_PATH)
    ttl_h = float(getattr(config, "ODDS_CACHE_TTL_HOURS", 0) or 0)
    if not p.is_file() or ttl_h <= 0:
        return False
    age_h = (time.time() - p.stat().st_mtime) / 3600.0
    return age_h < ttl_h


def _competitor_name(comp: dict[str, Any], *, side: str) -> str:
    for row in comp.get("competitors") or []:
        if str(row.get("side") or "").lower() != side:
            continue
        player = row.get("player") or {}
        name = str(player.get("full_name") or row.get("name") or "").strip()
        if name:
            return name
    return ""


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v and v > 1.0)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _parse_scoreboard(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fight in payload.get("competitions") or []:
        if not isinstance(fight, dict):
            continue
        home = _competitor_name(fight, side="home")
        away = _competitor_name(fight, side="away")
        if not home or not away:
            continue
        home_px: list[float] = []
        away_px: list[float] = []
        for line in fight.get("odds") or []:
            if not isinstance(line, dict):
                continue
            try:
                ml_home = line.get("ml_home")
                ml_away = line.get("ml_away")
                if ml_home is None or ml_away is None:
                    continue
                dh = american_to_decimal(float(ml_home))
                da = american_to_decimal(float(ml_away))
                if dh > 1.0 and da > 1.0:
                    home_px.append(dh)
                    away_px.append(da)
            except (TypeError, ValueError):
                continue
        f1 = _median(home_px)
        f2 = _median(away_px)
        if f1 is None or f2 is None:
            continue
        # Canonical fighter_1 / fighter_2 = home / away (AN sides)
        ip1, ip2 = _implied_probs(f1, f2)
        rows.append(
            {
                "event_id": str(fight.get("id") or ""),
                "commence_time": fight.get("start_time"),
                "fighter_1": home,
                "fighter_2": away,
                "f1_odds": round(f1, 4),
                "f2_odds": round(f2, 4),
                "implied_prob_f1": ip1,
                "implied_prob_f2": ip2,
                "bookmaker": "ActionNetwork",
                "bookmaker_count": max(len(home_px), 1),
                "odds_source": "ActionNetwork",
                "event_label": str((fight.get("meta") or {}).get("title") or ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_action_network_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch UFC moneylines from Action Network scoreboard (fail-soft).

    Returns empty DataFrame when disabled, blocked, or parse yields nothing.
    """
    if not bool(getattr(config, "ACTION_NETWORK_ENABLED", True)):
        logger.debug("Action Network odds disabled")
        return pd.DataFrame()

    cache_path = ACTION_NETWORK_CACHE_PATH
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if not force_refresh and _cache_fresh(cache_path):
        try:
            cached = pd.read_csv(cache_path)
            if not cached.empty:
                logger.info(
                    "Using cached Action Network odds (%s rows, ttl=%sm)",
                    len(cached),
                    getattr(config, "ODDS_CACHE_TTL_MINUTES", 20),
                )
                return cached
        except Exception as exc:
            logger.debug("Action Network cache read failed: %s", exc)

    url = str(
        getattr(config, "ACTION_NETWORK_UFC_URL", "") or DEFAULT_URL
    ).strip() or DEFAULT_URL
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.actionnetwork.com/ufc/odds",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            logger.info("Action Network: unexpected payload type %s", type(payload))
            return pd.DataFrame()
        df = _parse_scoreboard(payload)
        if df.empty:
            logger.info("Action Network: no usable UFC moneylines")
            return pd.DataFrame()
        try:
            df.to_csv(cache_path, index=False)
        except Exception as exc:
            logger.debug("Action Network cache write skipped: %s", exc)
        logger.info("Action Network odds fetched (%s fights)", len(df))
        return df
    except Exception as exc:
        # Fail-soft: never break the odds chain
        logger.info("Action Network odds unavailable (fail-soft): %s", exc)
        return pd.DataFrame()
