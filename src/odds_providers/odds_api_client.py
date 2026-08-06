"""Shared The Odds API request helpers (dashboard / predictor).

Never logs the full API key — only source path, length, and last4.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

import config

logger = logging.getLogger(__name__)

# Forced free-tier defaults (env may override regions/markets via refresh).
FORCED_SPORT = "mma_mixed_martial_arts"
DEFAULT_REGIONS = "us,eu,uk"
DEFAULT_MARKETS = "h2h"
DEFAULT_ODDS_FORMAT = "decimal"

MSG_MISSING_KEY = "set THE_ODDS_API_KEY in .env"
MSG_QUOTA_EXHAUSTED = (
    "Odds API quota exhausted — using cache if available; replace key or wait for reset"
)
MSG_KEY_REJECTED = "Odds API key rejected — check THE_ODDS_API_KEY"

# Last request diagnostics for UI / tests (no secrets).
LAST_REQUEST_META: dict[str, Any] = {
    "key_source": "",
    "key_loaded": False,
    "key_length": 0,
    "key_last4": "",
    "sport": FORCED_SPORT,
    "regions": DEFAULT_REGIONS,
    "markets": DEFAULT_MARKETS,
    "odds_format": DEFAULT_ODDS_FORMAT,
    "url": "",
    "status_code": None,
    "error_code": "",
    "requests_remaining": None,
    "requests_used": None,
}
# Set when a fetch degraded (e.g. quota → disk cache) so UI can show fail-closed text.
LAST_FETCH_WARNING: str = ""


class OddsApiFetchBlocked(RuntimeError):
    """Raised when ODDS_FETCH_ONCE forbids another live The Odds API call."""


def odds_fetch_once_blocks_live() -> bool:
    """True when ODDS_FETCH_ONCE and a prior odds download already exists on disk."""
    if not bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        return False
    cache_dir = Path(getattr(config, "CACHE_DIR", "") or ".")
    candidates = [
        getattr(config, "ODDS_CACHE_PATH", None),
        cache_dir / "ufc_odds_api.csv",
        cache_dir / "the_odds_api_prop_odds.csv",
        cache_dir / "the_odds_api_prop_odds.once",
        cache_dir / "draftkings_odds.csv",
        cache_dir / "draftkings_prop_odds.csv",
    ]
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        try:
            if not p.is_file():
                continue
            # .once marker is always enough; CSV needs a real payload/header.
            if p.suffix == ".once" or p.stat().st_size >= 16:
                return True
        except OSError:
            continue
    return False


def ensure_live_odds_api_allowed(*, context: str = "") -> None:
    """Hard stop for any live The Odds API HTTP call when fetch-once cache exists."""
    if not odds_fetch_once_blocks_live():
        return
    detail = context or "cached odds already on disk"
    msg = (
        f"ODDS_FETCH_ONCE: blocked live Odds API request ({detail}). "
        "Delete data/cache/ufc_odds_api.csv (and prop/DK caches) to allow a new download."
    )
    logger.warning(msg)
    raise OddsApiFetchBlocked(msg)


def clear_odds_api_fetch_once_caches() -> list[str]:
    """Delete Odds API / DK fetch-once cache files so a new live download is allowed.

    Used when the locked cache is from a prior card (low match rate to the loaded roster).
    """
    cache_dir = Path(getattr(config, "CACHE_DIR", "") or ".")
    candidates = [
        getattr(config, "ODDS_CACHE_PATH", None),
        cache_dir / "ufc_odds_api.csv",
        cache_dir / "the_odds_api_prop_odds.csv",
        cache_dir / "the_odds_api_prop_odds.once",
        cache_dir / "draftkings_odds.csv",
        cache_dir / "draftkings_prop_odds.csv",
    ]
    removed: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        try:
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except OSError as exc:
            logger.debug("Could not remove odds cache %s: %s", p, exc)
    if removed:
        logger.info("Cleared Odds API fetch-once caches: %s", ", ".join(removed))
    return removed

_SESSION: requests.Session | None = None


def normalize_odds_api_key(raw: str | None) -> str:
    return str(raw or "").strip().strip('"').strip("'")


def key_last4(key: str) -> str:
    k = normalize_odds_api_key(key)
    if not k:
        return ""
    return k[-4:] if len(k) >= 4 else k


def clear_odds_api_session() -> None:
    """Drop any cached HTTP session so a new key cannot reuse old auth state."""
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:
            pass
    _SESSION = None


def _fresh_session() -> requests.Session:
    clear_odds_api_session()
    global _SESSION
    _SESSION = requests.Session()
    _SESSION.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "UFC-Predictor/odds-api",
        }
    )
    return _SESSION


def odds_api_env_candidates(root: Path | None = None) -> list[Path]:
    """Project + dist .env paths (primary sources for THE_ODDS_API_KEY)."""
    from src.project_paths import resolve_root

    root = (root or resolve_root()).resolve()
    candidates = [
        root / ".env",
        root / "dist" / ".env",
    ]
    cwd_env = Path.cwd().resolve() / ".env"
    if cwd_env not in candidates:
        candidates.append(cwd_env)
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_odds_api_key_source(root: Path | None = None) -> tuple[str, str]:
    """
    Return (normalized_key, source_path).

    Prefers the newest non-empty THE_ODDS_API_KEY among project ``.env`` and
    ``dist/.env`` (by file mtime). Falls back to process env only if none found.
    """
    from src.project_paths import resolve_root

    root = root or resolve_root()
    found: list[tuple[float, Path, str]] = []
    for env_path in odds_api_env_candidates(root):
        if not env_path.is_file():
            continue
        vals = dotenv_values(env_path)
        candidate = normalize_odds_api_key(
            vals.get("THE_ODDS_API_KEY") or vals.get("ODDS_API_KEY")
        )
        if not candidate:
            continue
        try:
            mtime = float(env_path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        found.append((mtime, env_path.resolve(), candidate))

    if found:
        # Newest mtime wins; stable tie-break by path string
        found.sort(key=lambda t: (t[0], str(t[1])))
        _mtime, path, key = found[-1]
        return key, str(path)

    key = normalize_odds_api_key(
        os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    if key:
        return key, "process environment"
    return "", ""


def refresh_odds_api_runtime(*, root: Path | None = None) -> dict[str, Any]:
    """
    Re-read THE_ODDS_API_KEY + request params into config before any odds call.

    Clears any stale HTTP session so a newly loaded key is used immediately.
    Returns diagnostic meta (never the full key).
    """
    key, source = resolve_odds_api_key_source(root)
    # Force process env to match chosen file so nothing reuses a stale ambient key
    if key:
        os.environ["THE_ODDS_API_KEY"] = key
        os.environ["ODDS_API_KEY"] = key
    else:
        os.environ.pop("THE_ODDS_API_KEY", None)
        # Keep ODDS_API_KEY alias empty too when missing
        if "ODDS_API_KEY" in os.environ:
            os.environ["ODDS_API_KEY"] = ""

    sport = FORCED_SPORT
    regions = DEFAULT_REGIONS
    markets = DEFAULT_MARKETS
    odds_format = DEFAULT_ODDS_FORMAT
    env_markets = (os.getenv("ODDS_API_MARKETS") or "").strip()
    env_format = (os.getenv("ODDS_API_ODDS_FORMAT") or "").strip()
    env_regions = (os.getenv("ODDS_API_REGIONS") or "").strip()
    if env_markets:
        markets = env_markets
    if env_format:
        odds_format = env_format
    if env_regions:
        regions = env_regions
    base = (
        os.getenv("ODDS_API_BASE_URL") or "https://api.the-odds-api.com/v4"
    ).strip().rstrip("/")

    config.ODDS_API_KEY = key
    config.ODDS_API_KEY_SOURCE = source
    config.ODDS_API_SPORT = sport
    config.ODDS_API_REGIONS = regions
    config.ODDS_API_MARKETS = markets
    config.ODDS_API_ODDS_FORMAT = odds_format
    config.ODDS_API_BASE_URL = base

    clear_odds_api_session()

    last4 = key_last4(key)
    meta = {
        "key_source": source or "(none)",
        "key_loaded": bool(key),
        "key_length": len(key),
        "key_last4": last4,
        "sport": sport,
        "regions": regions,
        "markets": markets,
        "odds_format": odds_format,
        "base_url": base,
    }
    LAST_REQUEST_META.update(meta)
    logger.info(
        "Odds API key reload: source=%s len=%s last4=%s sport=%s regions=%s markets=%s",
        meta["key_source"],
        meta["key_length"],
        last4 or "-",
        meta["sport"],
        meta["regions"],
        meta["markets"],
    )
    return meta


def odds_api_fail_closed_message(
    *,
    status_code: int | None = None,
    error_code: str = "",
    detail: str = "",
) -> str:
    """Build fail-closed UI text including key/request diagnostics (never the key)."""
    key_loaded = "yes" if LAST_REQUEST_META.get("key_loaded") else "no"
    key_len = int(LAST_REQUEST_META.get("key_length") or 0)
    last4 = str(LAST_REQUEST_META.get("key_last4") or "") or "-"
    sport = LAST_REQUEST_META.get("sport") or FORCED_SPORT
    regions = LAST_REQUEST_META.get("regions") or DEFAULT_REGIONS
    source = LAST_REQUEST_META.get("key_source") or "(none)"
    diag = (
        f"key loaded={key_loaded}, key length={key_len}, last4={last4}, "
        f"source={source}, sport={sport}, regions={regions}"
    )
    code = str(error_code or "").upper()
    detail_l = str(detail or "").lower()

    if not LAST_REQUEST_META.get("key_loaded") or (
        detail and "missing" in detail_l and "key" in detail_l
    ):
        if not status_code or "missing" in detail_l:
            reason = MSG_MISSING_KEY
            return f"NO BET — no usable odds (fail-closed): {reason} [{diag}]"

    if status_code == 401 and "OUT_OF_USAGE" in code:
        reason = MSG_QUOTA_EXHAUSTED
    elif status_code == 401:
        # Auth rejection only — never call this "unauthorized quota"
        reason = MSG_KEY_REJECTED
    elif "OUT_OF_USAGE" in code or "quota" in detail_l:
        reason = MSG_QUOTA_EXHAUSTED
    elif status_code:
        reason = f"Odds API HTTP {status_code}"
    elif detail:
        # Map legacy missing-key detail to clear copy
        if "missing" in detail_l and "key" in detail_l:
            reason = MSG_MISSING_KEY
        else:
            reason = detail
    else:
        reason = "Odds API request failed"

    extra = ""
    if detail and detail not in reason and "missing" not in detail_l:
        extra = f" ({detail})"
    return f"NO BET — no usable odds (fail-closed): {reason}{extra} [{diag}]"


def odds_api_get(
    path: str,
    *,
    extra_params: dict[str, Any] | None = None,
    include_odds_params: bool = True,
    timeout: float | None = None,
) -> requests.Response:
    """
    Authenticated GET against The Odds API with a fresh session + diagnostics.

    ``path`` is relative (e.g. ``/sports/mma_mixed_martial_arts/odds``) or absolute.
    When ``include_odds_params`` is True, sends regions/markets/oddsFormat defaults.
    """
    refresh_odds_api_runtime()
    key = normalize_odds_api_key(getattr(config, "ODDS_API_KEY", ""))
    if not key:
        raise RuntimeError(odds_api_fail_closed_message(detail="THE_ODDS_API_KEY missing"))

    ensure_live_odds_api_allowed(context=path)

    base = str(getattr(config, "ODDS_API_BASE_URL", "") or "").rstrip("/")
    if path.startswith("http"):
        url = path
    else:
        url = f"{base}{path if path.startswith('/') else '/' + path}"

    params: dict[str, Any] = {"apiKey": key}
    if include_odds_params:
        params.update(
            {
                "regions": getattr(config, "ODDS_API_REGIONS", DEFAULT_REGIONS),
                "markets": getattr(config, "ODDS_API_MARKETS", DEFAULT_MARKETS),
                "oddsFormat": getattr(config, "ODDS_API_ODDS_FORMAT", DEFAULT_ODDS_FORMAT),
            }
        )
    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v is not None})
    params["apiKey"] = key

    LAST_REQUEST_META.update(
        {
            "url": url.split("?")[0],
            "sport": FORCED_SPORT,
            "regions": params.get("regions") or LAST_REQUEST_META.get("regions"),
            "markets": params.get("markets") or LAST_REQUEST_META.get("markets"),
            "odds_format": params.get("oddsFormat") or LAST_REQUEST_META.get("odds_format"),
            "status_code": None,
            "error_code": "",
        }
    )
    logger.info(
        "Odds API GET %s sport=%s regions=%s markets=%s key_len=%s last4=%s source=%s",
        LAST_REQUEST_META["url"],
        LAST_REQUEST_META.get("sport"),
        LAST_REQUEST_META.get("regions"),
        LAST_REQUEST_META.get("markets"),
        LAST_REQUEST_META.get("key_length"),
        LAST_REQUEST_META.get("key_last4") or "-",
        LAST_REQUEST_META.get("key_source"),
    )

    session = _fresh_session()
    resp = session.get(
        url,
        params=params,
        timeout=timeout
        if timeout is not None
        else float(getattr(config, "REQUEST_TIMEOUT_SEC", 30)),
    )
    LAST_REQUEST_META["status_code"] = resp.status_code
    LAST_REQUEST_META["requests_remaining"] = resp.headers.get("x-requests-remaining")
    LAST_REQUEST_META["requests_used"] = resp.headers.get("x-requests-used")
    try:
        body = resp.json()
        if isinstance(body, dict):
            LAST_REQUEST_META["error_code"] = str(body.get("error_code") or "")
    except Exception:
        pass

    if resp.status_code == 401:
        err = str(LAST_REQUEST_META.get("error_code") or "")
        if "OUT_OF_USAGE" in err.upper():
            logger.warning(
                "Odds API quota exhausted: remaining=%s used=%s key_len=%s last4=%s source=%s",
                LAST_REQUEST_META.get("requests_remaining"),
                LAST_REQUEST_META.get("requests_used"),
                LAST_REQUEST_META.get("key_length"),
                LAST_REQUEST_META.get("key_last4") or "-",
                LAST_REQUEST_META.get("key_source"),
            )
        else:
            logger.warning(
                "Odds API key rejected (401): error_code=%s key_len=%s last4=%s source=%s",
                err or "(none)",
                LAST_REQUEST_META.get("key_length"),
                LAST_REQUEST_META.get("key_last4") or "-",
                LAST_REQUEST_META.get("key_source"),
            )
    return resp
