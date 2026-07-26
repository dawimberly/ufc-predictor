"""BetNow.eu UFC odds scraper (requests + BeautifulSoup, optional Selenium fallback)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pandas as pd
import requests

import config
from src.data_loader import ensure_data_dirs
from src.odds_providers.prop_odds_common import (
    american_to_decimal,
    empty_prop_odds_df,
    parse_american_odds,
    prop_row,
)
from src.predictor import OddsAPIError, _implied_probs, _names_match

logger = logging.getLogger(__name__)

BETNOW_CACHE_PATH = config.CACHE_DIR / "betnow_odds.csv"
BETNOW_PROP_CACHE_PATH = config.CACHE_DIR / "betnow_prop_odds.csv"
BETNOW_UFC_URL = config.BETNOW_PROPS_URL
BETNOW_HOME_URL = "https://www.betnow.eu/"
BETNOW_URLS = [
    BETNOW_UFC_URL,
    "https://www.betnow.eu/sportsbook-info/fighting/professional-mma/",
]
_AMERICAN_RE = re.compile(r"(?<![0-9T])([+-]\d{2,4})(?![0-9])")
_ROTATION_RE = re.compile(r"^\d{5}$")
_SESSION_IN_URL_RE = re.compile(
    r"[?&](?:session|sid|PHPSESSID|token)=([A-Za-z0-9_\-.%=]+)",
    re.I,
)
_SESSION_IN_HTML_RE = re.compile(
    r"(?:session|sid|PHPSESSID)\s*[:=]\s*['\"]?([A-Za-z0-9_\-.]{8,})",
    re.I,
)

_PROP_LABEL_MAP: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"goes?\s+to\s+decision|fight\s+goes\s+to\s+decision", re.I), "goes_to_decision", "Yes"),
    (re.compile(r"inside\s+distance|does\s+not\s+go\s+to\s+decision|finish", re.I), "finish", "Yes"),
    (re.compile(r"\bko\b|ko/tko|wins?\s+by\s+ko", re.I), "ko_tko", "Yes"),
    (re.compile(r"submission|wins?\s+by\s+sub", re.I), "submission", "Yes"),
    (re.compile(r"round\s*1\s+finish|ends?\s+in\s+round\s*1", re.I), "round_1_finish", "Yes"),
    (re.compile(r"over\s*1\.?5|over\s*1\s*½", re.I), "over_1_5_rounds", "Over"),
    (re.compile(r"under\s*1\.?5|under\s*1\s*½", re.I), "round_1_finish", "Under 1.5"),
    (re.compile(r"wins?\s+by\s+ko.*", re.I), "fighter_ko", "Yes"),
    (re.compile(r"wins?\s+by\s+sub.*", re.I), "fighter_sub", "Yes"),
]

# Resolved once per scrape pass (env → page extract).
_ACTIVE_SESSION_TOKEN: str | None = None
_COOKIE_PLACEHOLDERS = frozenset({"", "your_cookie", "cookie", "changeme", "none", "null"})


def _cache_fresh(path: Any = None) -> bool:
    from pathlib import Path

    p = Path(path or BETNOW_CACHE_PATH)
    if not p.is_file() or config.ODDS_CACHE_TTL_HOURS <= 0:
        return False
    age_h = (time.time() - p.stat().st_mtime) / 3600
    return age_h < config.ODDS_CACHE_TTL_HOURS


def _env_session_token() -> str:
    """Prefer BETNOW_SESSION; also accept BETNOW_SESSION_TOKEN / SESSION_TOKEN."""
    return (
        (getattr(config, "BETNOW_SESSION_TOKEN", "") or "").strip()
        or (getattr(config, "SESSION_TOKEN", "") or "").strip()
    )


def _env_cookie() -> str:
    """Return BETNOW_COOKIE when set to a real value (not a placeholder)."""
    cookie = (getattr(config, "BETNOW_COOKIE", "") or "").strip()
    if not cookie or cookie.lower() in _COOKIE_PLACEHOLDERS:
        return ""
    return cookie


def _mask_token(token: str) -> str:
    t = str(token or "").strip()
    if not t:
        return "(empty)"
    if len(t) <= 12:
        return t
    return f"{t[:8]}...{t[-6:]} (len={len(t)})"


def _mask_cookie(cookie: str) -> str:
    c = str(cookie or "").strip()
    if not c:
        return "(empty / not set)"
    # Show first name=value pair prefix only — enough to confirm the right jar.
    first = c.split(";", 1)[0].strip()
    if len(first) > 48:
        first = first[:48] + "..."
    return f"{first}... ({len(c)} chars, {c.count(';') + 1} parts)"


def _append_session_query(url: str, token: str) -> str:
    """Append or replace ?session=TOKEN on a BetNow odds/prop URL."""
    token = str(token or "").strip()
    if not token or not url:
        return url
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs["session"] = [token]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _extract_session_token_from_text(text: str) -> str:
    if not text:
        return ""
    m = _SESSION_IN_URL_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _SESSION_IN_HTML_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_session_token_from_response(resp: requests.Response) -> str:
    """Pull session token from redirect URL, Set-Cookie, or HTML body."""
    candidates = [resp.url or "", str(resp.headers.get("Location") or "")]
    for cookie in resp.cookies:
        name = str(cookie.name or "").lower()
        if name in {"session", "sid", "phpsessid", "token", "betnow_session"}:
            return str(cookie.value or "").strip()
        candidates.append(f"{cookie.name}={cookie.value}")
    for c in candidates:
        tok = _extract_session_token_from_text(c)
        if tok:
            return tok
    return _extract_session_token_from_text(resp.text or "")


def _discover_session_token(headers: dict[str, str]) -> str:
    """Hit BetNow home/login page and try to extract a session token."""
    for url in (BETNOW_HOME_URL, BETNOW_UFC_URL):
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=config.REQUEST_TIMEOUT_SEC,
                allow_redirects=True,
            )
            tok = _extract_session_token_from_response(resp)
            if tok:
                logger.info(
                    "BetNow session token extracted from %s: %s",
                    url,
                    tok,
                )
                return tok
        except requests.RequestException as exc:
            logger.debug("BetNow session discovery failed for %s: %s", url, exc)
    return ""


def resolve_betnow_session_token(*, force_discover: bool = False) -> str:
    """
    Resolve BetNow URL session token from .env
    (BETNOW_SESSION / BETNOW_SESSION_TOKEN / SESSION_TOKEN).

    Cookie is optional — session alone is enough to build ?session= URLs.
    """
    global _ACTIVE_SESSION_TOKEN
    if _ACTIVE_SESSION_TOKEN and not force_discover:
        return _ACTIVE_SESSION_TOKEN

    token = _env_session_token()
    source = "env(BETNOW_SESSION)"
    if not token:
        token = _discover_session_token(_request_headers(log_auth=False))
        source = "page"
    _ACTIVE_SESSION_TOKEN = token
    if token:
        logger.info("BetNow session token in use (%s): %s", source, token)
    else:
        logger.info(
            "BetNow session token not set — will try public pages "
            "(set BETNOW_SESSION in .env; cookie optional)"
        )
    return token


def _log_auth_materials(
    session_token: str,
    cookie: str,
    *,
    url: str | None = None,
    cookie_header: str | None = None,
) -> None:
    """INFO log of exactly what auth material will be sent on the next request."""
    sent_cookie = cookie_header if cookie_header is not None else cookie
    if not sent_cookie:
        cookie_desc = "(none — session query only)" if session_token else "(empty / not set)"
    elif sent_cookie.startswith("session=") and not cookie:
        cookie_desc = f"Cookie: session=<BETNOW_SESSION> ({_mask_token(session_token)})"
    else:
        cookie_desc = f"Cookie: {_mask_cookie(sent_cookie)}"
    logger.info(
        "BetNow auth -> session_token=%s | %s%s",
        session_token if session_token else "(empty)",
        cookie_desc,
        f" | url={url}" if url else "",
    )


def _request_headers(*, log_auth: bool = False, session_token: str | None = None) -> dict[str, str]:
    """
    Build request headers. Cookie is optional — BETNOW_SESSION works without it.

    When cookie is missing but a session token exists, also send a lightweight
    ``session=<token>`` Cookie so sites that expect either form still work.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": BETNOW_HOME_URL,
    }
    cookie = _env_cookie()
    token = (session_token if session_token is not None else _env_session_token()).strip()
    if cookie:
        headers["Cookie"] = cookie
    elif token:
        # Session-only mode: no BETNOW_COOKIE required.
        headers["Cookie"] = f"session={token}"
    if log_auth:
        _log_auth_materials(token, cookie, cookie_header=headers.get("Cookie", ""))
    return headers


def _betnow_odds_urls(session_token: str) -> list[str]:
    """Odds/prop URLs with optional ?session=TOKEN appended."""
    return [_append_session_query(u, session_token) for u in BETNOW_URLS]


def _looks_like_login_wall(html: str) -> bool:
    text = (html or "").lower()
    if not text:
        return True
    # Real odds pages use id="odds"; CSS "#odds" never appears in raw HTML.
    if 'id="odds"' in text or "id='odds'" in text:
        return False
    markers = (
        "please log in",
        "please login",
        "sign in to view",
        "loginurl",
        "member login",
        "create an account",
    )
    return any(m in text for m in markers)


def _parse_page_payload(html: str, source_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    odds_root = soup.select_one("#odds")
    fights = _iter_fight_blocks(odds_root)
    if not fights:
        return [], []
    rows = _parse_moneyline_rows(fights)
    for row in rows:
        row["source_url"] = source_url
    props = _parse_totals_props(fights)
    props.extend(_parse_prop_sections(html, fights))
    return rows, props


def _fetch_and_parse(
    url: str,
    headers: dict[str, str],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """GET one URL and parse moneyline/props. Returns (rows, props, url)."""
    try:
        logger.info("BetNow fetching (%s): %s", label, url)
        resp = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("BetNow fetch failed (%s) %s: %s", label, url, exc)
        return [], [], url

    if _looks_like_login_wall(resp.text):
        logger.info("BetNow (%s) looks like a login wall — no public odds on %s", label, url)
        return [], [], url

    rows, props = _parse_page_payload(resp.text, url)
    if rows or props:
        logger.info(
            "BetNow (%s) parsed %d moneyline / %d prop rows from %s",
            label,
            len(rows),
            len(props),
            url,
        )
    else:
        logger.info("BetNow (%s) returned HTML but no parseable odds on %s", label, url)
    return rows, props, url


def _cell_odds_text(cell) -> str:
    """Extract visible odds text from a table cell, skipping login placeholders."""
    if cell is None:
        return ""
    classes = " ".join(cell.get("class", []))
    if "loginurl" in classes:
        return ""
    text = cell.get_text(" ", strip=True)
    if text in {"-", "—", ""}:
        return ""
    return text


def _decimal_from_cell_text(text: str) -> float | None:
    text = text.strip()
    if not text or text in {"-", "—"}:
        return None
    american = parse_american_odds(text)
    if american is not None:
        dec = american_to_decimal(american)
        return dec if dec > 1 else None
    try:
        val = float(text.replace(",", "."))
        return val if val > 1 else None
    except ValueError:
        return None


def _fighter_from_team_span(span_text: str) -> tuple[str, str]:
    """Return (rotation, fighter_name) from '24013 Michael Chandler'."""
    text = " ".join(span_text.split())
    parts = text.split()
    if parts and _ROTATION_RE.match(parts[0]):
        return parts[0], " ".join(parts[1:]).strip()
    return "", text.strip()


def _parse_prop_label(label: str, fighter_name: str) -> tuple[str, str] | None:
    for pattern, prop_key, selection in _PROP_LABEL_MAP:
        if pattern.search(label):
            if prop_key in ("fighter_ko", "fighter_sub") and fighter_name:
                return prop_key, f"{fighter_name} {selection}"
            return prop_key, selection
    return None


def _iter_fight_blocks(odds_root) -> list[dict[str, Any]]:
    """Parse #odds UFC fight blocks from sportsbook-info HTML."""
    fights: list[dict[str, Any]] = []
    if odds_root is None:
        return fights

    current: dict[str, Any] | None = None
    for child in odds_root.children:
        if getattr(child, "name", None) != "div":
            continue
        div_id = child.get("id") or ""
        classes = child.get("class") or []

        if div_id.startswith("game"):
            if current and current.get("fighters"):
                fights.append(current)
            current = {
                "event_title": child.get_text(" ", strip=True),
                "fighters": [],
                "props": [],
            }
            continue

        if "odd-info-teams" not in classes or current is None:
            continue

        cols = child.find_all("div", recursive=False)
        if len(cols) < 2:
            continue
        team_span = cols[0].find("span", class_="team-name")
        if team_span is None:
            continue
        rotation, fighter = _fighter_from_team_span(team_span.get_text(" ", strip=True))
        if not fighter:
            continue

        spread_txt = _cell_odds_text(cols[1]) if len(cols) > 1 else ""
        total_txt = _cell_odds_text(cols[2]) if len(cols) > 2 else ""
        ml_txt = _cell_odds_text(cols[3]) if len(cols) > 3 else ""

        current["fighters"].append(
            {
                "rotation": rotation,
                "name": fighter,
                "spread": spread_txt,
                "total": total_txt,
                "moneyline": ml_txt,
            }
        )

    if current and current.get("fighters"):
        fights.append(current)

    return fights


def _fight_pair(fight: dict[str, Any]) -> tuple[str, str, str, str]:
    fighters = fight.get("fighters") or []
    if len(fighters) < 2:
        return "", "", "", ""
    f1 = fighters[0]
    f2 = fighters[1]
    return (
        str(f1.get("name", "")).strip(),
        str(f2.get("name", "")).strip(),
        str(f1.get("rotation", "")).strip(),
        str(f2.get("rotation", "")).strip(),
    )


def _parse_moneyline_rows(fights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fight in fights:
        f1_name, f2_name, _, _ = _fight_pair(fight)
        if not f1_name or not f2_name:
            continue
        fighters = fight["fighters"]
        o1 = _decimal_from_cell_text(fighters[0].get("moneyline", ""))
        o2 = _decimal_from_cell_text(fighters[1].get("moneyline", ""))
        if not o1 or not o2:
            continue
        imp1, imp2 = _implied_probs(o1, o2)
        rows.append(
            {
                "fighter_1": f1_name,
                "fighter_2": f2_name,
                "f1_odds": round(o1, 3),
                "f2_odds": round(o2, 3),
                "implied_prob_f1": imp1,
                "implied_prob_f2": imp2,
                "bookmaker": "BetNow.eu",
                "bookmaker_count": 1,
                "source_url": BETNOW_UFC_URL,
            }
        )
    return rows


def _parse_totals_props(fights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse fight totals column (e.g. O1.5 / U1.5) when BetNow exposes text odds."""
    props: list[dict[str, Any]] = []
    over_re = re.compile(r"o(?:ver)?\s*1\.?5", re.I)
    under_re = re.compile(r"u(?:nder)?\s*1\.?5", re.I)

    for fight in fights:
        f1_name, f2_name, rot1, rot2 = _fight_pair(fight)
        if not f1_name or not f2_name:
            continue
        rotation = rot1 or rot2
        for fighter in fight.get("fighters", []):
            total_txt = str(fighter.get("total", "")).strip()
            if not total_txt:
                continue
            # Combined cell like "O1.5 -110" or separate over/under tokens
            american = parse_american_odds(total_txt)
            if american is None:
                continue
            decimal = american_to_decimal(american)
            if decimal <= 1:
                continue
            if over_re.search(total_txt):
                props.append(
                    prop_row(
                        fighter_1=f1_name,
                        fighter_2=f2_name,
                        prop_key="over_1_5_rounds",
                        selection="Over 1.5",
                        decimal_odds=decimal,
                        bookmaker="BetNow.eu",
                        odds_source="live",
                        market_key="totals",
                        point=1.5,
                        rotation=rotation,
                        american_odds=american,
                    )
                )
            elif under_re.search(total_txt):
                props.append(
                    prop_row(
                        fighter_1=f1_name,
                        fighter_2=f2_name,
                        prop_key="round_1_finish",
                        selection="Under 1.5",
                        decimal_odds=decimal,
                        bookmaker="BetNow.eu",
                        odds_source="live",
                        market_key="totals",
                        point=1.5,
                        rotation=rotation,
                        american_odds=american,
                    )
                )
    return props


def _parse_prop_sections(html_text: str, fights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Parse BetNow prop rows when present in HTML (authenticated sessions).

    Looks for label + american odds patterns near known fight names.
    """
    props: list[dict[str, Any]] = []
    for fight in fights:
        f1_name, f2_name, rot1, _ = _fight_pair(fight)
        if not f1_name or not f2_name:
            continue
        rotation = rot1
        for fighter in fight.get("fighters", []):
            name = str(fighter.get("name", ""))
            for field in ("spread", "total", "moneyline"):
                txt = str(fighter.get(field, ""))
                for label, prop_key, selection in _PROP_LABEL_MAP:
                    if label.search(txt):
                        american = parse_american_odds(txt)
                        if american is None:
                            continue
                        decimal = american_to_decimal(american)
                        if decimal <= 1:
                            continue
                        props.append(
                            prop_row(
                                fighter_1=f1_name,
                                fighter_2=f2_name,
                                prop_key=prop_key,
                                selection=selection,
                                decimal_odds=decimal,
                                bookmaker="BetNow.eu",
                                odds_source="live",
                                market_key="prop",
                                rotation=rotation,
                                american_odds=american,
                            )
                        )

        # Global search for fighter-specific method props in page text chunks
        chunk_pat = re.compile(
            rf"{re.escape(f1_name)}.+?({_AMERICAN_RE.pattern})|"
            rf"{re.escape(f2_name)}.+?({_AMERICAN_RE.pattern})",
            re.I | re.S,
        )
        for m in chunk_pat.finditer(html_text):
            snippet = m.group(0)[:240]
            american = parse_american_odds(snippet)
            if american is None:
                continue
            mapped = None
            for label, prop_key, selection in _PROP_LABEL_MAP:
                if label.search(snippet):
                    mapped = (prop_key, selection)
                    break
            if not mapped:
                continue
            prop_key, selection = mapped
            decimal = american_to_decimal(american)
            if decimal <= 1:
                continue
            props.append(
                prop_row(
                    fighter_1=f1_name,
                    fighter_2=f2_name,
                    prop_key=prop_key,
                    selection=selection,
                    decimal_odds=decimal,
                    bookmaker="BetNow.eu",
                    odds_source="live",
                    market_key="prop",
                    rotation=rotation,
                    american_odds=american,
                )
            )
    return props


def _scrape_sportsbook_info() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """
    Scrape BetNow moneyline + props.

    Order:
      1. Session URLs (?session=BETNOW_SESSION) — cookie optional / empty OK
      2. Public URLs without session if auth returns nothing / login wall
    """
    session_token = resolve_betnow_session_token()
    cookie = _env_cookie()
    headers = _request_headers(log_auth=False, session_token=session_token)
    last_url = BETNOW_UFC_URL

    # --- Pass 1: session-authenticated URLs (works with session alone) ---
    if session_token:
        for url in _betnow_odds_urls(session_token):
            _log_auth_materials(
                session_token,
                cookie,
                url=url,
                cookie_header=headers.get("Cookie", ""),
            )
            rows, props, last_url = _fetch_and_parse(url, headers, label="session")
            if rows or props:
                return rows, props, last_url
        logger.info(
            "BetNow session auth returned no odds - falling back to public pages "
            "(session=%s)",
            session_token,
        )
    else:
        logger.info("BetNow: no BETNOW_SESSION set — trying public pages only")

    # --- Pass 2: public / guest pages (no ?session=) ---
    public_headers = _request_headers(log_auth=False, session_token="")
    # Do not force synthetic session cookie on public fallback.
    if not cookie:
        public_headers.pop("Cookie", None)
    for url in BETNOW_URLS:
        _log_auth_materials(
            "",
            cookie,
            url=url,
            cookie_header=public_headers.get("Cookie", ""),
        )
        rows, props, last_url = _fetch_and_parse(url, public_headers, label="public")
        if rows or props:
            return rows, props, last_url

    return [], [], last_url


def _scrape_with_selenium() -> list[dict[str, Any]]:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        logger.debug("Selenium not installed — skipping BetNow browser fallback.")
        return []

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    rows: list[dict[str, Any]] = []
    driver = None
    session_token = resolve_betnow_session_token()
    try:
        driver = webdriver.Chrome(options=options)
        urls: list[str] = []
        if session_token:
            urls.extend(_betnow_odds_urls(session_token))
        urls.extend(BETNOW_URLS)  # public fallback after session
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            label = "session" if "session=" in url else "public"
            logger.info("BetNow Selenium fetching (%s): %s", label, url)
            driver.get(url)
            time.sleep(4)
            parsed, _props = _parse_page_payload(driver.page_source, url)
            if parsed:
                rows.extend(parsed)
                break
    except Exception as exc:
        logger.warning("BetNow Selenium fallback failed: %s", exc)
    finally:
        if driver is not None:
            driver.quit()
    return rows


def fetch_betnow_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """Scrape BetNow.eu UFC moneyline lines; cache to data/cache/betnow_odds.csv."""
    ensure_data_dirs()
    if not force_refresh and _cache_fresh(BETNOW_CACHE_PATH):
        cached = pd.read_csv(BETNOW_CACHE_PATH)
        if not cached.empty:
            logger.info("Using cached BetNow odds (%s rows)", len(cached))
            return cached

    global _ACTIVE_SESSION_TOKEN
    _ACTIVE_SESSION_TOKEN = None  # re-resolve on live scrape
    rows, _, source_url = _scrape_sportsbook_info()
    if not rows:
        logger.info("BetNow HTML scrape empty — trying Selenium fallback")
        rows = _scrape_with_selenium()

    if not rows:
        raise OddsAPIError(
            "Could not scrape BetNow.eu UFC odds (session + public pages empty). "
            "Set BETNOW_SESSION in .env (cookie optional). Dashboard will fall back "
            "to The Odds API / other books."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["fighter_1", "fighter_2"], keep="first")
    if "source_url" not in df.columns:
        df["source_url"] = source_url
    df.to_csv(BETNOW_CACHE_PATH, index=False)
    logger.info("Scraped %s BetNow moneyline rows (source=%s)", len(df), source_url)
    return df


def fetch_betnow_prop_odds(*, force_refresh: bool = False) -> pd.DataFrame:
    """Scrape BetNow.eu UFC prop lines (method, totals, decision) when exposed in HTML."""
    ensure_data_dirs()
    if not config.ENABLE_PROPS:
        return empty_prop_odds_df()

    if not force_refresh and _cache_fresh(BETNOW_PROP_CACHE_PATH):
        cached = pd.read_csv(BETNOW_PROP_CACHE_PATH)
        if not cached.empty:
            logger.info("Using cached BetNow prop odds (%s rows)", len(cached))
            return cached

    global _ACTIVE_SESSION_TOKEN
    _ACTIVE_SESSION_TOKEN = None
    _, props, source_url = _scrape_sportsbook_info()
    df = pd.DataFrame(props)
    if df.empty:
        logger.info(
            "BetNow prop scrape returned no live lines "
            "(session/public pages had no props; source=%s).",
            source_url,
        )
        return empty_prop_odds_df()

    df = df.drop_duplicates(
        subset=["fighter_1", "fighter_2", "prop_key", "selection"],
        keep="first",
    )
    df.to_csv(BETNOW_PROP_CACHE_PATH, index=False)
    logger.info("Scraped %s BetNow prop lines", len(df))
    return df


def match_betnow_row(fighter_1: str, fighter_2: str, odds: pd.DataFrame) -> pd.Series | None:
    """Lookup helper for tests."""
    for _, row in odds.iterrows():
        if _names_match(fighter_1, row["fighter_1"]) and _names_match(fighter_2, row["fighter_2"]):
            return row
        if _names_match(fighter_1, row["fighter_2"]) and _names_match(fighter_2, row["fighter_1"]):
            swapped = row.copy()
            swapped["fighter_1"], swapped["fighter_2"] = row["fighter_2"], row["fighter_1"]
            swapped["f1_odds"], swapped["f2_odds"] = row["f2_odds"], row["f1_odds"]
            swapped["implied_prob_f1"], swapped["implied_prob_f2"] = (
                row["implied_prob_f2"],
                row["implied_prob_f1"],
            )
            return swapped
    return None
