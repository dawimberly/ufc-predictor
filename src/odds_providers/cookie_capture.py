"""
Interactive book login + cookie capture for BetNow / MyBookie.

Opens a headed browser (Playwright preferred, else Selenium). User logs in
manually; we detect session cookies, write them to .env, and reload config.
"""

from __future__ import annotations

import logging
import os
import re
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import config

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

DEFAULT_TIMEOUT_SEC = int(os.getenv("COOKIE_CAPTURE_TIMEOUT_SEC", "180"))
POLL_INTERVAL_SEC = 2.0

# Only these keys are ever written to .env — never username/password.
_ALLOWED_ENV_KEYS = frozenset(
    {
        "BETNOW_COOKIE",
        "BETNOW_SESSION",
        "BETNOW_SESSION_TOKEN",
        "SESSION_TOKEN",
        "MYBOOKIE_COOKIE",
    }
)

_BETNOW_COOKIE_NAMES = frozenset(
    {
        "session",
        "sid",
        "phpsessid",
        "token",
        "betnow_session",
        "betnowsession",
        "auth",
        "jwt",
    }
)
_MYBOOKIE_COOKIE_NAMES = frozenset(
    {
        "session",
        "sid",
        "phpsessid",
        "asp.net_sessionid",
        ".aspxauth",
        "auth",
        "jwt",
        "token",
        "mybookie",
        "mb_session",
        "customer",
        "loggedin",
    }
)
_LOGIN_URL_HINTS = ("login", "signin", "sign-in", "log-in", "auth", "account/login")


@dataclass
class BookCaptureSpec:
    key: str
    label: str
    start_url: str
    cookie_env: str
    session_env: str | None = None
    auth_cookie_names: frozenset[str] = field(default_factory=frozenset)
    success_path_hints: tuple[str, ...] = ()


@dataclass
class CaptureResult:
    book: str
    ok: bool
    status: str  # captured | skipped | timeout | error | already_ok
    cookie: str = ""
    session: str = ""
    message: str = ""
    backend: str = ""


def _progress(cb: ProgressFn | None, msg: str) -> None:
    logger.info(msg)
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def _cookie_header_from_list(cookies: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for c in cookies:
        name = str(c.get("name") or "").strip()
        value = str(c.get("value") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parts.append(f"{name}={value}")
    return "; ".join(parts)


def _cookie_names(cookies: list[dict[str, Any]]) -> set[str]:
    return {str(c.get("name") or "").strip().lower() for c in cookies if c.get("name")}


def _extract_session_from_cookies(cookies: list[dict[str, Any]]) -> str:
    preferred = ("session", "sid", "phpsessid", "token", "betnow_session")
    by_name = {
        str(c.get("name") or "").strip().lower(): str(c.get("value") or "").strip()
        for c in cookies
        if c.get("name")
    }
    for name in preferred:
        if by_name.get(name):
            return by_name[name]
    return ""


def _extract_session_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    for key in ("session", "sid", "token", "PHPSESSID"):
        vals = qs.get(key) or qs.get(key.lower())
        if vals and vals[0]:
            return str(vals[0]).strip()
    m = re.search(r"[?&](?:session|sid|token)=([A-Za-z0-9_\-.%=]+)", url, re.I)
    return m.group(1).strip() if m else ""


def _url_looks_like_login(url: str) -> bool:
    low = (url or "").lower()
    return any(h in low for h in _LOGIN_URL_HINTS)


def _login_detected(
    spec: BookCaptureSpec,
    *,
    url: str,
    cookies: list[dict[str, Any]],
) -> bool:
    """Heuristic: auth cookie present, or session in URL, or sportsbook page with jar."""
    names = _cookie_names(cookies)
    if names & {n.lower() for n in spec.auth_cookie_names}:
        return True
    session = _extract_session_from_url(url) or _extract_session_from_cookies(cookies)
    if session and len(session) >= 8:
        return True
    header = _cookie_header_from_list(cookies)
    if len(header) < 40:
        return False
    if _url_looks_like_login(url):
        return False
    low = (url or "").lower()
    if spec.success_path_hints and any(h in low for h in spec.success_path_hints):
        return True
    # Substantial cookie jar off the login page is enough
    return len(cookies) >= 2 and not _url_looks_like_login(url)


def book_specs(*, include_mybookie: bool | None = None) -> list[BookCaptureSpec]:
    specs = [
        BookCaptureSpec(
            key="betnow",
            label="BetNow",
            start_url=getattr(config, "BETNOW_PROPS_URL", None)
            or "https://www.betnow.eu/sportsbook-info/fighting/ufc/",
            cookie_env="BETNOW_COOKIE",
            session_env="BETNOW_SESSION",
            auth_cookie_names=_BETNOW_COOKIE_NAMES,
            success_path_hints=("sportsbook", "fighting", "ufc", "mma"),
        )
    ]
    mb = config.MYBOOKIE_ENABLED if include_mybookie is None else include_mybookie
    if mb:
        specs.append(
            BookCaptureSpec(
                key="mybookie",
                label="MyBookie",
                start_url=getattr(config, "MYBOOKIE_UFC_URL", None)
                or "https://www.mybookie.ag/sportsbook/ufc/",
                cookie_env="MYBOOKIE_COOKIE",
                session_env=None,
                auth_cookie_names=_MYBOOKIE_COOKIE_NAMES,
                success_path_hints=("sportsbook", "ufc", "mma", "account"),
            )
        )
    return specs


def _env_has_usable_auth(spec: BookCaptureSpec) -> bool:
    placeholders = frozenset({"", "your_cookie", "cookie", "changeme", "none", "null"})
    cookie = (os.getenv(spec.cookie_env) or getattr(config, spec.cookie_env, "") or "").strip()
    if cookie and cookie.lower() not in placeholders and len(cookie) > 20:
        return True
    if spec.session_env:
        session = (
            os.getenv(spec.session_env)
            or os.getenv("BETNOW_SESSION_TOKEN")
            or os.getenv("SESSION_TOKEN")
            or getattr(config, "BETNOW_SESSION_TOKEN", "")
            or ""
        ).strip()
        if session and session.lower() not in placeholders and len(session) >= 8:
            return True
    return False


def _backend_available() -> str:
    """Return 'playwright' | 'selenium' | ''."""
    try:
        import playwright  # noqa: F401

        return "playwright"
    except ImportError:
        pass
    try:
        import selenium  # noqa: F401

        return "selenium"
    except ImportError:
        pass
    return ""


def _persist_auth(spec: BookCaptureSpec, cookie: str, session: str) -> None:
    """
    Save Cookie header (+ optional session token) only.

    Never stores username, password, or form credentials.
    """
    updates: dict[str, str] = {}
    if cookie:
        updates[spec.cookie_env] = cookie
    if session and spec.session_env:
        updates[spec.session_env] = session
    # Hard filter — refuse anything outside the allow-list
    updates = {k: v for k, v in updates.items() if k in _ALLOWED_ENV_KEYS and v}
    if not updates:
        return
    logger.info(
        "Persisting auth for %s (keys=%s; no username/password stored)",
        spec.label,
        sorted(updates.keys()),
    )
    config.upsert_env_vars(updates)
    # Clear BetNow in-memory session so next scrape picks up fresh env
    try:
        from src.odds_providers import betnow_scraper as bn

        bn._ACTIVE_SESSION_TOKEN = None
    except Exception:
        pass


def _capture_with_playwright(
    spec: BookCaptureSpec,
    *,
    timeout_sec: float,
    progress: ProgressFn | None,
) -> CaptureResult:
    from playwright.sync_api import sync_playwright

    _progress(progress, f"Waiting for login… ({spec.label})")
    deadline = time.time() + timeout_sec
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(spec.start_url, wait_until="domcontentloaded", timeout=60_000)
        except Exception as exc:
            logger.warning("%s Playwright navigate: %s", spec.label, exc)
        captured: CaptureResult | None = None
        while time.time() < deadline:
            try:
                url = page.url or ""
                cookies = context.cookies()
            except Exception as exc:
                logger.debug("%s poll failed: %s", spec.label, exc)
                time.sleep(POLL_INTERVAL_SEC)
                continue
            if _login_detected(spec, url=url, cookies=cookies):
                cookie_hdr = _cookie_header_from_list(cookies)
                session = _extract_session_from_url(url) or _extract_session_from_cookies(
                    cookies
                )
                _persist_auth(spec, cookie_hdr, session)
                captured = CaptureResult(
                    book=spec.label,
                    ok=True,
                    status="captured",
                    cookie=cookie_hdr,
                    session=session,
                    message=f"Cookies captured successfully ({spec.label})",
                    backend="playwright",
                )
                _progress(progress, captured.message)
                break
            remaining = int(deadline - time.time())
            if remaining % 15 < POLL_INTERVAL_SEC:
                _progress(
                    progress,
                    f"Waiting for login… ({spec.label}, {remaining}s left)",
                )
            time.sleep(POLL_INTERVAL_SEC)
        try:
            browser.close()
        except Exception:
            pass
    if captured:
        return captured
    return CaptureResult(
        book=spec.label,
        ok=False,
        status="timeout",
        message=(
            f"{spec.label} login timed out after {int(timeout_sec)}s — "
            "continuing in public/guest mode"
        ),
        backend="playwright",
    )


def _capture_with_selenium(
    spec: BookCaptureSpec,
    *,
    timeout_sec: float,
    progress: ProgressFn | None,
) -> CaptureResult:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    _progress(progress, f"Waiting for login… ({spec.label})")
    options = Options()
    # headed — user must log in
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(spec.start_url)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                url = driver.current_url or ""
                raw = driver.get_cookies() or []
                cookies = [{"name": c.get("name"), "value": c.get("value")} for c in raw]
            except Exception as exc:
                logger.debug("%s selenium poll: %s", spec.label, exc)
                time.sleep(POLL_INTERVAL_SEC)
                continue
            if _login_detected(spec, url=url, cookies=cookies):
                cookie_hdr = _cookie_header_from_list(cookies)
                session = _extract_session_from_url(url) or _extract_session_from_cookies(
                    cookies
                )
                _persist_auth(spec, cookie_hdr, session)
                msg = f"Cookies captured successfully ({spec.label})"
                _progress(progress, msg)
                return CaptureResult(
                    book=spec.label,
                    ok=True,
                    status="captured",
                    cookie=cookie_hdr,
                    session=session,
                    message=msg,
                    backend="selenium",
                )
            remaining = int(deadline - time.time())
            if remaining % 15 < POLL_INTERVAL_SEC:
                _progress(
                    progress,
                    f"Waiting for login… ({spec.label}, {remaining}s left)",
                )
            time.sleep(POLL_INTERVAL_SEC)
        return CaptureResult(
            book=spec.label,
            ok=False,
            status="timeout",
            message=(
                f"{spec.label} login timed out after {int(timeout_sec)}s — "
                "continuing in public/guest mode"
            ),
            backend="selenium",
        )
    except Exception as exc:
        logger.warning("%s Selenium capture failed: %s", spec.label, exc)
        return CaptureResult(
            book=spec.label,
            ok=False,
            status="error",
            message=f"{spec.label} browser failed ({exc}) — continuing in public/guest mode",
            backend="selenium",
        )
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def _capture_fallback_open_only(
    spec: BookCaptureSpec,
    *,
    progress: ProgressFn | None,
) -> CaptureResult:
    """No Playwright/Selenium — open default browser and skip capture (guest mode)."""
    try:
        webbrowser.open(spec.start_url)
    except Exception as exc:
        logger.debug("webbrowser.open failed: %s", exc)
    msg = (
        f"Opened {spec.label} in your browser. "
        "Install Playwright (`pip install playwright` then `playwright install chromium`) "
        "for automatic cookie capture. Continuing in public/guest mode."
    )
    _progress(progress, msg)
    return CaptureResult(
        book=spec.label,
        ok=False,
        status="skipped",
        message=msg,
        backend="webbrowser",
    )


def capture_book_login(
    spec: BookCaptureSpec,
    *,
    timeout_sec: float | None = None,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> CaptureResult:
    """
    Open book site, wait for login, save Cookie (+ session) to .env.

    force=False skips when usable auth already present.
    """
    timeout = float(timeout_sec if timeout_sec is not None else DEFAULT_TIMEOUT_SEC)
    if not force and _env_has_usable_auth(spec):
        msg = f"{spec.label} cookies already set — skipping login"
        _progress(progress, msg)
        return CaptureResult(
            book=spec.label,
            ok=True,
            status="already_ok",
            message=msg,
        )

    backend = _backend_available()
    try:
        if backend == "playwright":
            return _capture_with_playwright(spec, timeout_sec=timeout, progress=progress)
        if backend == "selenium":
            return _capture_with_selenium(spec, timeout_sec=timeout, progress=progress)
        return _capture_fallback_open_only(spec, progress=progress)
    except Exception as exc:
        logger.warning("Cookie capture error for %s: %s", spec.label, exc)
        return CaptureResult(
            book=spec.label,
            ok=False,
            status="error",
            message=f"{spec.label} capture failed ({exc}) — continuing",
            backend=backend or "none",
        )


def capture_all_book_cookies(
    *,
    force: bool = False,
    timeout_sec: float | None = None,
    progress: ProgressFn | None = None,
    include_mybookie: bool | None = None,
) -> list[CaptureResult]:
    """
    Capture BetNow (then MyBookie if enabled). Fail-soft: always returns results;
    caller proceeds with odds scrape regardless.
    """
    results: list[CaptureResult] = []
    for spec in book_specs(include_mybookie=include_mybookie):
        results.append(
            capture_book_login(
                spec,
                timeout_sec=timeout_sec,
                progress=progress,
                force=force,
            )
        )
    captured_n = sum(1 for r in results if r.status == "captured")
    if captured_n:
        _progress(
            progress,
            "Cookies captured successfully — scraping odds with fresh cookies…",
        )
    else:
        _progress(
            progress,
            "Proceeding with odds scrape (public/guest mode or existing cookies)…",
        )
    return results


def ensure_cookies_before_refresh(
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> list[CaptureResult]:
    """
    Entry point for dashboard Refresh / Capture Cookies flows.

    When COOKIE_CAPTURE_ON_REFRESH is false and force is false, no-op.
    Timeout defaults to 3 minutes. Never stores username/password.
    """
    enabled = os.getenv("COOKIE_CAPTURE_ON_REFRESH", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    if not force and not enabled:
        return []
    return capture_all_book_cookies(force=force, progress=progress)
