"""Cookie capture helpers (no live browser)."""

from __future__ import annotations

from pathlib import Path

import config
from src.odds_providers.cookie_capture import (
    BookCaptureSpec,
    _cookie_header_from_list,
    _env_has_usable_auth,
    _extract_session_from_cookies,
    _extract_session_from_url,
    _login_detected,
    book_specs,
    capture_book_login,
)


def test_cookie_header_and_session_extract():
    cookies = [
        {"name": "PHPSESSID", "value": "abc12345"},
        {"name": "session", "value": "tok-999"},
        {"name": "other", "value": "x"},
    ]
    hdr = _cookie_header_from_list(cookies)
    assert "session=tok-999" in hdr
    assert "PHPSESSID=abc12345" in hdr
    assert _extract_session_from_cookies(cookies) == "tok-999"
    assert (
        _extract_session_from_url(
            "https://www.betnow.eu/sportsbook-info/fighting/ufc/?session=1783-xyz"
        )
        == "1783-xyz"
    )


def test_login_detected_betnow():
    spec = BookCaptureSpec(
        key="betnow",
        label="BetNow",
        start_url="https://www.betnow.eu/",
        cookie_env="BETNOW_COOKIE",
        session_env="BETNOW_SESSION",
        auth_cookie_names=frozenset({"session", "sid"}),
        success_path_hints=("ufc", "sportsbook"),
    )
    assert _login_detected(
        spec,
        url="https://www.betnow.eu/sportsbook-info/fighting/ufc/",
        cookies=[{"name": "session", "value": "alive-token-1"}],
    )
    assert not _login_detected(
        spec,
        url="https://www.betnow.eu/login",
        cookies=[{"name": "tracking", "value": "1"}],
    )


def test_book_specs_respect_mybookie(monkeypatch):
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", True)
    keys = [s.key for s in book_specs()]
    assert keys == ["betnow", "mybookie"]
    monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)
    keys = [s.key for s in book_specs()]
    assert keys == ["betnow"]


def test_upsert_env_vars_roundtrip(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("FOO=1\nBETNOW_COOKIE=old\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    config.upsert_env_vars(
        {"BETNOW_COOKIE": "a=1; b=2", "BETNOW_SESSION": "sess-12345678"},
        env_path=env_path,
    )
    text = env_path.read_text(encoding="utf-8")
    assert "BETNOW_COOKIE=a=1; b=2" in text
    assert "BETNOW_SESSION=sess-12345678" in text
    assert config.BETNOW_COOKIE == "a=1; b=2"
    assert config.BETNOW_SESSION_TOKEN == "sess-12345678"


def test_skip_when_auth_present(monkeypatch):
    monkeypatch.setenv("BETNOW_SESSION", "1783704275000-1281694-aaahfrk")
    monkeypatch.setenv("BETNOW_COOKIE", "")
    monkeypatch.setattr(config, "BETNOW_SESSION_TOKEN", "1783704275000-1281694-aaahfrk")
    monkeypatch.setattr(config, "BETNOW_COOKIE", "")
    spec = book_specs(include_mybookie=False)[0]
    assert _env_has_usable_auth(spec)
    result = capture_book_login(spec, force=False, progress=None)
    assert result.status == "already_ok"


def test_force_without_browser_skips_gracefully(monkeypatch):
    monkeypatch.setattr(
        "src.odds_providers.cookie_capture._backend_available",
        lambda: "",
    )
    monkeypatch.setattr(
        "src.odds_providers.cookie_capture.webbrowser.open",
        lambda url: True,
    )
    msgs: list[str] = []
    spec = book_specs(include_mybookie=False)[0]
    result = capture_book_login(
        spec, force=True, timeout_sec=5, progress=msgs.append
    )
    assert result.status == "skipped"
    assert result.ok is False
    assert any("Opened" in m or "Playwright" in m or "guest" in m.lower() for m in msgs)


def test_persist_auth_allowlist_only(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    from src.odds_providers.cookie_capture import _persist_auth, BookCaptureSpec

    spec = BookCaptureSpec(
        key="betnow",
        label="BetNow",
        start_url="https://www.betnow.eu/",
        cookie_env="BETNOW_COOKIE",
        session_env="BETNOW_SESSION",
        auth_cookie_names=frozenset({"session"}),
    )
    _persist_auth(spec, "session=abc; other=1", "sess-token-99")
    text = env_path.read_text(encoding="utf-8")
    assert "BETNOW_COOKIE=" in text
    assert "BETNOW_SESSION=" in text
    assert "password" not in text.lower()
    assert "username" not in text.lower()
