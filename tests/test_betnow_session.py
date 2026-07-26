"""BetNow session token / URL helpers."""

from __future__ import annotations

from src.odds_providers.betnow_scraper import (
    _append_session_query,
    _extract_session_token_from_text,
)


def test_append_session_query_adds_param():
    url = _append_session_query(
        "https://www.betnow.eu/sportsbook-info/fighting/ufc/",
        "abcTOKEN",
    )
    assert url.endswith("?session=abcTOKEN")


def test_append_session_query_replaces_existing():
    url = _append_session_query(
        "https://www.betnow.eu/sportsbook-info/fighting/ufc/?session=old&x=1",
        "newTok",
    )
    assert "session=newTok" in url
    assert "session=old" not in url
    assert "x=1" in url


def test_extract_session_from_url_text():
    html = '<a href="/sportsbook-info/fighting/ufc/?session=XYZ789token">UFC</a>'
    assert _extract_session_token_from_text(html) == "XYZ789token"
