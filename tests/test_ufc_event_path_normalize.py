"""Unit tests for UFC.com event path normalization / discovery filtering."""

from __future__ import annotations

from src.data_loader import _normalize_ufc_com_event_path


def test_normalize_accepts_relative_event_path():
    assert _normalize_ufc_com_event_path("/event/ufc-fight-night-august-01-2026") == (
        "/event/ufc-fight-night-august-01-2026"
    )


def test_normalize_rejects_ticket_partner_urls():
    assert (
        _normalize_ufc_com_event_path(
            "https://tickets.rs/event/ufc_fight_night_belgrade_26702"
        )
        is None
    )
    assert (
        _normalize_ufc_com_event_path(
            "https://ticketmaster.com/event/ufc-something"
        )
        is None
    )


def test_normalize_accepts_ufc_com_absolute_urls():
    assert _normalize_ufc_com_event_path(
        "https://www.ufc.com/event/ufc-330#section"
    ) == "/event/ufc-330"
    assert _normalize_ufc_com_event_path(
        "https://ufc.com/event/ufc-fight-night-august-01-2026"
    ) == "/event/ufc-fight-night-august-01-2026"


def test_normalize_rejects_empty_and_non_event():
    assert _normalize_ufc_com_event_path("") is None
    assert _normalize_ufc_com_event_path("/athlete/jon-jones") is None
    assert _normalize_ufc_com_event_path("https://www.ufc.com/athletes") is None
