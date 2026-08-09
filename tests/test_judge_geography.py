"""Tests for judge geography display helpers."""

from __future__ import annotations

from src.judge_geography import format_panel_geography_note, judge_country


def test_format_panel_note() -> None:
    assert "majority event-country" in format_panel_geography_note(
        ["A", "B", "C"], panel_event_country_share=1.0, event_country="usa"
    )
    assert "mixed/neutral" in format_panel_geography_note(
        "A; B; C", panel_event_country_share=0.33, event_country="uk"
    )


def test_judge_country_seed() -> None:
    assert judge_country("Sal D'Amato") == "usa"
    assert judge_country("Ben Cartlidge") == "uk"
    assert judge_country("Junichiro Kamijo") == "japan"
    assert judge_country("Unknown Person") == ""
