"""Tests for home-country mapping (card / nationality proxy)."""

from __future__ import annotations

from src.home_country import location_to_country


def test_location_to_country_examples() -> None:
    assert location_to_country("Sydney, NSW") == "australia"
    assert location_to_country("Melbourne") == "australia"
    assert location_to_country("Gold Coast, QLD") == "australia"
    assert location_to_country("Abu Dhabi") == "uae"
    assert location_to_country("London") == "uk"
    assert location_to_country("Sao Paulo") == "brazil"
    assert location_to_country("Denver, CO") == "usa"
    assert location_to_country("Toronto, ON") == "canada"
    assert location_to_country("") == ""
