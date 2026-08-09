"""Tests for gym local-advantage location matching."""

from __future__ import annotations

from src.gym_data import _location_overlap as overlap


def test_location_overlap_city_and_state_abbrev() -> None:
    assert overlap("Denver Colorado", "Denver, CO") is True
    assert overlap("Englewood Colorado", "Denver, CO") is True  # state token
    assert overlap("Albuquerque New Mexico", "Albuquerque, NM") is True
    assert overlap("Phoenix Arizona", "Las Vegas, NV") is False
    assert overlap("", "Denver, CO") is False
