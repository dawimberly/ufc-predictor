"""Tests for conservative dynamic threshold adjustments."""

import pytest
from ufc_betting_bot.modules.dynamic_thresholds import (
    get_profile_thresholds,
    model_confidence_from_prob,
)


def test_small_bankroll_stricter_than_reference():
    ref = get_profile_thresholds(1000.0, 0.5, 0.55, hours_to_event=48.0, profile="research")
    small = get_profile_thresholds(250.0, 0.5, 0.55, hours_to_event=48.0, profile="research")
    assert small.alert_min_edge >= ref.alert_min_edge


def test_large_bankroll_stricter_than_reference():
    ref = get_profile_thresholds(1000.0, 0.5, 0.55, hours_to_event=48.0, profile="research")
    large = get_profile_thresholds(10000.0, 0.5, 0.55, hours_to_event=48.0, profile="research")
    assert large.alert_min_edge > ref.alert_min_edge


def test_cold_streak_tightens():
    neutral = get_profile_thresholds(1000.0, 0.5, 0.55, profile="research")
    cold = get_profile_thresholds(1000.0, 0.35, 0.55, profile="research")
    assert cold.alert_min_edge > neutral.alert_min_edge


def test_segment_health_fail_closed_never_loosens():
    base = get_profile_thresholds(
        1000.0, 0.5, 0.55, profile="research", segment_health=None
    )
    incomplete = get_profile_thresholds(
        1000.0,
        0.5,
        0.55,
        profile="research",
        segment_health={"complete": False, "fail_closed": True, "trade_count": 2},
    )
    assert incomplete.alert_min_edge > base.alert_min_edge
    healthy = get_profile_thresholds(
        1000.0,
        0.5,
        0.55,
        profile="research",
        segment_health={
            "complete": True,
            "fail_closed": False,
            "trade_count": 20,
            "roi": 0.20,
            "hit_rate": 0.65,
            "avg_clv": 0.02,
        },
    )
    # Healthy can loosen slightly vs incomplete fail-closed path
    assert healthy.alert_min_edge <= incomplete.alert_min_edge


def test_live_base_higher_than_research():
    research = get_profile_thresholds(1000.0, 0.5, 0.55, profile="research")
    live = get_profile_thresholds(1000.0, 0.5, 0.55, profile="live")
    assert live.alert_min_edge > research.alert_min_edge


def test_model_confidence_from_prob():
    assert model_confidence_from_prob(0.5) == 0.0
    assert model_confidence_from_prob(0.8) == pytest.approx(0.6)
