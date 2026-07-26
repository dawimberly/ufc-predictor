"""Tests for segment strategy rating / Kelly clamps."""

from __future__ import annotations

import config
from src.strategy_performance import (
    classify_bet_segments,
    odds_bucket_from_decimal,
    record_closed_bet,
    compute_segment_metrics,
    _fetch_bets,
)
from src.strategy_rating import (
    apply_rating_to_kelly_fraction,
    clear_rating_cache,
    combine_segment_multipliers,
    kelly_multiplier_for_context,
    score_to_kelly_multiplier,
    refresh_ratings,
)
from src.strategy import StrategyConfig, kelly_stake


def test_odds_bucket_boundaries():
    assert odds_bucket_from_decimal(1.50) == "favorite"
    assert odds_bucket_from_decimal(2.00) == "pickem"
    assert odds_bucket_from_decimal(2.80) == "mild_dog"
    assert odds_bucket_from_decimal(5.00) == "longshot"
    assert odds_bucket_from_decimal(None) == "unknown"


def test_classify_bet_segments():
    segs = classify_bet_segments(
        weight_class="Lightweight Title",
        decimal_odds=1.65,
        confidence_label="high",
        prop_type="moneyline",
    )
    assert segs["weight_class"] == "lightweight"
    assert segs["odds_bucket"] == "favorite"
    assert segs["confidence_label"] == "high"
    assert segs["prop_type"] == "moneyline"


def test_score_to_kelly_multiplier_thin_sample_neutral():
    assert score_to_kelly_multiplier(90.0, trade_count=2) == 1.0
    assert score_to_kelly_multiplier(10.0, trade_count=2) == 1.0


def test_score_to_kelly_multiplier_clamped(monkeypatch):
    monkeypatch.setattr(config, "STRATEGY_RATING_MIN_TRADES", 5)
    monkeypatch.setattr(config, "STRATEGY_RATING_MULT_MIN", 0.8)
    monkeypatch.setattr(config, "STRATEGY_RATING_MULT_MAX", 1.2)
    hi = score_to_kelly_multiplier(100.0, trade_count=20)
    lo = score_to_kelly_multiplier(0.0, trade_count=20)
    mid = score_to_kelly_multiplier(50.0, trade_count=20)
    assert hi == 1.2
    assert lo == 0.8
    assert mid == 1.0


def test_live_profile_fail_closed_unless_enabled(monkeypatch):
    monkeypatch.setattr(config, "STRATEGY_RATING_ENABLED", True)
    monkeypatch.setattr(config, "STRATEGY_RATING_LIVE_ENABLED", False)
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    assert config.effective_strategy_rating_enabled() is False
    assert kelly_multiplier_for_context(confidence_label="high") == 1.0

    monkeypatch.setattr(config, "STRATEGY_RATING_LIVE_ENABLED", True)
    assert config.effective_strategy_rating_enabled() is True


def test_paper_enabled_by_default(monkeypatch):
    monkeypatch.setattr(config, "STRATEGY_RATING_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    assert config.effective_strategy_rating_enabled() is True


def test_record_and_score_segments(tmp_path, monkeypatch):
    db = tmp_path / "strategy_metrics.db"
    monkeypatch.setattr(config, "STRATEGY_METRICS_DB", db)
    monkeypatch.setattr(config, "STRATEGY_PERFORMANCE_JSON", tmp_path / "perf.json")
    monkeypatch.setattr(config, "STRATEGY_RATING_MIN_TRADES", 5)
    monkeypatch.setattr(config, "STRATEGY_RATING_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    clear_rating_cache()

    for i in range(6):
        record_closed_bet(
            prediction_id=f"win-{i}",
            weight_class="welterweight",
            odds_bucket="favorite",
            confidence_label="high",
            prop_type="moneyline",
            correct=True,
            stake=10.0,
            decimal_odds=1.70,
            source="test",
        )
    for i in range(6):
        record_closed_bet(
            prediction_id=f"lose-{i}",
            weight_class="heavyweight",
            odds_bucket="longshot",
            confidence_label="low",
            prop_type="finish",
            correct=False,
            stake=10.0,
            decimal_odds=4.00,
            source="test",
        )

    metrics = compute_segment_metrics(_fetch_bets(None))
    assert metrics["weight_class:welterweight"]["trade_count"] == 6
    assert metrics["weight_class:welterweight"]["risk_adjusted_score"] > 50
    assert metrics["weight_class:heavyweight"]["risk_adjusted_score"] < 50

    snap = refresh_ratings(force=True)
    ww = snap["segments"]["weight_class:welterweight"]["kelly_mult"]
    hw = snap["segments"]["weight_class:heavyweight"]["kelly_mult"]
    assert ww > 1.0
    assert hw < 1.0

    mult = combine_segment_multipliers(
        {
            "weight_class": "welterweight",
            "odds_bucket": "favorite",
            "confidence_label": "high",
            "prop_type": "moneyline",
        }
    )
    assert mult > 1.0


def test_kelly_stake_applies_rating_mult(monkeypatch):
    monkeypatch.setattr(config, "STRATEGY_RATING_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    cfg = StrategyConfig(
        kelly_fraction=0.25,
        min_edge=0.01,
        min_bet_fraction=0.0,
        max_bet_fraction=0.50,
    )
    base = kelly_stake(1000.0, prob=0.60, decimal_odds=2.0, edge=0.10, config=cfg, rating_mult=1.0)
    boosted = kelly_stake(1000.0, prob=0.60, decimal_odds=2.0, edge=0.10, config=cfg, rating_mult=1.2)
    cut = kelly_stake(1000.0, prob=0.60, decimal_odds=2.0, edge=0.10, config=cfg, rating_mult=0.8)
    assert boosted > base > cut
    assert abs(boosted / base - 1.2) < 1e-6
    assert abs(cut / base - 0.8) < 1e-6


def test_apply_rating_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(config, "STRATEGY_RATING_ENABLED", False)
    assert apply_rating_to_kelly_fraction(0.25, rating_mult=1.2) == 0.25
