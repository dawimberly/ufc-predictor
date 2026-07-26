"""Settlement PnL/CLV + segment-health feedback into dynamic thresholds."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src.settlement import (
    closing_odds_from_fight_row,
    compute_clv,
    compute_pnl,
    health_lookback_days,
    settlement_complete,
)
from src.strategy_performance import record_closed_bet, segment_health
from src.strategy_rating import clear_rating_cache as _clear_rating
from ufc_betting_bot.modules.dynamic_thresholds import get_profile_thresholds


def test_compute_pnl_and_clv():
    assert compute_pnl(correct=True, stake=10.0, opening_odds=2.0) == pytest.approx(10.0)
    assert compute_pnl(correct=False, stake=10.0, opening_odds=2.0) == pytest.approx(-10.0)
    # Fail-closed: missing stake/odds → None
    assert compute_pnl(correct=True, stake=None, opening_odds=2.0) is None
    assert compute_pnl(correct=True, stake=10.0, opening_odds=None) is None

    # Beat the close: opened at 2.20, closed at 2.00 → positive CLV
    clv = compute_clv(opening_odds=2.20, closing_odds=2.00)
    assert clv is not None and clv > 0
    assert compute_clv(opening_odds=2.0, closing_odds=None) is None


def test_closing_odds_from_fight_row():
    hit = pd.Series(
        {
            "f1_odds": 1.80,
            "f2_odds": 2.10,
            "fighter_1": "A Fighter",
            "fighter_2": "B Fighter",
        }
    )
    assert closing_odds_from_fight_row(
        hit, pick="A Fighter", fighter_1="A Fighter", fighter_2="B Fighter"
    ) == pytest.approx(1.80)
    assert closing_odds_from_fight_row(
        hit, pick="B Fighter", fighter_1="A Fighter", fighter_2="B Fighter"
    ) == pytest.approx(2.10)


def test_settlement_complete_fail_closed():
    assert settlement_complete(stake=10, opening_odds=1.9, pnl=9.0) is True
    assert settlement_complete(stake=None, opening_odds=1.9, pnl=None) is False
    assert settlement_complete(stake=10, opening_odds=None, pnl=None) is False


def test_paper_live_lookback(monkeypatch):
    monkeypatch.setattr(config, "PAPER_HEALTH_LOOKBACK_DAYS", 90)
    monkeypatch.setattr(config, "LIVE_HEALTH_LOOKBACK_DAYS", 180)
    assert health_lookback_days("paper") == 90
    assert health_lookback_days("live") == 180


def test_segment_health_and_threshold_feedback(tmp_path, monkeypatch):
    db = tmp_path / "strategy_metrics.db"
    monkeypatch.setattr(config, "STRATEGY_METRICS_DB", db)
    monkeypatch.setattr(config, "STRATEGY_PERFORMANCE_JSON", tmp_path / "perf.json")
    monkeypatch.setattr(config, "HEALTH_MIN_SETTLED_BETS", 5)
    monkeypatch.setattr(config, "HEALTH_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "PAPER_HEALTH_LOOKBACK_DAYS", 365)
    _clear_rating()

    # Incomplete settlements excluded from health
    record_closed_bet(
        prediction_id="incomplete-1",
        profile="paper",
        correct=True,
        stake=None,
        decimal_odds=None,
        settlement_complete=False,
        source="test",
    )
    thin = segment_health(profile="paper", days=365)
    assert thin["complete"] is False
    assert thin["fail_closed"] is True

    base = get_profile_thresholds(
        1000.0, 0.5, 0.55, hours_to_event=48.0, profile="research", segment_health=None
    )
    fail_closed = get_profile_thresholds(
        1000.0, 0.5, 0.55, hours_to_event=48.0, profile="research", segment_health=thin
    )
    assert fail_closed.alert_min_edge > base.alert_min_edge

    for i in range(6):
        record_closed_bet(
            prediction_id=f"win-clv-{i}",
            profile="paper",
            weight_class="lightweight",
            odds_bucket="favorite",
            confidence_label="high",
            prop_type="moneyline",
            correct=True,
            stake=10.0,
            decimal_odds=1.90,
            closing_odds=1.75,
            settlement_complete=True,
            source="test",
        )

    healthy = segment_health(profile="paper", days=365)
    assert healthy["complete"] is True
    assert healthy["hit_rate"] == pytest.approx(1.0)
    assert healthy["avg_clv"] is not None and healthy["avg_clv"] > 0

    # Losing segment → tighten vs healthy positive ROI
    for i in range(6):
        record_closed_bet(
            prediction_id=f"lose-{i}",
            profile="paper",
            correct=False,
            stake=10.0,
            decimal_odds=1.90,
            closing_odds=2.10,
            settlement_complete=True,
            source="test",
        )
    cold = segment_health(profile="paper", days=365)
    cold_t = get_profile_thresholds(
        1000.0, 0.5, 0.55, hours_to_event=48.0, profile="research", segment_health=cold
    )
    assert cold_t.alert_min_edge >= fail_closed.alert_min_edge or any(
        "ROI" in a or "hit rate" in a or "CLV" in a for a in cold_t.adjustments
    )


def test_bet_journal_settlement(tmp_path, monkeypatch):
    journal = tmp_path / "bet_journal.csv"
    monkeypatch.setattr(config, "BET_JOURNAL_CSV", journal)
    from src.bet_journal import log_settlement

    log_settlement(
        prediction_id="abc123",
        event="UFC Test",
        fight="A vs B",
        pick="A",
        correct=True,
        stake=25.0,
        opening_odds=2.0,
        closing_odds=1.85,
        pnl=25.0,
        clv=compute_clv(opening_odds=2.0, closing_odds=1.85),
        weight_class="welterweight",
        odds_bucket="pickem",
        prop_type="moneyline",
        settlement_complete=True,
    )
    text = journal.read_text(encoding="utf-8")
    assert "settle" in text
    assert "abc123" in text
    assert "closing_odds" in text.splitlines()[0]
