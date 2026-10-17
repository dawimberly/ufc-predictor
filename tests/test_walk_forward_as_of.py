"""Strict as-of prices + walk-forward past-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufc_betting_bot.backtester.backtest_2025 import walk_forward_events
from ufc_betting_bot.config.settings import BankrollSettings
from ufc_betting_bot.modules.bankroll import simulate_bankroll_bets
from ufc_betting_bot.modules.edge import as_of_market_probs, as_of_price_from_row, validated_as_of_price


def test_as_of_market_probs_prefers_opening_and_refuses_closing_only():
    both = pd.Series(
        {
            "f1_opening_odds": 2.0,
            "f2_opening_odds": 2.0,
            "f1_odds": 1.40,
            "f2_odds": 3.20,
            "f1_closing_odds": 1.40,
            "f2_closing_odds": 3.20,
        }
    )
    probs, timing = as_of_market_probs(both)
    assert timing == "opening"
    assert probs is not None
    assert abs(probs[0] - 0.5) < 1e-9

    closing_only = pd.Series(
        {
            "f1_odds": 1.40,
            "f2_odds": 3.20,
            "f1_closing_odds": 1.40,
            "f2_closing_odds": 3.20,
        }
    )
    refused, refused_timing = as_of_market_probs(closing_only)
    assert refused is None
    assert refused_timing == "closing_not_as_of"


def test_unproven_unlabeled_odds_excluded_from_strict_as_of():
    unlabeled = pd.Series({"f1_odds": 2.0, "f2_odds": 2.0})
    stored, stored_timing = as_of_market_probs(unlabeled)
    assert stored is None
    assert stored_timing == "unproven"
    price, timing = validated_as_of_price(unlabeled)
    assert price is None and timing == "unproven"


def test_one_validated_price_used_for_selection_sizing_settlement():
    """Opening price drives edge, Kelly stake odds, and settlement PnL together."""
    predictions = pd.DataFrame(
        [
            {
                "event_date": "2025-03-01",
                "event_name": "Card",
                "f1_win": 1,
                "prob_f1_win": 0.70,
                "prob_f2_win": 0.30,
                # Closing/unlabeled would be 1.40 — must not be used.
                "f1_odds": 1.40,
                "f2_odds": 3.20,
                "f1_closing_odds": 1.40,
                "f2_closing_odds": 3.20,
                "f1_opening_odds": 2.10,
                "f2_opening_odds": 1.80,
                "odds_timing": "opening",
                "as_of_decimal_f1": 2.10,
                "as_of_decimal_f2": 1.80,
                "implied_prob_f1": 1 / 2.10 / (1 / 2.10 + 1 / 1.80),
                "implied_prob_f2": 1 / 1.80 / (1 / 2.10 + 1 / 1.80),
            }
        ]
    )
    price = as_of_price_from_row(predictions.iloc[0])
    assert price is not None
    assert price.timing == "opening"
    assert price.decimal_f1 == 2.10

    settings = BankrollSettings(
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        min_bet_fraction=0.0,
        max_card_risk_fraction=1.0,
        daily_loss_limit_fraction=1.0,
        min_edge=0.01,
        flat_stake=10.0,
    )
    trades, _ = simulate_bankroll_bets(predictions, settings)
    assert len(trades) == 1
    assert float(trades.iloc[0]["odds"]) == 2.10
    assert trades.iloc[0]["odds_timing"] == "opening"
    # Win at opening 2.10, not at closing 1.40.
    stake = float(trades.iloc[0]["stake"])
    assert float(trades.iloc[0]["pnl"]) == pytest.approx(stake * (2.10 - 1.0))


def test_bankroll_skips_closing_only_and_unproven():
    settings = BankrollSettings(
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        min_bet_fraction=0.0,
        max_card_risk_fraction=1.0,
        daily_loss_limit_fraction=1.0,
        min_edge=0.01,
        flat_stake=10.0,
    )
    closing_only = pd.DataFrame(
        [
            {
                "event_date": "2025-03-01",
                "event_name": "Card",
                "f1_win": 1,
                "prob_f1_win": 0.80,
                "prob_f2_win": 0.20,
                "f1_odds": 1.40,
                "f2_odds": 3.20,
                "f1_closing_odds": 1.40,
                "f2_closing_odds": 3.20,
                "odds_timing": "closing_not_as_of",
            }
        ]
    )
    trades, summary = simulate_bankroll_bets(closing_only, settings)
    assert trades.empty
    assert summary.get("trades", 0) == 0

    unproven = pd.DataFrame(
        [
            {
                "event_date": "2025-03-01",
                "event_name": "Card",
                "f1_win": 1,
                "prob_f1_win": 0.80,
                "prob_f2_win": 0.20,
                "f1_odds": 2.0,
                "f2_odds": 2.0,
                "odds_timing": "unproven",
            }
        ]
    )
    trades2, summary2 = simulate_bankroll_bets(unproven, settings)
    assert trades2.empty
    assert summary2.get("trades", 0) == 0


class _Frozen:
    feature_columns = ["x"]
    interaction_specs: list = []

    class model:
        @staticmethod
        def predict_proba(frame: pd.DataFrame) -> np.ndarray:
            col = np.full(len(frame), 0.9)
            return np.column_stack([1.0 - col, col])


def test_walk_forward_refit_uses_only_earlier_fights():
    seen: list[pd.Series] = []

    class _Model:
        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            col = np.full(len(frame), self.probability)
            return np.column_stack([1.0 - col, col])

    def _trainer(train: pd.DataFrame, _cols: list[str]) -> _Model:
        dates = pd.to_datetime(train["event_date"])
        seen.append(dates)
        assert dates.max() < pd.Timestamp("2025-03-01")
        assert "Late" not in set(train["event"].astype(str))
        return _Model(float(train["f1_win"].mean()))

    features = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "event": "Early",
                "fighter_1": "A",
                "fighter_2": "B",
                "f1_win": 0,
                "x": 0.0,
                "f1_odds": 2.0,
                "f2_odds": 2.0,
            },
            {
                "event_date": "2025-03-01",
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 1.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
                "f1_closing_odds": 1.40,
                "f2_closing_odds": 3.20,
                "f1_odds": 1.40,
                "f2_odds": 3.20,
            },
        ]
    )
    out = walk_forward_events(
        features,
        _Frozen(),
        target_year=2025,
        refit_model=True,
        trainer=_trainer,
        min_train_rows=1,
    )
    assert len(seen) == 1
    assert len(out) == 1
    assert out.loc[0, "evaluation_mode"] == "walk_forward"
    assert float(out.loc[0, "prob_f1_win"]) == 0.0
    assert out.loc[0, "odds_timing"] == "opening"
    assert abs(float(out.loc[0, "implied_prob_f1"]) - 0.5) < 1e-9
    assert float(out.loc[0, "as_of_decimal_f1"]) == 2.0

    frozen = walk_forward_events(features, _Frozen(), target_year=2025, refit_model=False)
    assert frozen.loc[0, "evaluation_mode"] == "frozen_model"
    assert float(frozen.loc[0, "prob_f1_win"]) == 0.9


def test_walk_forward_imputer_and_fit_ignore_future_rows(monkeypatch):
    """Future feature values must not affect the past-only imputer mean."""
    from src import feature_engineering as fe

    captured_means: list[float] = []
    real_fit = fe.fit_imputer

    def _spy(train: pd.DataFrame):
        stats = real_fit(train)
        if "x" in train.columns:
            captured_means.append(float(pd.to_numeric(train["x"], errors="coerce").mean()))
        return stats

    monkeypatch.setattr(fe, "fit_imputer", _spy)

    features = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "event": "Early",
                "fighter_1": "A",
                "fighter_2": "B",
                "f1_win": 0,
                "x": 10.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
            },
            {
                "event_date": "2024-06-01",
                "event": "Mid",
                "fighter_1": "E",
                "fighter_2": "F",
                "f1_win": 1,
                "x": 20.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
            },
            {
                "event_date": "2025-03-01",
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 999.0,  # must never enter imputer for the 2025 fold
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
            },
        ]
    )
    out = walk_forward_events(
        features,
        _Frozen(),
        target_year=2025,
        refit_model=True,
        min_train_rows=1,
    )
    assert len(out) == 1
    assert out.loc[0, "evaluation_mode"] == "walk_forward"
    assert captured_means
    # Past-only mean of x is (10+20)/2 = 15, never pulled toward 999.
    assert all(abs(m - 15.0) < 1e-9 for m in captured_means)


def test_walk_forward_skips_frozen_interaction_specs(monkeypatch):
    applied: list[int] = []

    def _boom(df, specs):
        applied.append(len(specs))
        return df

    monkeypatch.setattr(
        "src.feature_engineering.apply_interaction_specs",
        _boom,
    )

    class _WithSpecs(_Frozen):
        interaction_specs = [{"name": "leaky", "factor_a": "x", "factor_b": "x"}]

    features = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "event": "Early",
                "fighter_1": "A",
                "fighter_2": "B",
                "f1_win": 0,
                "x": 0.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
            },
            {
                "event_date": "2025-03-01",
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 1.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
            },
        ]
    )
    walk_forward_events(
        features,
        _WithSpecs(),
        target_year=2025,
        refit_model=True,
        min_train_rows=1,
    )
    assert applied == []

    walk_forward_events(
        features,
        _WithSpecs(),
        target_year=2025,
        refit_model=False,
    )
    assert applied  # frozen path may still apply production specs
