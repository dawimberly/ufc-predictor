"""Strict as-of prices + walk-forward past-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufc_betting_bot.backtester.backtest_2025 import walk_forward_events
from ufc_betting_bot.config.settings import BankrollSettings
from ufc_betting_bot.modules.bankroll import simulate_bankroll_bets
from ufc_betting_bot.modules.edge import (
    as_of_eligibility_counts,
    as_of_market_probs,
    as_of_price_from_row,
    validated_as_of_price,
)

OPEN_TS = "2025-02-15T18:00:00+00:00"
EVENT_DAY = "2025-03-01"


def _opening_row(**extra):
    row = {
        "f1_opening_odds": 2.0,
        "f2_opening_odds": 2.0,
        "opening_odds_observed_at": OPEN_TS,
        "event_date": EVENT_DAY,
        "f1_odds": 1.40,
        "f2_odds": 3.20,
        "f1_closing_odds": 1.40,
        "f2_closing_odds": 3.20,
    }
    row.update(extra)
    return pd.Series(row)


def test_as_of_market_probs_prefers_opening_and_refuses_closing_only():
    probs, timing = as_of_market_probs(_opening_row())
    assert timing == "opening"
    assert probs is not None
    assert abs(probs[0] - 0.5) < 1e-9

    closing_only = pd.Series(
        {
            "f1_odds": 1.40,
            "f2_odds": 3.20,
            "f1_closing_odds": 1.40,
            "f2_closing_odds": 3.20,
            "event_date": EVENT_DAY,
        }
    )
    refused, refused_timing = as_of_market_probs(closing_only)
    assert refused is None
    assert refused_timing == "closing_not_as_of"


def test_opening_label_without_timestamp_is_unproven():
    labeled = pd.Series(
        {
            "f1_opening_odds": 2.0,
            "f2_opening_odds": 2.0,
            "event_date": EVENT_DAY,
        }
    )
    price, timing = validated_as_of_price(labeled)
    assert price is None
    assert timing == "unproven"


def test_opening_after_event_cutoff_is_unproven():
    late = _opening_row(opening_odds_observed_at="2025-03-02T00:00:00+00:00")
    price, timing = validated_as_of_price(late)
    assert price is None
    assert timing == "unproven"


def test_naive_opening_quote_timestamp_is_unproven():
    naive = _opening_row(opening_odds_observed_at="2025-02-15T18:00:00")  # no tz
    price, timing = validated_as_of_price(naive)
    assert price is None
    assert timing == "unproven"


def test_post_fight_as_of_cannot_authorize_post_fight_quote():
    """Cutoff is clamped to fight start; later as_of must not help."""
    row = pd.Series(
        {
            "f1_opening_odds": 2.0,
            "f2_opening_odds": 2.0,
            "opening_odds_observed_at": "2025-03-01T21:00:00+00:00",
            "commence_time": "2025-03-01T20:00:00+00:00",
            "event_date": "2025-03-01",
        }
    )
    assert validated_as_of_price(row)[1] == "unproven"
    price, timing = validated_as_of_price(row, as_of="2025-03-02T00:00:00+00:00")
    assert price is None
    assert timing == "unproven"


def test_json_safe_undefined_metrics_are_na():
    from ufc_betting_bot.backtester.backtest_2025 import _json_safe
    import json

    payload = _json_safe({"roi_pct": float("nan"), "hit_rate": float("nan"), "trades": 0})
    assert payload["roi_pct"] == "N/A"
    assert payload["hit_rate"] == "N/A"
    assert '"NaN"' not in json.dumps(payload)


def test_unproven_unlabeled_odds_excluded_from_strict_as_of():
    unlabeled = pd.Series({"f1_odds": 2.0, "f2_odds": 2.0, "event_date": EVENT_DAY})
    stored, stored_timing = as_of_market_probs(unlabeled)
    assert stored is None
    assert stored_timing == "unproven"


def test_attached_fields_cannot_bypass_column_rejection():
    """Forged odds_timing=opening with closing decimals must not size."""
    forged = pd.DataFrame(
        [
            {
                "event_date": EVENT_DAY,
                "event_name": "Card",
                "f1_win": 1,
                "prob_f1_win": 0.80,
                "prob_f2_win": 0.20,
                "f1_odds": 1.40,
                "f2_odds": 3.20,
                "f1_closing_odds": 1.40,
                "f2_closing_odds": 3.20,
                "odds_timing": "opening",
                "as_of_decimal_f1": 1.40,
                "as_of_decimal_f2": 3.20,
                "implied_prob_f1": 0.7,
                "implied_prob_f2": 0.3,
            }
        ]
    )
    assert as_of_price_from_row(forged.iloc[0]) is None
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
    trades, summary = simulate_bankroll_bets(forged, settings)
    assert trades.empty
    assert summary.get("performance_available") is False
    assert summary.get("trades") == 0


def test_one_validated_price_used_for_selection_sizing_settlement():
    """Opening price drives edge, Kelly stake odds, and settlement PnL together."""
    m1 = 1 / 2.10 / (1 / 2.10 + 1 / 1.80)
    m2 = 1 / 1.80 / (1 / 2.10 + 1 / 1.80)
    predictions = pd.DataFrame(
        [
            {
                "event_date": EVENT_DAY,
                "event_name": "Card",
                "f1_win": 1,
                "prob_f1_win": 0.70,
                "prob_f2_win": 0.30,
                "f1_odds": 1.40,
                "f2_odds": 3.20,
                "f1_closing_odds": 1.40,
                "f2_closing_odds": 3.20,
                "f1_opening_odds": 2.10,
                "f2_opening_odds": 1.80,
                "opening_odds_observed_at": OPEN_TS,
                "odds_timing": "opening",
                "as_of_decimal_f1": 2.10,
                "as_of_decimal_f2": 1.80,
                "implied_prob_f1": m1,
                "implied_prob_f2": m2,
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
                "event_date": EVENT_DAY,
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
    assert summary.get("trades") == 0
    assert summary.get("performance_available") is False


def test_eligibility_counts_and_zero_eligible_nan_roi():
    preds = pd.DataFrame(
        {
            "odds_timing": ["opening", "closing_not_as_of", "unproven", "missing"],
        }
    )
    counts = as_of_eligibility_counts(preds)
    assert counts["n_total"] == 4
    assert counts["n_as_of_eligible"] == 1
    assert counts["n_closing_not_as_of"] == 1
    assert counts["n_unproven"] == 1
    assert counts["n_missing"] == 1


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
                "event_date": EVENT_DAY,
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 1.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
                "opening_odds_observed_at": OPEN_TS,
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
    assert out.loc[0, "evaluation_mode"] == "walk_forward_simplified"
    assert out.loc[0, "trainer_identity"] == "_trainer"
    assert float(out.loc[0, "prob_f1_win"]) == 0.0
    assert out.loc[0, "odds_timing"] == "opening"
    assert abs(float(out.loc[0, "implied_prob_f1"]) - 0.5) < 1e-9
    assert float(out.loc[0, "as_of_decimal_f1"]) == 2.0

    frozen = walk_forward_events(features, _Frozen(), target_year=2025, refit_model=False)
    assert frozen.loc[0, "evaluation_mode"] == "frozen_model"
    assert float(frozen.loc[0, "prob_f1_win"]) == 0.9


def test_default_wf_trainer_labeled_simplified():
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
                "opening_odds_observed_at": "2023-12-01T00:00:00+00:00",
            },
            {
                "event_date": EVENT_DAY,
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 1.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
                "opening_odds_observed_at": OPEN_TS,
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
    assert out.loc[0, "evaluation_mode"] == "walk_forward_simplified"
    assert out.loc[0, "trainer_identity"] == "simplified_lgbm_isotonic"


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
                "opening_odds_observed_at": "2023-12-01T00:00:00+00:00",
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
                "opening_odds_observed_at": "2024-05-01T00:00:00+00:00",
            },
            {
                "event_date": EVENT_DAY,
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 999.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
                "opening_odds_observed_at": OPEN_TS,
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
    assert out.loc[0, "evaluation_mode"] == "walk_forward_simplified"
    assert captured_means
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
        feature_columns = ["x", "leaky"]

    features = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "event": "Early",
                "fighter_1": "A",
                "fighter_2": "B",
                "f1_win": 0,
                "x": 0.0,
                "leaky": 0.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
                "opening_odds_observed_at": "2023-12-01T00:00:00+00:00",
            },
            {
                "event_date": EVENT_DAY,
                "event": "Late",
                "fighter_1": "C",
                "fighter_2": "D",
                "f1_win": 1,
                "x": 1.0,
                "leaky": 1.0,
                "f1_opening_odds": 2.0,
                "f2_opening_odds": 2.0,
                "opening_odds_observed_at": OPEN_TS,
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
    assert applied
