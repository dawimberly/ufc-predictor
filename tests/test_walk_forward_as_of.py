"""Walk-forward scoring uses past rows only and does not price closing odds as as-of."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufc_betting_bot.backtester.backtest_2025 import walk_forward_events
from ufc_betting_bot.modules.edge import as_of_market_probs


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

    unlabeled = pd.Series({"f1_odds": 2.0, "f2_odds": 2.0})
    stored, stored_timing = as_of_market_probs(unlabeled)
    assert stored_timing == "unspecified"
    assert stored is not None
    assert abs(stored[0] - 0.5) < 1e-9


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

    frozen = walk_forward_events(features, _Frozen(), target_year=2025, refit_model=False)
    assert frozen.loc[0, "evaluation_mode"] == "frozen_model"
    assert float(frozen.loc[0, "prob_f1_win"]) == 0.9
