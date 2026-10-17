"""Tests for year-scoped event walk-forward backtest."""

from __future__ import annotations

import pytest

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.append(str(_repo))

from src.backtester import backtest_2025
from ufc_betting_bot.modules.edge import raw_kelly_fraction


def test_kelly_fraction_positive_edge():
    frac = raw_kelly_fraction(0.6, 2.0)
    assert frac > 0


def test_kelly_fraction_no_bet_bad_odds():
    assert raw_kelly_fraction(0.6, 1.0) == 0.0


def test_backtest_2025_raises_without_year_data(monkeypatch):
    import pandas as pd
    import config

    features = pd.DataFrame(
        {
            config.DATE_COLUMN: pd.to_datetime(["2024-01-01", "2024-06-01"]),
            config.TARGET_COLUMN: [1, 0],
            config.FIGHT_ID_COLUMN: ["a", "b"],
            "fighter_1": ["A", "C"],
            "fighter_2": ["B", "D"],
            "event": ["Early", "Mid"],
        }
    )
    for col in config.FEATURE_COLUMNS[:5]:
        features[col] = 0.0

    class _Stub:
        feature_columns = [c for c in config.FEATURE_COLUMNS[:5] if c in features.columns]
        interaction_specs: list = []

        class model:
            @staticmethod
            def predict_proba(frame: pd.DataFrame):
                import numpy as np

                col = np.full(len(frame), 0.5)
                return np.column_stack([1.0 - col, col])

    import importlib

    bt_mod = importlib.import_module("ufc_betting_bot.backtester.backtest_2025")
    monkeypatch.setattr(bt_mod, "model_exists", lambda: True)
    monkeypatch.setattr(bt_mod, "get_predictor", lambda: _Stub())

    with pytest.raises(ValueError, match="No fights for 2025"):
        bt_mod.backtest_2025(features, save_outputs=False)
