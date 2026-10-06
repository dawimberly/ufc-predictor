"""Walk-forward conformal width must not be squeezed into the blue-tier band."""

from __future__ import annotations

import config
from src.ha_backtest import clamp_walkforward_conformal_q


def test_raw_wide_quantile_stays_wide(monkeypatch):
    monkeypatch.setattr(config, "HA_WF_CONFORMAL_Q_CAP", None)
    monkeypatch.setattr(config, "HA_WF_CONFORMAL_Q_FLOOR", 0.05)
    assert clamp_walkforward_conformal_q(0.60) == 0.60


def test_explicit_cap_still_applies(monkeypatch):
    monkeypatch.setattr(config, "HA_WF_CONFORMAL_Q_CAP", 0.14)
    monkeypatch.setattr(config, "HA_WF_CONFORMAL_Q_FLOOR", 0.05)
    assert clamp_walkforward_conformal_q(0.60) == 0.14


def test_floor_raises_tiny_quantile(monkeypatch):
    monkeypatch.setattr(config, "HA_WF_CONFORMAL_Q_CAP", None)
    monkeypatch.setattr(config, "HA_WF_CONFORMAL_Q_FLOOR", 0.05)
    assert clamp_walkforward_conformal_q(0.01) == 0.05
