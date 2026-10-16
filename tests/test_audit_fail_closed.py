"""Safety checks from the sizing / leakage audit."""

from __future__ import annotations

import pandas as pd

import config
from src.odds_providers.odds_api_client import odds_api_fail_closed_message
from src.safe_io import redact_secrets
from src.strategy import edge_for_predicted_pick, strategy_from_profile


def test_edge_for_predicted_pick_ignores_opponent_best_edge():
    row = pd.Series(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "predicted_winner": "A",
            "edge_f1": -0.04,
            "edge_f2": 0.15,
            "best_edge": 0.15,
            "best_edge_side": "f2",
            "edge_pct": -4.0,
        }
    )
    edge, pick = edge_for_predicted_pick(row)
    assert pick == "A"
    assert edge == -0.04


def test_dynamic_singles_edge_cannot_undercut_live_floor(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "HEALTH_FEEDBACK_ENABLED", False)

    class _Thresholds:
        alert_min_edge = 0.06
        parlay_min_edge = 0.05
        parlay_min_combined_prob = 0.30
        parlay_min_ev = 0.10

    monkeypatch.setattr(
        "ufc_betting_bot.modules.dynamic_thresholds.get_profile_thresholds",
        lambda *args, **kwargs: _Thresholds(),
    )
    strat = strategy_from_profile(bankroll=1000.0, use_dynamic_thresholds=True)
    assert strat.min_edge >= 0.09
    assert strat.parlay_min_edge >= 0.08


def test_redact_secrets_strips_api_key_and_session():
    raw = (
        "HTTP 401 for url: https://api.example/odds?apiKey=super-secret&regions=us "
        "session=abc123token"
    )
    cleaned = redact_secrets(raw)
    assert "super-secret" not in cleaned
    assert "abc123token" not in cleaned
    assert "apiKey=<redacted>" in cleaned
    assert "session=<redacted>" in cleaned


def test_fail_closed_message_redacts_api_key():
    msg = odds_api_fail_closed_message(
        status_code=500,
        detail="HTTP 500 for url: https://api.example/odds?apiKey=super-secret&regions=us",
    )
    assert "super-secret" not in msg
    assert "apiKey=<redacted>" in msg
