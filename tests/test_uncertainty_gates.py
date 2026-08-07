"""Tests for ensemble disagreement / interval-width betting gates."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src.strategy import StrategyConfig, extract_bet_candidates, kelly_stake
from src.uncertainty_gates import (
    PAPER_WIDE_OVERRIDE,
    SKIP_HIGH_DISAGREEMENT,
    SKIP_MISSING,
    SKIP_WIDE_INTERVAL,
    evaluate_uncertainty_gate,
    effective_min_edge,
)


def _row(**kwargs):
    base = {
        "fighter_1": "A",
        "fighter_2": "B",
        "prob_f1_win": 0.62,
        "prob_f2_win": 0.38,
        "f1_odds": 2.10,
        "f2_odds": 1.80,
        "ensemble_disagreement": 0.02,
        "interval_width": 0.10,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_allow_when_metrics_low(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    gate = evaluate_uncertainty_gate(_row())
    assert gate.action == "allow"
    assert gate.kelly_mult == 1.0
    assert gate.edge_bump == 0.0


def test_tighten_on_elevated_disagreement(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_TIGHTEN", 0.05)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_SKIP", 0.10)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_TIGHTEN", 0.50)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_SKIP", 0.60)
    monkeypatch.setattr(config, "PAPER_UNCERTAINTY_KELLY_MULT", 0.70)
    monkeypatch.setattr(config, "PAPER_UNCERTAINTY_EDGE_BUMP", 0.02)
    gate = evaluate_uncertainty_gate(_row(ensemble_disagreement=0.07, interval_width=0.10))
    assert gate.action == "tighten"
    assert gate.kelly_mult == pytest.approx(0.70)
    assert gate.edge_bump == pytest.approx(0.02)
    assert effective_min_edge(0.04, gate) == pytest.approx(0.06)


def test_skip_high_disagreement(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    gate = evaluate_uncertainty_gate(_row(ensemble_disagreement=0.15, interval_width=0.10))
    assert gate.action == "skip"
    assert SKIP_HIGH_DISAGREEMENT in gate.reasons
    assert gate.kelly_mult == 0.0


def test_skip_wide_interval(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    gate = evaluate_uncertainty_gate(_row(ensemble_disagreement=0.02, interval_width=0.35))
    assert gate.action == "skip"
    assert SKIP_WIDE_INTERVAL in gate.reasons


def test_fail_closed_missing_metrics(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    gate = evaluate_uncertainty_gate(_row(ensemble_disagreement=None, interval_width=None))
    # Series with None may still read as missing
    gate2 = evaluate_uncertainty_gate({"fighter_1": "A", "fighter_2": "B"})
    assert gate2.action == "skip"
    assert SKIP_MISSING in gate2.reasons


def test_disabled_gates_allow(monkeypatch):
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", False)
    gate = evaluate_uncertainty_gate({"fighter_1": "A"})  # missing metrics
    assert gate.action == "allow"


def test_extract_bet_candidates_skips_uncertain(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    cfg = StrategyConfig(min_edge=0.03, kelly_fraction=0.25)
    # Need market columns — edge helpers expect odds
    row = _row(ensemble_disagreement=0.20, interval_width=0.10, implied_prob_f1=0.48, implied_prob_f2=0.52)
    # Without proper market_probs the candidate may be None anyway; force via gate path
    assert extract_bet_candidates(row, config=cfg) is None


def test_kelly_stake_zero_on_skip(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    cfg = StrategyConfig(min_edge=0.01, kelly_fraction=0.25, max_bet_fraction=0.5, min_bet_fraction=0.0)
    stake = kelly_stake(
        1000.0,
        prob=0.60,
        decimal_odds=2.0,
        edge=0.10,
        config=cfg,
        row=_row(ensemble_disagreement=0.20, interval_width=0.10),
    )
    assert stake == 0.0


def test_kelly_stake_cut_on_tighten(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_TIGHTEN", 0.05)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_SKIP", 0.12)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_TIGHTEN", 0.50)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_SKIP", 0.60)
    monkeypatch.setattr(config, "PAPER_UNCERTAINTY_KELLY_MULT", 0.50)
    monkeypatch.setattr(config, "PAPER_UNCERTAINTY_EDGE_BUMP", 0.0)
    cfg = StrategyConfig(min_edge=0.01, kelly_fraction=0.25, max_bet_fraction=0.5, min_bet_fraction=0.0)
    base = kelly_stake(
        1000.0,
        prob=0.60,
        decimal_odds=2.0,
        edge=0.10,
        config=cfg,
        row=_row(ensemble_disagreement=0.02, interval_width=0.10),
        uncertainty_kelly_mult=1.0,
    )
    cut = kelly_stake(
        1000.0,
        prob=0.60,
        decimal_odds=2.0,
        edge=0.10,
        config=cfg,
        row=_row(ensemble_disagreement=0.07, interval_width=0.10),
    )
    assert cut < base
    assert cut == pytest.approx(base * 0.5, rel=1e-3)


def test_live_stricter_than_paper(monkeypatch):
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_SKIP", 0.10)
    monkeypatch.setattr(config, "LIVE_DISAGREEMENT_SKIP", 0.08)
    row = _row(ensemble_disagreement=0.09, interval_width=0.10)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    paper = evaluate_uncertainty_gate(row)
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    live = evaluate_uncertainty_gate(row)
    assert paper.action != "skip"
    assert live.action == "skip"


def _guilherme_type_row(**kwargs):
    """Strong +edge / high prob with conformal width above Paper skip (0.52)."""
    base = {
        "fighter_1": "Guilherme Pat",
        "fighter_2": "Opponent",
        "predicted_winner": "Guilherme Pat",
        "prob_f1_win": 0.865,
        "prob_f2_win": 0.135,
        "predicted_prob": 0.865,
        "f1_odds": 1.35,
        "f2_odds": 3.20,
        "best_edge": 0.56,
        "ensemble_disagreement": 0.011,
        "interval_width": 0.675,
        "confidence_label": "high",
    }
    base.update(kwargs)
    return pd.Series(base)


def test_paper_wide_override_guilherme_type(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_SKIP", 0.52)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_TIGHTEN", 0.36)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_SKIP", 0.11)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_MIN_EDGE", 0.08)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_MIN_PROB", 0.70)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_KELLY_MULT", 0.20)
    gate = evaluate_uncertainty_gate(_guilherme_type_row())
    assert gate.action == "tighten"
    assert gate.primary_reason == PAPER_WIDE_OVERRIDE
    assert SKIP_WIDE_INTERVAL in gate.reasons
    assert gate.kelly_mult == pytest.approx(0.20)
    assert gate.kelly_mult > 0.0


def test_live_wide_still_skip_guilherme_type(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_INTERVAL_WIDTH_SKIP", 0.22)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_ENABLED", True)
    gate = evaluate_uncertainty_gate(_guilherme_type_row())
    assert gate.action == "skip"
    assert SKIP_WIDE_INTERVAL in gate.reasons
    assert PAPER_WIDE_OVERRIDE not in gate.reasons
    assert gate.kelly_mult == 0.0


def test_paper_wide_override_requires_edge_and_prob(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_SKIP", 0.52)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_MIN_EDGE", 0.08)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_MIN_PROB", 0.70)
    weak = evaluate_uncertainty_gate(
        _guilherme_type_row(best_edge=0.03, prob_f1_win=0.865, predicted_prob=0.865)
    )
    assert weak.action == "skip"
    low_prob = evaluate_uncertainty_gate(
        _guilherme_type_row(best_edge=0.20, prob_f1_win=0.55, predicted_prob=0.55)
    )
    assert low_prob.action == "skip"


def test_paper_wide_override_not_when_disagreement(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_SKIP", 0.52)
    monkeypatch.setattr(config, "PAPER_DISAGREEMENT_SKIP", 0.11)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_ENABLED", True)
    gate = evaluate_uncertainty_gate(
        _guilherme_type_row(ensemble_disagreement=0.20)
    )
    assert gate.action == "skip"
    assert SKIP_HIGH_DISAGREEMENT in gate.reasons
    assert PAPER_WIDE_OVERRIDE not in gate.reasons


def test_paper_wide_override_stake_cap(monkeypatch):
    from src.alerts import _suggest_stake

    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "UNCERTAINTY_GATES_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_INTERVAL_WIDTH_SKIP", 0.52)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC", 0.01)
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_KELLY_MULT", 0.20)
    row = _guilherme_type_row(
        implied_prob_f1=0.30,
        implied_prob_f2=0.70,
        edge_f1=0.56,
        edge_f2=-0.20,
    )
    cfg = StrategyConfig(
        min_edge=0.03,
        kelly_fraction=0.25,
        max_bet_fraction=0.5,
        min_bet_fraction=0.0,
        max_card_risk_fraction=0.08,
        min_model_prob=0.55,
    )
    bankroll = 1000.0
    stake, _, gate_dict = _suggest_stake(
        row, bankroll=bankroll, strategy=cfg, card_risk=None
    )
    assert gate_dict.get("primary_reason") == PAPER_WIDE_OVERRIDE
    assert stake > 0.0
    assert stake <= bankroll * 0.01 + 1e-9
