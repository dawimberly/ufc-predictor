"""Prop reliability ranking helpers."""

from __future__ import annotations

import pandas as pd

from src.prop_reliability import _recommend, _reliability_score, analyze_prop_reliability


def test_recommend_use_strong():
    row = {
        "n": 5000,
        "n_ge_70": 2000,
        "n_ge_75": 800,
        "n_ge_80": 200,
        "hit_rate_ge_70": 0.74,
        "hit_rate_ge_75": 0.78,
        "hit_rate_ge_80": 0.82,
        "calibration_gap": 0.03,
        "roi_edge_ge_5pct": 0.02,
        "n_bets_edge_ge_5pct": 100,
    }
    assert _recommend(row) == "Use"


def test_recommend_avoid_weak_high_conf():
    row = {
        "n": 5000,
        "n_ge_70": 500,
        "n_ge_75": 100,
        "n_ge_80": 50,
        "hit_rate_ge_70": 0.55,
        "hit_rate_ge_75": 0.52,
        "hit_rate_ge_80": 0.48,
        "calibration_gap": 0.12,
        "roi_edge_ge_5pct": -0.25,
        "n_bets_edge_ge_5pct": 80,
    }
    assert _recommend(row) == "Avoid for now"


def test_reliability_score_orders_use_above_avoid():
    use = {
        "hit_rate_ge_80": 0.85,
        "hit_rate_ge_75": 0.80,
        "hit_rate_ge_70": 0.75,
        "n_ge_80": 100,
        "n_ge_75": 200,
        "calibration_gap": 0.02,
        "roi_edge_ge_5pct": 0.05,
        "recommendation": "Use",
    }
    avoid = {
        "hit_rate_ge_80": 0.45,
        "hit_rate_ge_75": 0.50,
        "hit_rate_ge_70": 0.55,
        "n_ge_80": 100,
        "n_ge_75": 200,
        "calibration_gap": 0.20,
        "roi_edge_ge_5pct": -0.20,
        "recommendation": "Avoid for now",
    }
    assert _reliability_score(use) > _reliability_score(avoid)
