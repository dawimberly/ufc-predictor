"""Tests for per-sleeve performance reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sleeve_stats import (
    compute_sleeve_stats,
    confidence_bucket,
    format_sleeve_stats_report,
    load_settled_sleeve_rows,
    model_prob_bucket,
    normalize_bet_type,
    rank_sleeves,
    run_sleeve_stats,
    uncertainty_level_from_row,
    write_sleeve_stats_csv,
)


def test_normalize_bet_type():
    assert normalize_bet_type(market_type="moneyline") == "single"
    assert normalize_bet_type(market_type="parlay", notes="2-leg") == "2leg_parlay"
    assert normalize_bet_type(prop_type="over_1_5_rounds") == "over_1_5"
    assert normalize_bet_type(prop_type="under_1_5_rounds") == "other_prop"


def test_model_prob_and_confidence_buckets():
    assert model_prob_bucket(0.50) == "lt_55"
    assert model_prob_bucket(0.70) == "65_75"
    assert model_prob_bucket(0.90) == "ge_85"
    assert model_prob_bucket(72) == "65_75"  # percent form
    assert confidence_bucket("High") == "high"
    assert confidence_bucket("med") == "medium"


def test_uncertainty_level_from_metrics():
    assert uncertainty_level_from_row({"uncertainty_level": "low"}) == "low"
    assert uncertainty_level_from_row({"uncertainty_action": "skip"}) == "high"
    assert uncertainty_level_from_row({"uncertainty_action": "tighten"}) == "medium"
    assert uncertainty_level_from_row({"uncertainty_action": "allow"}) == "low"


def test_compute_sleeve_stats_and_csv(tmp_path: Path, monkeypatch):
    bank = pd.DataFrame(
        [
            {
                "prediction_id": "a1",
                "status": "settled",
                "correct": "1",
                "pick_prob": "0.72",
                "odds": "1.90",
                "stake": "10",
                "pnl": "9",
                "weight_class": "Lightweight",
                "odds_bucket": "favorite",
                "confidence": "high",
                "market_type": "moneyline",
                "prop_type": "moneyline",
                "uncertainty_level": "low",
            },
            {
                "prediction_id": "a2",
                "status": "settled",
                "correct": "0",
                "pick_prob": "0.60",
                "odds": "2.50",
                "stake": "10",
                "pnl": "-10",
                "weight_class": "Lightweight",
                "odds_bucket": "mild_dog",
                "confidence": "medium",
                "market_type": "moneyline",
                "prop_type": "moneyline",
                "uncertainty_level": "medium",
            },
            {
                "prediction_id": "a3",
                "status": "settled",
                "correct": "1",
                "pick_prob": "0.80",
                "odds": "1.70",
                "stake": "10",
                "pnl": "7",
                "weight_class": "Welterweight",
                "odds_bucket": "favorite",
                "confidence": "high",
                "market_type": "moneyline",
                "prop_type": "over_1_5_rounds",
                "uncertainty_level": "low",
            },
            {
                "prediction_id": "a4",
                "status": "open",
                "correct": "",
                "pick_prob": "0.70",
                "odds": "1.90",
                "stake": "10",
                "pnl": "",
                "weight_class": "Lightweight",
                "odds_bucket": "favorite",
                "confidence": "high",
                "market_type": "moneyline",
                "prop_type": "moneyline",
                "uncertainty_level": "low",
            },
        ]
    )
    bank_path = tmp_path / "bank.csv"
    bank.to_csv(bank_path, index=False)
    monkeypatch.setattr(
        "src.sleeve_stats.config.BET_JOURNAL_CSV",
        tmp_path / "missing_journal.csv",
    )

    df = load_settled_sleeve_rows(bank_path=bank_path, include_journal=False)
    assert len(df) == 3
    assert set(df["bet_type"]) >= {"single", "over_1_5"}

    stats = compute_sleeve_stats(df)
    assert any(r["dimension"] == "bet_type" for r in stats)
    assert any(r["dimension"] == "weight_class" for r in stats)
    assert any(r["dimension"] == "uncertainty_level" for r in stats)

    # Lightweight sleeve: 2 bets, 1 hit
    lw = next(
        r for r in stats if r["dimension"] == "weight_class" and r["sleeve"] == "lightweight"
    )
    assert lw["n_bets"] == 2
    assert lw["hits"] == 1
    assert lw["hit_rate"] == pytest.approx(0.5)
    assert lw["avg_model_prob"] is not None
    assert lw["pnl"] is not None

    out = write_sleeve_stats_csv(stats, tmp_path / "sleeve_stats_20260725.csv")
    assert out.is_file()
    loaded = pd.read_csv(out)
    assert "hit_rate" in loaded.columns
    assert "roi" in loaded.columns
    assert "avg_model_prob" in loaded.columns

    report = run_sleeve_stats(
        csv_path=tmp_path / "sleeve_stats_out.csv",
        bank_path=bank_path,
        min_n_rank=1,
    )
    text = format_sleeve_stats_report(report)
    assert "SLEEVE PERFORMANCE" in text
    assert report["n_rows"] == 3
    top, bottom = rank_sleeves(stats, min_n=1, limit=2)
    assert top
    assert bottom
