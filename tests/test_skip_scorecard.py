"""Skip-reason scorecard: tokenize, log, weekly rollup."""

from __future__ import annotations

import json

import config
from src.skip_scorecard import (
    EDGE_LEFT_CODES,
    NOISE_FILTER_CODES,
    SKIP_HIGH_DISAGREEMENT,
    SKIP_MIN_EDGE,
    SKIP_MISSING_UNCERTAINTY,
    SKIP_UNKNOWN,
    format_rollup_text,
    log_skip,
    rollup_skip_reasons,
    tokenize_skip_reason,
)


def test_tokenize_canonical_and_aliases():
    assert tokenize_skip_reason("high_disagreement") == SKIP_HIGH_DISAGREEMENT
    assert tokenize_skip_reason("min_edge") == SKIP_MIN_EDGE
    assert tokenize_skip_reason("missing_uncertainty") == SKIP_MISSING_UNCERTAINTY
    assert tokenize_skip_reason("below_tightened_min_edge") == "below_tightened_min_edge"
    assert tokenize_skip_reason("daily loss circuit breaker") == "circuit"
    assert tokenize_skip_reason("") == SKIP_UNKNOWN
    assert tokenize_skip_reason("something_weird_xyz") == SKIP_UNKNOWN


def test_log_skip_and_rollup(tmp_path, monkeypatch):
    journal = tmp_path / "bet_journal.csv"
    jsonl = tmp_path / "skip_scorecard.jsonl"
    out_json = tmp_path / "skip_scorecard.json"
    monkeypatch.setattr(config, "BET_JOURNAL_CSV", journal)
    monkeypatch.setattr(config, "SKIP_SCORECARD_JSONL", jsonl)
    monkeypatch.setattr(config, "SKIP_SCORECARD_JSON", out_json)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    log_skip("high_disagreement", event="UFC Test", fight="A vs B", pick="A", context="test")
    log_skip("min_edge", event="UFC Test", fight="C vs D", pick="C", edge_pct="+3.0", context="test")
    log_skip("min_edge", event="UFC Test", fight="E vs F", pick="E", context="test")
    log_skip("", event="UFC Test", fight="G vs H", pick="G", context="test")  # → unknown

    assert journal.is_file()
    jtext = journal.read_text(encoding="utf-8")
    assert "skip_reason" in jtext.splitlines()[0]
    assert "high_disagreement" in jtext

    lines = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 4

    rollup = rollup_skip_reasons(days=7, write_json=True)
    assert rollup["complete"] is True
    assert rollup["total_skips"] == 4
    by = {r["skip_reason"]: r for r in rollup["by_reason"]}
    assert by["min_edge"]["count"] == 2
    assert by["min_edge"]["pct"] == 50.0
    assert by["high_disagreement"]["bucket"] == "noise_filter"
    assert by["min_edge"]["bucket"] == "edge_left"
    assert by["unknown"]["count"] == 1
    assert out_json.is_file()

    text = format_rollup_text(rollup)
    assert "min_edge" in text
    assert "noise" in text.lower() or "edge" in text.lower()


def test_empty_rollup_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SKIP_SCORECARD_JSONL", tmp_path / "empty.jsonl")
    monkeypatch.setattr(config, "SKIP_SCORECARD_JSON", tmp_path / "empty.json")
    monkeypatch.setattr(config, "BET_JOURNAL_CSV", tmp_path / "no_journal.csv")
    rollup = rollup_skip_reasons(days=7, write_json=False)
    assert rollup["complete"] is False
    assert rollup["fail_closed"] is True
    assert rollup["total_skips"] == 0
    assert "fail-closed" in rollup["interpretation"].lower() or "no skip" in rollup["interpretation"].lower()


def test_noise_vs_edge_buckets():
    assert SKIP_HIGH_DISAGREEMENT in NOISE_FILTER_CODES
    assert SKIP_MIN_EDGE in EDGE_LEFT_CODES
    assert SKIP_MISSING_UNCERTAINTY in NOISE_FILTER_CODES
