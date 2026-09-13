"""Tests for controversy labeling / ref catalog."""

from __future__ import annotations

import pytest

from src.controversy import (
    build_and_save_catalog,
    is_controversial_method,
    method_controversy_kinds,
    referee_controversy_stats,
)


def test_method_kinds_split_dq_doctor():
    assert "split_decision" in method_controversy_kinds("Decision - Split")
    assert "split_decision" in method_controversy_kinds("S-DEC")
    assert "dq" in method_controversy_kinds("DQ")
    assert "doctor_stoppage" in method_controversy_kinds("TKO - Doctor's Stoppage")
    assert not is_controversial_method("KO/TKO")
    assert not is_controversial_method("Decision - Unanimous")


def test_referee_stats_and_catalog_smoke():
    from src.controversy import GRECO_RESULTS

    if not GRECO_RESULTS.is_file():
        pytest.skip(f"missing Greco results: {GRECO_RESULTS}")
    stats = referee_controversy_stats(min_bouts=40)
    assert not stats.empty
    assert "z_vs_league" in stats.columns
    cat = build_and_save_catalog(min_bouts=40)
    assert "flagged_referees" in cat
    assert "watchlist_referees" in cat
    assert "method_rules" in cat
