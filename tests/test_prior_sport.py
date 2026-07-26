"""Tests for prior-sport background tiers."""

from __future__ import annotations

import pandas as pd

import config
from src.feature_engineering import build_matchup_features
from src.prior_sport import (
    WRESTLING_TIERS,
    parse_prior_sport_text,
    fill_history_from_prior_sport,
)


def test_wrestling_olympic_tier():
    bg = parse_prior_sport_text(
        "Olympic gold medalist freestyle wrestler; NCAA All-American",
        name="Gable Steveson",
        source="test",
    )
    assert bg.primary_base == "wrestling"
    assert bg.tiers["wrestling"] == WRESTLING_TIERS["international_olympic"]
    assert bg.base_level_tier == 1.0


def test_bjj_world_and_boxing_multi_base():
    bg = parse_prior_sport_text(
        "IBJJF World Champion black belt; also pro boxing contender with WBA title eliminator",
        name="Hybrid Fighter",
        source="test",
    )
    assert bg.tiers["bjj"] >= 0.9
    assert bg.tiers["boxing"] >= 0.7
    assert bg.multi_base == 1.0

def test_muay_thai_lumpinee():
    bg = parse_prior_sport_text(
        "Former Lumpinee Stadium Muay Thai champion",
        name="Thai Striker",
        source="test",
    )
    assert bg.primary_base == "muay_thai"
    assert bg.tiers["muay_thai"] >= 0.9


def test_unknown_fail_soft():
    bg = parse_prior_sport_text("", name="Unknown")
    assert bg.primary_base == "other"
    assert bg.base_level_tier == 0.0
    feats = bg.feature_dict()
    assert feats["base_other"] == 1.0
    assert feats["base_level_tier"] == 0.0


def test_matchup_base_features():
    f1 = {k: 0.0 for k in (
        "age", "height_in", "reach_in", "stance_orthodox", "stance_southpaw", "stance_switch",
        "win_rate", "sig_strike_acc", "td_acc", "sub_avg", "ko_rate", "last5_win_rate", "momentum",
        "sig_strikes_per_min", "td_defense", "control_time_per_min", "elo", "days_since_last_fight",
        "fight_count", "striker_score", "grappler_score", "similar_opp_win_rate", "sos_opp_win_rate",
        "avg_opp_elo", "short_notice_flag", "long_layoff_flag", "short_notice_win_rate",
        "long_layoff_win_rate", "kd_rate", "head_strike_pct", "body_strike_pct", "leg_strike_pct",
        "distance_strike_pct", "clinch_strike_pct", "ground_strike_pct", "power_proxy",
        "sherdog_win_rate", "sherdog_fight_count", "sherdog_finish_rate",
        "base_level_tier", "multi_base", "base_grappling", "base_striking",
        "base_wrestling", "base_bjj", "base_boxing", "base_muay_thai", "base_kickboxing",
        "base_sambo", "base_judo", "base_other",
    )}
    f2 = dict(f1)
    f1.update({"base_wrestling": 1.0, "base_grappling": 1.0, "base_level_tier": 1.0, "multi_base": 0.0})
    f2.update({"base_boxing": 1.0, "base_striking": 1.0, "base_level_tier": 0.25, "multi_base": 0.0})
    diffs = build_matchup_features(f1, f2)
    assert diffs["base_level_diff"] == 0.75
    assert diffs["base_family_clash"] == 1.0
    assert diffs["same_primary_base"] == 0.0


def test_fill_history_from_gym_notes(tmp_path, monkeypatch):
    monkeypatch.setattr("src.prior_sport.PRIOR_SPORT_CACHE", tmp_path / "prior.csv")
    # Seed gym index via monkeypatch
    monkeypatch.setattr(
        "src.prior_sport._gym_text_index",
        lambda: {"gable steveson": "Olympic wrestling base; UFC debut"},
    )
    monkeypatch.setattr(
        "src.prior_sport.load_prior_sport_profiles",
        lambda: pd.DataFrame(columns=["name", "base_level_tier"]),
    )
    history = pd.DataFrame(
        {
            "fighter": ["Gable Steveson", "Random Guy"],
            config.DATE_COLUMN: pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )
    out = fill_history_from_prior_sport(history)
    assert float(out.loc[0, "base_level_tier"]) >= 0.9
    assert out.loc[0, "primary_base"] == "wrestling"
    assert float(out.loc[1, "base_level_tier"]) == 0.0


def test_schema_includes_prior_sport():
    assert config.FEATURE_SCHEMA_VERSION >= 4
    assert "base_level_diff" in config.FEATURE_COLUMNS
    assert "base_family_clash" in config.FEATURE_COLUMNS
    assert "multi_base_flag_diff" in config.FEATURE_COLUMNS
