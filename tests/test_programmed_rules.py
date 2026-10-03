"""Signed programmed matchup rules (log-odds nudge, capped, order-invariant)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.fight_brief import build_fight_brief
from src.fight_context import build_fight_context, format_fight_context_lines
from src.predictor import apply_style_calibration, compute_style_matchup_bonus
from src.programmed_rules import (
    compute_programmed_bonus,
    evaluate_programmed_rules,
    format_programmed_line,
    rules_supporting_pick,
)


def _bonus(row: dict) -> float:
    return compute_programmed_bonus(row)


def test_style_rules_still_positive_and_capped():
    row = {
        "striker_vs_grappler": 1.0,
        "striker_score_diff": 0.2,
        "southpaw_advantage": 0.08,
        "style_clash": 1.0,
        "grappler_score_diff": -0.1,
        "stance_matchup": 1.0,
    }
    bonus = compute_style_matchup_bonus(row)
    assert bonus > 0
    assert bonus <= config.STYLE_BONUS_MAX
    # Stance mismatch alone must not add an unsigned fighter-1 bump.
    # Southpaw +0.04, striker path +0.04, grappler path -0.02 → +0.06, then cap.
    assert bonus == pytest.approx(config.STYLE_BONUS_MAX)


def test_southpaw_is_signed_not_card_order():
    southpaw_listed_first = _bonus(
        {"southpaw_advantage": 0.08, "stance_matchup": 1.0}
    )
    orthodox_listed_first = _bonus(
        {"southpaw_advantage": -0.08, "stance_matchup": 1.0}
    )
    assert southpaw_listed_first == pytest.approx(0.04)
    assert orthodox_listed_first == pytest.approx(-0.04)
    assert southpaw_listed_first == pytest.approx(-orthodox_listed_first)


def test_reach_lands_requires_accuracy_same_direction():
    lands = _bonus(
        {
            "reach_diff": 4.0,
            "striking_acc_diff": 0.05,
            "sig_strikes_per_min_diff": 0.8,
        }
    )
    assert lands == pytest.approx(0.012 + 0.006)
    disagrees = _bonus({"reach_diff": 4.0, "striking_acc_diff": -0.05})
    assert disagrees == pytest.approx(0.0)
    short_reach = _bonus({"reach_diff": 1.5, "striking_acc_diff": 0.05})
    assert short_reach == pytest.approx(0.0)


def test_length_stack_same_direction_only():
    stacked = _bonus({"reach_diff": 3.0, "height_diff": 2.5})
    assert stacked == pytest.approx(0.008)
    split = _bonus({"reach_diff": 3.0, "height_diff": -2.5})
    assert split == pytest.approx(0.0)


def test_short_notice_penalizes_and_shrinks_when_record_is_good():
    taxed = _bonus({"short_notice_flag_diff": 1.0})
    proven = _bonus(
        {"short_notice_flag_diff": 1.0, "short_notice_perf_diff": 0.20}
    )
    assert taxed == pytest.approx(-0.016)
    assert proven == pytest.approx(-0.016 * 0.4)
    # Year-scale flag is used when present; it does not stack on the 10-day flag.
    both = _bonus(
        {"hv_short_notice_flag_diff": 1.0, "short_notice_flag_diff": 1.0}
    )
    assert both == pytest.approx(-0.016)


def test_layoff_uses_stricter_flag_without_stacking():
    half_year = _bonus({"long_layoff_flag_diff": 1.0})
    year = _bonus(
        {"long_layoff_flag_diff": 1.0, "hv_long_layoff_flag_diff": 1.0}
    )
    assert half_year == pytest.approx(-0.010)
    assert year == pytest.approx(-0.014)
    proven = _bonus(
        {"hv_long_layoff_flag_diff": 1.0, "long_layoff_perf_diff": 0.10}
    )
    assert proven == pytest.approx(-0.014 * 0.4)


def test_new_division_and_past_peak_are_signed():
    assert _bonus({"first_fight_new_wc_flag_diff": 1.0}) == pytest.approx(-0.012)
    assert _bonus({"first_fight_new_wc_flag_diff": -1.0}) == pytest.approx(0.012)
    assert _bonus({"division_age_adj_diff": 4.0}) == pytest.approx(-0.010)
    assert _bonus({"division_age_adj_diff": 1.5}) == pytest.approx(0.0)


def test_chin_tax_grows_when_opponent_has_power():
    base = _bonus({"ko_losses_career_flag_diff": 1.0})
    powered = _bonus(
        {"ko_losses_career_flag_diff": 1.0, "ko_rate_diff": -0.20}
    )
    same_side_power = _bonus(
        {"ko_losses_career_flag_diff": 1.0, "ko_rate_diff": 0.20}
    )
    assert base == pytest.approx(-0.010)
    assert powered == pytest.approx(-0.016)
    assert same_side_power == pytest.approx(-0.010)


def test_wrestling_path_blocked_by_takedown_defense_wall():
    open_path = _bonus(
        {
            "style_clash": 1.0,
            "grappler_score_diff": 0.2,
            "takedown_acc_diff": 0.15,
            "td_defense_diff": 0.0,
        }
    )
    walled = _bonus(
        {
            "style_clash": 1.0,
            "grappler_score_diff": 0.2,
            "takedown_acc_diff": 0.15,
            "td_defense_diff": -0.20,
        }
    )
    assert open_path == pytest.approx(0.02 + 0.010)
    assert walled == pytest.approx(0.02)


def test_form_and_experience_need_agreement():
    streak = _bonus({"last5_winrate_diff": 0.6, "momentum_diff": 0.3})
    assert streak == pytest.approx(0.008)
    split = _bonus({"last5_winrate_diff": 0.6, "momentum_diff": -0.3})
    assert split == pytest.approx(0.0)
    veteran = _bonus({"experience_diff": 10.0, "last5_winrate_diff": 0.4})
    assert veteran == pytest.approx(0.008)
    veteran_cold = _bonus({"experience_diff": 10.0, "last5_winrate_diff": -0.4})
    assert veteran_cold == pytest.approx(0.0)


def test_missing_and_nan_contribute_nothing():
    assert _bonus({}) == pytest.approx(0.0)
    assert _bonus({"reach_diff": float("nan"), "striking_acc_diff": None}) == pytest.approx(
        0.0
    )
    assert evaluate_programmed_rules(None) == (0.0, [])


def test_many_agreeing_rules_stay_inside_cap():
    row = {
        "southpaw_advantage": 0.08,
        "striker_vs_grappler": 1.0,
        "striker_score_diff": 0.3,
        "style_clash": 1.0,
        "grappler_score_diff": 0.2,
        "reach_diff": 5.0,
        "height_diff": 3.0,
        "striking_acc_diff": 0.05,
        "sig_strikes_per_min_diff": 1.0,
        "last5_winrate_diff": 0.6,
        "momentum_diff": 0.4,
        "experience_diff": 12.0,
    }
    total, hits = evaluate_programmed_rules(row)
    assert total == pytest.approx(config.STYLE_BONUS_MAX)
    assert hits
    assert sum(hit.bonus for hit in hits) == pytest.approx(total)


def test_mirroring_signed_diffs_negates_bonus():
    row = {
        "southpaw_advantage": 0.08,
        "stance_matchup": 1.0,
        "striker_vs_grappler": 1.0,
        "striker_score_diff": 0.25,
        "style_clash": 1.0,
        "grappler_score_diff": 0.15,
        "reach_diff": 4.0,
        "height_diff": 2.0,
        "striking_acc_diff": 0.04,
        "short_notice_flag_diff": -1.0,
        "ko_losses_career_flag_diff": -1.0,
        "ko_rate_diff": 0.2,
        "division_age_adj_diff": -4.0,
        "first_fight_new_wc_flag_diff": -1.0,
        "last5_winrate_diff": 0.5,
        "momentum_diff": 0.3,
        "experience_diff": 9.0,
        "takedown_acc_diff": 0.12,
    }
    mirrored = {}
    unsigned = {"stance_matchup", "striker_vs_grappler", "style_clash"}
    for key, value in row.items():
        mirrored[key] = value if key in unsigned else -float(value)
    assert _bonus(row) == pytest.approx(-_bonus(mirrored))


def test_calibration_moves_probability_with_the_bonus():
    frame = pd.DataFrame(
        [
            {
                "reach_diff": 4.0,
                "striking_acc_diff": 0.05,
                "sig_strikes_per_min_diff": 1.0,
            }
        ]
    )
    adjusted, bonuses = apply_style_calibration(frame, np.array([0.55]))
    assert bonuses[0] > 0
    assert adjusted[0] > 0.55


def test_context_and_brief_name_the_rule():
    row = {
        "fighter_1": "Longer",
        "fighter_2": "Shorter",
        "predicted_winner": "Longer",
        "predicted_prob": 0.62,
        "reach_diff": 4.0,
        "striking_acc_diff": 0.05,
        "odds_matched": False,
    }
    ctx = build_fight_context(row)
    assert "Programmed" in ctx["programmed"]
    assert "Longer" in ctx["programmed"]
    assert any("Programmed" in line for line in format_fight_context_lines(ctx))
    assert format_programmed_line(row)
    assert "Reach that lands" in rules_supporting_pick(pd.Series(row))
    brief = build_fight_brief(pd.Series(row), edge_pct=4.0)
    assert "Reach that lands" in brief
