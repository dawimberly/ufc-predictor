"""Unit tests for fight-row color tiers (math + decision, not vibes)."""

from __future__ import annotations

import config
from src.bet_tiers import (
    TIER_BLUE,
    TIER_GREEN,
    TIER_RED,
    TIER_SKY_BLUE,
    TIER_YELLOW,
    classify_bet_tier,
    format_tier_legend,
    is_sky_blue_ticket,
)


def test_guilherme_skip_wide_huge_edge_is_not_blue() -> None:
    # 56% "edge" is a scraper/odds glitch — caution, never BET THIS or FUN ONLY.
    tier, reason = classify_bet_tier(
        None,
        status="SKIP:wide",
        edge=0.56,
        model_prob=0.67,
        pick="Guilherme Pat",
        debug=False,
    )
    assert tier == TIER_YELLOW, (tier, reason)
    assert reason == "suspect_edge"


def test_negative_edge_skip_is_red() -> None:
    tier, reason = classify_bet_tier(
        None,
        status="SKIP:wide",
        edge=-0.08,
        model_prob=0.63,
        pick="Alexia Thainara",
        debug=False,
    )
    assert tier == TIER_RED, (tier, reason)
    assert reason == "negative_edge"


def test_stake_positive_is_blue() -> None:
    tier, reason = classify_bet_tier(
        None,
        status="BET",
        edge=0.08,
        model_prob=0.72,
        stake_pct=2.5,
        pick="Sized Fighter",
        debug=False,
    )
    assert tier == TIER_BLUE, (tier, reason)


def test_skip_never_blue_even_with_clears_gates() -> None:
    tier, reason = classify_bet_tier(
        None,
        clears_gates=True,
        status="SKIP:wide",
        edge=0.115,
        model_prob=0.55,
        pick="Bruno Lopes",
        debug=False,
    )
    assert tier != TIER_BLUE, (tier, reason)
    assert tier != TIER_SKY_BLUE, (tier, reason)
    assert tier == TIER_YELLOW, (tier, reason)


def test_gamrot_52_pct_skip_wide_is_yellow() -> None:
    # Displayed 52% (0.516–0.524) with thin-ish positive edge → yellow
    tier, reason = classify_bet_tier(
        None,
        status="SKIP:wide",
        edge=0.058,
        model_prob=0.516,
        pick="Mateusz Gamrot",
        debug=False,
    )
    assert tier == TIER_YELLOW, (tier, reason)


def test_diego_juliana_style_green() -> None:
    for pick, edge, prob in (
        ("Diego Ferreira", 0.193, 0.83),
        ("Juliana Miller", 0.145, 0.86),
    ):
        tier, reason = classify_bet_tier(
            None,
            status="SKIP:wide",
            edge=edge,
            model_prob=prob,
            pick=pick,
            debug=False,
        )
        assert tier == TIER_GREEN, (pick, tier, reason)


def test_no_odds_is_red() -> None:
    tier, reason = classify_bet_tier(
        None,
        status="SKIP:wide",
        edge=None,
        model_prob=0.81,
        pick="Yadier del Valle",
        debug=False,
    )
    assert tier == TIER_RED, (tier, reason)
    assert reason == "no_usable_odds"


def test_paper_wide_override_is_sky_blue(monkeypatch) -> None:
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    tier, reason = classify_bet_tier(
        None,
        status="1.00% paper_wide_override",
        edge=0.189,
        model_prob=0.833,
        stake_pct=1.0,
        uncertainty_reason="paper_wide_override",
        pick="Diego Ferreira",
        debug=False,
    )
    assert tier == TIER_SKY_BLUE, (tier, reason)
    assert reason == "paper_wide_override"


def test_book_table_kelly_status_sky_blue_without_row_stake(monkeypatch) -> None:
    """Book tabs pass Kelly text only; row has no stake_* — must still be sky blue."""
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    row = {
        "fighter_1": "Diego Ferreira",
        "fighter_2": "Billy Quarantillo",
        "predicted_winner": "Diego Ferreira",
        "prob_f1_win": 0.833,
        "prob_f2_win": 0.167,
        "edge_f1": 0.189,
        "edge_f2": -0.10,
        "best_edge": 0.189,
        "f1_odds": 1.40,
        "f2_odds": 3.10,
        "odds_matched": True,
    }
    tier, reason = classify_bet_tier(
        row,
        clears_gates=False,
        status="1.00% paper_wide_override",
        edge=0.189,
        model_prob=0.833,
        pick="Diego Ferreira",
        debug=False,
    )
    assert tier == TIER_SKY_BLUE, (tier, reason)
    assert reason == "paper_wide_override"


def test_live_never_sky_blue(monkeypatch) -> None:
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    assert not is_sky_blue_ticket(
        stake_pct=1.0,
        uncertainty_reason="paper_wide_override",
    )
    tier, reason = classify_bet_tier(
        None,
        status="1.00% paper_wide_override",
        edge=0.189,
        model_prob=0.833,
        stake_pct=1.0,
        uncertainty_reason="paper_wide_override",
        pick="Diego Ferreira",
        debug=False,
    )
    assert tier == TIER_BLUE, (tier, reason)


def test_sky_blue_requires_stake(monkeypatch) -> None:
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    assert not is_sky_blue_ticket(
        stake_pct=0.0,
        uncertainty_reason="paper_wide_override",
    )


def test_legend_mentions_sky_blue() -> None:
    text = format_tier_legend().lower()
    assert "sky blue" in text
    assert "paper" in text
    assert "bet this" in text
    assert "fun only" in text


def test_action_label_fun_vs_bet() -> None:
    from src.bet_tiers import action_label_for_bet, format_what_to_do_header

    fun = action_label_for_bet(
        {"bet_tier": "green", "pick": "Diego Ferreira", "fun_bet": True}
    )
    assert fun.startswith("FUN ONLY")
    assert "$0" in fun

    real = action_label_for_bet(
        {"bet_tier": "blue", "pick": "A", "stake_usd": 12.5, "stake_pct": 5}
    )
    assert real.startswith("BET THIS")
    assert "12.50" in real

    header = format_what_to_do_header(
        slip=[{"bet_tier": "green", "pick": "Diego Ferreira", "fun_bet": True}]
    )
    assert "NONE" in header
    assert "FUN ONLY" in header
    assert "Diego Ferreira" in header
    assert "passed gates" in header


def test_what_to_do_header_splits_ha_pass_from_failed_wide_ci() -> None:
    from src.bet_tiers import format_what_to_do_header

    ha_only = format_what_to_do_header(
        slip=[
            {
                "bet_tier": "blue",
                "pick": "Islam Makhachev",
                "stake_usd": 4.2,
                "advisory": False,
            }
        ]
    )
    assert "passed gates" in ha_only
    assert "Islam Makhachev" in ha_only
    assert "BET THIS" in ha_only
    assert "failed wide CI" not in ha_only

    sky_only = format_what_to_do_header(
        slip=[
            {
                "bet_tier": "sky_blue",
                "pick": "Diego Ferreira",
                "stake_usd": 0.75,
                "uncertainty_reason": "paper_wide_override",
            }
        ]
    )
    assert "WHAT TO BET (HA — passed gates): NONE" in sky_only
    assert "failed wide CI" in sky_only
    assert "Diego Ferreira" in sky_only
    assert "TINY PAPER BET" in sky_only

    both = format_what_to_do_header(
        slip=[
            {"bet_tier": "blue", "pick": "Islam Makhachev", "stake_usd": 4.2},
            {"bet_tier": "sky_blue", "pick": "Diego Ferreira", "stake_usd": 0.75},
        ]
    )
    assert "Islam Makhachev" in both
    assert "Diego Ferreira" in both
    assert "passed gates" in both
    assert "failed wide CI" in both
    assert both.index("Islam Makhachev") < both.index("Diego Ferreira")
