"""Unit tests for fight-row color tiers (math + decision, not vibes)."""

from __future__ import annotations

from src.bet_tiers import TIER_BLUE, TIER_GREEN, TIER_RED, TIER_YELLOW, classify_bet_tier


def test_guilherme_skip_wide_huge_edge_is_green() -> None:
    tier, reason = classify_bet_tier(
        None,
        status="SKIP:wide",
        edge=0.56,
        model_prob=0.67,
        pick="Guilherme Pat",
        debug=False,
    )
    assert tier == TIER_GREEN, (tier, reason)
    assert "skip" in reason


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
