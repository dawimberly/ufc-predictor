"""Blue must only come from HA clears — never invent from leftover stakes."""

from __future__ import annotations

from src.bet_tiers import TIER_BLUE, TIER_YELLOW, action_label_for_bet
from src.grok_analysis import build_best_bets_briefing, collect_card_analysis_inputs


def test_synthetic_prop_never_blue_even_with_display_stake() -> None:
    from src.bet_tiers import TIER_BLUE, TIER_SKY_BLUE, classify_prop_bet_tier, rank_prop_bet_tiers

    prop = {
        "fight_id": "f1",
        "fight": "Magny vs Brahimaj",
        "prop_key": "over_1_5_rounds",
        "prop_short": "Over 1.5 Rounds",
        "prob": 0.82,
        "edge": 0.15,
        "odds_source": "synthetic",
        "strict_qualified": False,
        "suggested_stake": 18.0,
        "stake_pct": 40.0,
        "market_type": "prop",
    }
    tier, reason = classify_prop_bet_tier(prop, debug=False)
    assert tier not in {TIER_BLUE, TIER_SKY_BLUE}, (tier, reason)
    buckets = rank_prop_bet_tiers([prop], limit_per_tier=4)
    assert buckets[TIER_BLUE] == []
    assert buckets[TIER_SKY_BLUE] == []
    fun = buckets["green"] or buckets["yellow"]
    assert fun
    assert float(fun[0].get("stake_usd") or 0) == 0.0


def test_live_relaxed_mybookie_prop_not_blue_from_stake() -> None:
    from src.bet_tiers import TIER_BLUE, TIER_SKY_BLUE, classify_prop_bet_tier

    prop = {
        "fight_id": "f1",
        "fight": "Islam vs Garry",
        "prop_key": "over_1_5_rounds",
        "prob": 0.70,
        "edge": 0.08,
        "odds_source": "live",
        "strict_qualified": False,
        "suggested_stake": 12.0,
        "stake_pct": 25.0,
        "book": "MyBookie",
        "market_type": "prop",
    }
    tier, reason = classify_prop_bet_tier(prop, debug=False)
    assert tier not in {TIER_BLUE, TIER_SKY_BLUE}, (tier, reason)


def test_action_label_does_not_invent_blue_from_stake_alone() -> None:
    label = action_label_for_bet(
        {
            "pick": "Islam Makhachev",
            "stake_usd": 25.0,
            "stake_pct": 40.0,
            # no bet_tier — Overview/Ollama must not say BET THIS
        }
    )
    assert "BET THIS" not in label
    assert "FUN ONLY" in label or "CAUTION" in label or "SKIP" in label or "DO NOT" in label


def test_best_bets_briefing_does_not_force_blue() -> None:
    result = {
        "event": "UFC Test",
        "bet_slip": [
            {
                "side": "Alice over Bob",
                "pick": "Alice",
                "fight": "Alice vs Bob",
                "stake_usd": 10.0,
                "stake_pct": 20.0,
                "edge": 0.1,
                "prob": 0.7,
                # Not advisory, has stake — old code force-painted Blue
            }
        ],
    }
    text = build_best_bets_briefing(result)
    assert "BET THIS $10" not in text
    assert "Alice (BET THIS" not in text
    assert "NONE" in text or "no decent" in text.lower() or "Sized bankroll = $0" in text or "Blue) — none" in text


def test_mybookie_over_15_suspect_edge_never_blue() -> None:
    """Donte Johnson vs McConico Over 1.5 @ MyBookie +26.3% must not be BET THIS."""
    from src.bet_tiers import (
        TIER_BLUE,
        TIER_SKY_BLUE,
        action_label_for_bet,
        classify_prop_bet_tier,
        rank_prop_bet_tiers,
    )
    from src.grok_analysis import collect_card_analysis_inputs
    from src.strategy import attach_prop_stakes, prop_may_receive_ha_stake

    prop = {
        "fight_id": "donte-mcconico",
        "fight": "Donte Johnson vs Eric McConico",
        "prop_key": "over_1_5_rounds",
        "prop_short": "Over 1.5 Rounds",
        "label": "Over 1.5 Rounds (Donte Johnson vs Eric McConico)",
        "prob": 0.82,
        "edge": 0.263,
        "edge_pct": 26.3,
        "odds_source": "live",
        "strict_qualified": True,
        "suggested_stake": 7.17,
        "stake_usd": 7.17,
        "stake_pct": 40.0,
        "decimal_odds": 1.45,
        "book": "MyBookie",
        "market_type": "prop",
    }
    assert prop_may_receive_ha_stake(prop) is False
    tier, reason = classify_prop_bet_tier(prop, debug=False)
    assert tier not in {TIER_BLUE, TIER_SKY_BLUE}, (tier, reason)
    assert reason == "suspect_edge"
    assert "BET THIS" not in action_label_for_bet({**prop, "bet_tier": tier})

    buckets = rank_prop_bet_tiers([prop], limit_per_tier=4)
    assert buckets[TIER_BLUE] == []
    assert buckets[TIER_SKY_BLUE] == []

    sized = attach_prop_stakes(
        [prop],
        {"total_bankroll": 75, "card_budget": 12, "use_mybookie": True},
        "MyBookie",
        profile="paper",
    )
    assert all(float(s.get("suggested_stake") or 0) == 0.0 for s in sized)

    books = {
        "MyBookie": {
            "alerts": {"singles": [], "prop_singles": [prop]},
            "props": {"singles": [prop]},
            "predictions": None,
        },
        "Overview": {"alerts": {"singles": []}, "predictions": None},
    }
    out = collect_card_analysis_inputs(
        books,
        {"total_bankroll": 75, "card_budget": 12, "use_mybookie": True},
        event_label="UFC Fight Night",
    )
    blues = [
        t
        for t in (out.get("tickets") or [])
        if str(t.get("bet_tier") or "") in {TIER_BLUE, TIER_SKY_BLUE}
        and not t.get("advisory")
    ]
    assert blues == []
    assert out.get("n_actionable") == 0


def test_collect_inputs_zeros_unclear_overview_stakes() -> None:
    books = {
        "Odds API": {
            "alerts": {"singles": []},  # nothing HA-cleared
            "predictions": None,
        },
        "Overview": {
            "alerts": {"singles": []},
            "predictions": None,
        },
    }
    # Monkeypatch overview aggregator via books that yield empty alerts —
    # collect should not invent Blue tickets.
    out = collect_card_analysis_inputs(
        books,
        {"total_bankroll": 100, "card_budget": 12},
        event_label="UFC Test",
    )
    blues = [
        t
        for t in (out.get("tickets") or [])
        if str(t.get("bet_tier") or "") == TIER_BLUE and not t.get("advisory")
    ]
    assert blues == []
    assert out.get("n_actionable") == 0
