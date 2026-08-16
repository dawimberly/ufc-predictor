"""Fighter-photo desk: vision JSON parse, Over 1.5 fade, no invented Blue."""

from __future__ import annotations

import pandas as pd

from src.bet_tiers import (
    TIER_BLUE,
    TIER_SKY_BLUE,
    TIER_YELLOW,
    classify_bet_tier,
    classify_prop_bet_tier,
    demote_photo_caution_ticket,
)
from src.gym_data import gym_matchup_summary, shared_gym_caution
from src.photo_analysis import (
    fighter_names_from_record,
    parse_vision_payload,
    photo_over_15_blocks,
)
from src.strategy import prop_may_receive_ha_stake


def test_parse_both_finishers_sets_over_15_caution() -> None:
    raw = {
        "fighter_1": {
            "physique": "bulky",
            "finisher_look": "knockout_artist",
            "note": "thick puncher",
            "confidence": 0.8,
        },
        "fighter_2": {
            "physique": "athletic",
            "finisher_look": "knockout_artist",
            "note": "power in the still",
            "confidence": 0.7,
        },
        "size_mismatch": False,
        "both_early_finishers": True,
        "over_15_caution": True,
        "summary": "Two finishers in the photos",
    }
    out = parse_vision_payload(raw, fighter_1="Donte Johnson", fighter_2="McConico")
    assert out.both_early_finishers is True
    assert out.over_15_caution is True
    line = out.line()
    assert "Photos:" in line
    assert "Over 1.5" in line


def test_parse_infers_both_finishers_from_looks() -> None:
    raw = {
        "fighter_1": {"physique": "compact", "finisher_look": "power_puncher"},
        "fighter_2": {"physique": "bulky", "finisher_look": "finisher"},
        "summary": "Both look like KO guys",
    }
    out = parse_vision_payload(raw, fighter_1="A", fighter_2="B")
    assert out.both_early_finishers is True
    assert out.over_15_caution is True


def test_parse_empty_is_skipped_not_caution() -> None:
    out = parse_vision_payload("", fighter_1="A", fighter_2="B")
    assert out.source == "skipped"
    assert out.over_15_caution is False
    assert out.line() == ""


def test_fighter_names_from_fight_label() -> None:
    f1, f2 = fighter_names_from_record(
        {"fight": "Donte Johnson vs McConico", "prop_key": "over_1_5_rounds"}
    )
    assert f1 == "Donte Johnson"
    assert f2 == "McConico"


def test_photo_over_15_blocks_from_flag_only() -> None:
    prop = {
        "prop_key": "over_1_5_rounds",
        "fight": "Donte Johnson vs McConico",
        "photo_over_15_caution": True,
        "odds_source": "the_odds_api",
        "strict_qualified": True,
        "edge": 0.08,
        "prob": 0.66,
        "decimal_odds": 1.80,
    }
    assert photo_over_15_blocks(prop) is True
    assert prop_may_receive_ha_stake(prop) is False
    tier, reason = classify_prop_bet_tier(prop, debug=False)
    assert tier not in {TIER_BLUE, TIER_SKY_BLUE}, (tier, reason)
    assert reason == "photo_finishers"


def test_photo_over_15_does_not_block_without_cache_or_flag() -> None:
    prop = {
        "prop_key": "over_1_5_rounds",
        "fight": "Unknown Alpha vs Unknown Beta",
        "odds_source": "the_odds_api",
        "strict_qualified": True,
        "edge": 0.08,
        "prob": 0.72,
        "decimal_odds": 1.80,
        "suggested_stake": 5.0,
        "stake_pct": 8.0,
    }
    assert photo_over_15_blocks(prop) is False
    assert prop_may_receive_ha_stake(prop) is True


def test_leftover_stake_photo_caution_is_yellow_not_blue() -> None:
    ticket = {
        "prop_key": "over_1_5_rounds",
        "fight": "Donte Johnson vs McConico",
        "photo_over_15_caution": True,
        "odds_source": "the_odds_api",
        "strict_qualified": True,
        "edge": 0.08,
        "edge_pct": 8.0,
        "prob": 0.66,
        "suggested_stake": 7.17,
        "stake_pct": 12.0,
        "stake_usd": 7.17,
        "bet_tier": TIER_BLUE,
    }
    out = demote_photo_caution_ticket(ticket)
    assert out["bet_tier"] == TIER_YELLOW
    assert float(out["suggested_stake"] or 0) == 0.0
    tier, reason = classify_bet_tier(
        ticket,
        status="BET",
        edge=0.08,
        model_prob=0.66,
        stake_pct=12.0,
        stake_usd=7.17,
        debug=False,
    )
    assert tier == TIER_YELLOW
    assert reason == "photo_finishers"


def test_shared_gym_caution_mentions_camp_switch() -> None:
    note = shared_gym_caution("Xtreme Couture")
    assert "Xtreme Couture" in note
    assert "camps elsewhere" in note.lower() or "camp" in note.lower()
    profiles = pd.DataFrame(
        [
            {
                "fighter_name": "Dustin Stoltzfus",
                "gym": "Xtreme Couture",
                "location": "Las Vegas",
                "strengths": "grappling",
                "notes": "Usually XC; switched camps for this fight",
            },
            {
                "fighter_name": "Abdul-Malik",
                "gym": "Xtreme Couture",
                "location": "Las Vegas",
                "strengths": "wrestling",
                "notes": "",
            },
        ]
    )
    from src.data_loader import clean_fighter_name

    profiles["fighter_key"] = profiles["fighter_name"].map(clean_fighter_name)
    summary = gym_matchup_summary(
        "Dustin Stoltzfus", "Abdul-Malik", profiles=profiles
    )
    assert "Shared gym" in summary
    assert "camp switch" in summary.lower()


def test_fight_context_includes_photo_and_gym(monkeypatch) -> None:
    from src import fight_context as fc

    monkeypatch.setattr(
        "src.photo_analysis.format_photo_analysis_line",
        lambda *a, **k: "Photos: both look like early finishers - fade Over 1.5",
    )
    monkeypatch.setattr(
        "src.gym_data.gym_matchup_summary",
        lambda *a, **k: "Shared gym (Xtreme Couture) - camp familiarity",
    )
    ctx = fc.build_fight_context(
        {
            "fighter_1": "Stoltzfus",
            "fighter_2": "Abdul-Malik",
            "predicted_winner": "Abdul-Malik",
            "odds_matched": False,
        }
    )
    assert "photos" in ctx
    assert "Over 1.5" in ctx["photos"]
    assert "gym" in ctx
    assert "Xtreme Couture" in ctx["gym"]
    lines = fc.format_fight_context_lines(ctx)
    assert any(ln.startswith("Photos:") for ln in lines)
    assert any(ln.startswith("Camp:") for ln in lines)


def test_grok_prompt_includes_photo_notes() -> None:
    from src.grok_analysis import build_grok_prompt

    prompt = build_grok_prompt(
        {
            "event": "UFC 330",
            "profile": "paper",
            "bankroll": 75.0,
            "card_budget": 12.0,
            "total_stake_pct": 0.0,
            "total_stake_usd": 0.0,
            "n_actionable": 0,
            "n_advisory": 1,
            "tickets": [
                {
                    "id": "p1",
                    "side": "Johnson vs McConico — Over 1.5",
                    "market": "Over 1.5 Rounds",
                    "book": "MyBookie",
                    "stake_pct": 0.0,
                    "stake_usd": 0.0,
                    "prob": 0.66,
                    "edge_pct": 8.0,
                    "confidence": "medium",
                    "advisory": True,
                    "fun_bet": True,
                    "photo_note": "Photos: both look like early finishers - fade Over 1.5",
                    "photo_over_15_caution": True,
                }
            ],
            "photo_notes": "- Johnson vs McConico: Photos: fade Over 1.5",
            "recommended_parlays": [],
        }
    )
    assert "Photo desk" in prompt
    assert "PHOTO_FADE_OVER_15" in prompt
    assert "fade Over 1.5" in prompt
