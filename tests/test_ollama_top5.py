"""Ollama Top 5: actionable HA tickets + advisory fillers with warning."""

from __future__ import annotations

from src.grok_analysis import (
    TOP5_WARNING,
    build_grok_prompt,
    merge_ollama_reasons_into_slip,
    _ticket_to_slip_row,
)


def test_ticket_to_slip_row_includes_event_and_prop_line():
    raw = {
        "fight_id": "f1",
        "fight": "Islam Makhachev vs Ian Machado Garry",
        "event_name": "UFC 330",
        "prop_key": "over_4_5_rounds",
        "prop_short": "Over 4.5 Rounds",
        "market_type": "prop",
        "book": "MyBookie",
        "edge": 0.12,
        "prob": 0.71,
        "decimal_odds": 1.62,
        "suggested_stake": 0.0,
        "stake_pct": 0.0,
    }
    row = _ticket_to_slip_row(raw, rank=1, tier="advisory")
    assert row["event"] == "UFC 330"
    assert row["market"] == "Over 4.5 Rounds"


def test_prompt_lists_event_per_ticket():
    inputs = {
        "event": "UFC 330 + Fight Night",
        "profile": "paper",
        "bankroll": 100,
        "card_budget": 12,
        "total_stake_pct": 80,
        "total_stake_usd": 9.6,
        "n_actionable": 1,
        "n_advisory": 0,
        "top5_warning": TOP5_WARNING,
        "tickets": [
            {
                "id": "a",
                "event": "UFC 330",
                "side": "Islam",
                "market": "moneyline",
                "book": "Odds API",
                "stake_pct": 40,
                "stake_usd": 4.8,
                "prob": 0.7,
                "edge_pct": 8,
                "confidence": "high",
                "advisory": False,
            }
        ],
        "skipped": [],
    }
    prompt = build_grok_prompt(inputs)
    assert "event=UFC 330" in prompt
    assert "Name the event on every pick" in prompt


def test_ticket_to_slip_row_advisory_zeros_stake():
    raw = {
        "fight_id": "f1",
        "pick": "Alice",
        "pick_line": "Alice over Bob",
        "book": "Odds API",
        "edge": 0.12,
        "prob": 0.71,
        "confidence": "medium",
        "decimal_odds": 1.9,
        "suggested_stake": 5.0,
        "stake_pct": 40.0,
        "market_type": "moneyline",
    }
    row = _ticket_to_slip_row(raw, rank=2, tier="advisory")
    assert row["advisory"] is True
    assert row["stake_usd"] == 0.0
    assert row["stake_pct"] == 0.0
    assert row["rank"] == 2


def test_merge_keeps_advisory_zero_and_prefixes_reason():
    tickets = [
        {
            "id": "f1",
            "side": "Alice over Bob",
            "market": "moneyline",
            "book": "Odds API",
            "stake_pct": 0.0,
            "stake_usd": 0.0,
            "confidence": "high",
            "edge_pct": 9.0,
            "advisory": True,
            "tier": "advisory",
        }
    ]
    picks = [{"id": "f1", "reason": "strong style matchup", "conviction": "high"}]
    out = merge_ollama_reasons_into_slip(tickets, picks)
    assert len(out) == 1
    assert out[0]["stake_usd"] == 0.0
    assert out[0]["advisory"] is True
    assert str(out[0]["reason"]).upper().startswith("ADVISORY")


def test_prompt_includes_top5_warning_and_tiers():
    inputs = {
        "event": "UFC Test",
        "profile": "paper",
        "bankroll": 100,
        "card_budget": 12,
        "total_stake_pct": 80,
        "total_stake_usd": 9.6,
        "n_actionable": 2,
        "n_advisory": 3,
        "top5_warning": TOP5_WARNING,
        "tickets": [
            {
                "id": "a",
                "side": "A",
                "market": "moneyline",
                "book": "Odds API",
                "stake_pct": 40,
                "stake_usd": 4.8,
                "prob": 0.7,
                "edge_pct": 8,
                "confidence": "high",
                "advisory": False,
            },
            {
                "id": "b",
                "side": "B",
                "market": "moneyline",
                "book": "Odds API",
                "stake_pct": 0,
                "stake_usd": 0,
                "prob": 0.66,
                "edge_pct": 6,
                "confidence": "medium",
                "advisory": True,
            },
        ],
        "skipped": [],
    }
    prompt = build_grok_prompt(inputs)
    assert "FUN ONLY" in prompt
    assert "BET THIS" in prompt
    assert "Clarity" in prompt
    assert "event=" in prompt or "Name the event" in prompt
