"""Card-budget % stake allocation (conf/odds strength; sum ≤ 100%)."""

from __future__ import annotations

from src.strategy import (
    allocate_card_budget_pct,
    allocate_alerts_card_budget_pct,
    compute_ticket_strength,
    format_stake_pct_dollars,
    ticket_allocation_weight,
)


def test_allocate_pct_at_most_100():
    tickets = [
        {"edge": 0.10, "prob": 0.72, "confidence": "high", "decimal_odds": 1.70, "uncertainty_action": "allow"},
        {"edge": 0.07, "prob": 0.68, "confidence": "medium", "decimal_odds": 1.85, "uncertainty_action": "allow"},
        {"edge": 0.05, "prob": 0.70, "confidence": "high", "decimal_odds": 1.60, "uncertainty_action": "tighten", "is_parlay": True, "n_legs": 2},
    ]
    out = allocate_card_budget_pct(tickets, pool_usd=100.0, profile="paper")
    assert sum(t["stake_pct"] for t in out) <= 100.0 + 0.05
    assert sum(t["suggested_stake"] for t in out) <= 100.0 + 0.05
    assert all("strength_score" in t for t in out)
    assert all(t.get("sizing_mode") == "conf_odds" for t in out)


def test_weak_tickets_leave_residual():
    tickets = [
        {"edge": 0.04, "prob": 0.62, "confidence": "low", "decimal_odds": 1.95, "uncertainty_action": "tighten"},
        {"edge": 0.03, "prob": 0.60, "confidence": "low", "decimal_odds": 2.10, "uncertainty_action": "tighten"},
    ]
    out = allocate_card_budget_pct(tickets, pool_usd=100.0, profile="paper", inplace=False)
    assert sum(float(t["stake_pct"]) for t in out) < 95.0


def test_missing_odds_no_inflate():
    tickets = [
        {"edge": 0.12, "prob": 0.75, "confidence": "high", "decimal_odds": 1.55, "uncertainty_action": "allow"},
        {"edge": 0.15, "prob": 0.80, "confidence": "high", "uncertainty_action": "allow"},  # no odds
    ]
    out = allocate_card_budget_pct([dict(t) for t in tickets], 100.0, profile="paper", inplace=False)
    missing = [t for t in out if t.get("decimal_odds") is None and not t.get("odds")]
    # Second ticket has no odds → strength 0 / fail-closed
    assert float(out[1]["strength_score"]) == 0.0
    assert float(out[1]["stake_pct"]) == 0.0
    assert out[1].get("sizing_fail_closed") == "missing_odds"
    assert float(out[0]["stake_pct"]) > 0
    assert sum(float(t["stake_pct"]) for t in out) <= 100.0 + 0.05
    assert missing or out[1].get("sizing_no_inflate")


def test_live_allocation_less_aggressive_than_paper():
    tickets = [
        {"edge": 0.12, "prob": 0.75, "confidence": "high", "decimal_odds": 1.55, "uncertainty_action": "allow"},
        {"edge": 0.06, "prob": 0.70, "confidence": "medium", "decimal_odds": 1.80, "uncertainty_action": "allow"},
        {"edge": 0.05, "prob": 0.68, "confidence": "medium", "decimal_odds": 1.90, "uncertainty_action": "allow"},
    ]
    paper = allocate_card_budget_pct([dict(t) for t in tickets], 100.0, profile="paper")
    live = allocate_card_budget_pct([dict(t) for t in tickets], 100.0, profile="live")
    paper_top = max(t["stake_pct"] for t in paper)
    live_top = max(t["stake_pct"] for t in live)
    paper_sum = sum(t["stake_pct"] for t in paper)
    live_sum = sum(t["stake_pct"] for t in live)
    assert paper_top >= live_top - 0.5
    assert paper_sum + 1e-6 >= live_sum


def test_alerts_allocator_splits_singles_parlays_props():
    alerts = {
        "singles": [
            {"fight": "A vs B", "pick": "A", "edge": 0.09, "prob": 0.71, "confidence": "high", "decimal_odds": 1.65},
        ],
        "parlays": [
            {
                "picks": "A + C",
                "n_legs": 2,
                "combined_prob": 0.48,
                "combined_odds": 3.2,
                "expected_value": 0.2,
                "edge": 0.07,
                "min_leg_edge": 0.07,
            }
        ],
    }
    props = [
        {
            "prop_key": "over_1_5_rounds",
            "fight": "X vs Y",
            "edge": 0.06,
            "prob": 0.80,
            "label": "Over 1.5",
            "decimal_odds": 1.40,
        }
    ]
    out = allocate_alerts_card_budget_pct(alerts, 50.0, profile="paper", prop_singles=props)
    assert out["stake_allocation"]["sum_pct"] <= 100.0 + 0.05
    assert out["stake_allocation"].get("sizing") == "conf_odds"
    assert len(out["singles"]) + len(out["parlays"]) + len(out["prop_singles"]) == 3
    assert sum(t["suggested_stake"] for t in out["singles"] + out["parlays"] + out["prop_singles"]) <= 50.0 + 0.05


def test_format_stake_pct_dollars():
    txt = format_stake_pct_dollars({"stake_pct": 38.0, "suggested_stake": 4.56})
    assert txt == "38% · $4.56"


def test_stronger_edge_gets_more_weight():
    strong = {"edge": 0.12, "prob": 0.74, "confidence": "high", "decimal_odds": 1.55, "uncertainty_action": "allow"}
    weak = {"edge": 0.05, "prob": 0.68, "confidence": "medium", "decimal_odds": 1.90, "uncertainty_action": "tighten"}
    assert ticket_allocation_weight(strong, live=False) > ticket_allocation_weight(weak, live=False)
    assert compute_ticket_strength(strong, live=False)["strength"] > compute_ticket_strength(weak, live=False)["strength"]
