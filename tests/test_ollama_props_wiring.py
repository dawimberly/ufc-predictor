"""Props path for Ollama analysis — books[X]['props']['singles'] → slip + tiers."""

from __future__ import annotations

from src.bet_tiers import (
    TIER_GREEN,
    TIER_YELLOW,
    classify_prop_bet_tier,
    collect_props_from_books,
    rank_prop_bet_tiers,
)
from src.grok_analysis import collect_card_analysis_inputs
from src.strategy import _overview_over_15_props


def _prop_row(**overrides):
    base = {
        "fight_id": "f1",
        "fight": "Alice vs Bob",
        "prop_key": "over_1_5_rounds",
        "prop_type": "Over 1.5",
        "label": "Over 1.5 Rounds",
        "prob": 0.66,
        "edge": 0.12,
        "odds": 1.70,
        "odds_source": "the_odds_api",
        "strict_qualified": False,
        "market_type": "prop",
    }
    base.update(overrides)
    return base


def test_overview_reads_props_singles_payload() -> None:
    books = {
        "Odds API": {
            "props": {
                "singles": [
                    _prop_row(strict_qualified=True, odds_source="the_odds_api"),
                ],
            }
        }
    }
    out = _overview_over_15_props(books, limit=2)
    assert len(out) == 1
    assert out[0]["prop_key"] == "over_1_5_rounds"
    assert out[0]["market_type"] == "prop"
    assert "Alice" in str(out[0].get("fight") or "")


def test_overview_skips_relaxed_synthetic_props() -> None:
    books = {
        "Odds API": {
            "props": {
                "singles": [
                    _prop_row(strict_qualified=False, odds_source="synthetic"),
                    _prop_row(
                        fight_id="f2",
                        fight="X vs Y",
                        strict_qualified=False,
                        odds_source="the_odds_api",
                    ),
                ],
            }
        }
    }
    assert _overview_over_15_props(books, limit=2) == []


def test_collect_props_from_books_dedupes() -> None:
    books = {
        "Odds API": {"props": {"singles": [_prop_row()]}},
        "MyBookie": {
            "props": {
                "singles": [
                    _prop_row(book="MyBookie"),
                    _prop_row(fight_id="f2", fight="Carol vs Dana", prob=0.61, edge=0.08),
                ]
            }
        },
    }
    rows = collect_props_from_books(books, limit=6)
    assert len(rows) == 2
    ids = {r["fight_id"] for r in rows}
    assert ids == {"f1", "f2"}


def test_prop_strong_skip_is_green() -> None:
    tier, reason = classify_prop_bet_tier(
        _prop_row(prob=0.66, edge=0.12, strict_qualified=False),
        debug=False,
    )
    assert tier == TIER_GREEN, (tier, reason)


def test_prop_thin_is_yellow() -> None:
    tier, reason = classify_prop_bet_tier(
        _prop_row(prob=0.55, edge=0.03, odds_source="synthetic"),
        debug=False,
    )
    assert tier == TIER_YELLOW, (tier, reason)


def test_collect_card_analysis_inputs_includes_props(monkeypatch) -> None:
    monkeypatch.setattr("config.ENABLE_PROPS", True, raising=False)
    books = {
        "Odds API": {
            "alerts": {"singles": [], "skipped": []},
            "props": {
                "singles": [
                    _prop_row(),  # relaxed live → fun advisory, not HA-sized
                    _prop_row(
                        fight_id="f2",
                        fight="Carol vs Dana",
                        prob=0.62,
                        edge=0.09,
                        odds_source="synthetic",
                        strict_qualified=False,
                    ),
                ]
            },
            "predictions": None,
        },
        "Overview": {"alerts": {"singles": [], "skipped": []}},
    }
    inputs = collect_card_analysis_inputs(
        books,
        {"total_bankroll": 100.0},
        event_label="Test Card",
        max_props=4,
    )
    props = list(inputs.get("props") or [])
    assert len(props) >= 1, inputs.get("tickets")
    assert any("Over 1.5" in str(t.get("market") or "") for t in props)
    # Relaxed/synthetic must not get real stake
    assert all(float(t.get("stake_usd") or 0) == 0 for t in props)
    assert inputs.get("prop_tiers")
    fun = inputs.get("fun_tiers") or {}
    assert fun.get("green") or fun.get("yellow") or (inputs.get("prop_tiers") or {}).get("green")


def test_rank_prop_bet_tiers_buckets() -> None:
    tiers = rank_prop_bet_tiers(
        [
            _prop_row(),
            _prop_row(fight_id="f2", fight="X vs Y", prob=0.54, edge=0.02),
        ],
        limit_per_tier=4,
    )
    assert tiers[TIER_GREEN] or tiers[TIER_YELLOW]
