"""Live quote identity, freshness, and book provenance."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import config
from src.odds_providers.odds_fallback import _normalize_odds_frame
from src.odds_providers.prop_odds_common import lookup_prop_odds_row
from src.predictor import _lookup_odds_row, _names_match
from src.odds_providers.the_odds_api import _pick_best_totals_price
from src.quote_integrity import (
    classify_book_price,
    live_quote_block_reason,
    names_match_actionable,
    price_for_selection,
    read_quote_cache,
    select_moneyline_quote,
)
from src.strategy import allocate_card_budget_pct
from tests.test_audit_fail_closed import _fresh_quote, _strong_ticket


NOW = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)


def _quote_frame(**extra) -> dict:
    fields = _fresh_quote(
        quote_timestamp=(NOW - timedelta(minutes=5)).isoformat(),
        commence_time=(NOW + timedelta(days=2)).isoformat(),
    )
    fields.update(extra)
    return fields


def test_valid_fresh_quote_sizes_within_kelly_ceiling():
    ticket = _strong_ticket(**_quote_frame(), kelly_stake_usd=1.25, suggested_stake=40.0)
    out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
    stake = float(out[0]["suggested_stake"])
    assert 0.0 < stake <= 1.25 + 1e-9
    assert out[0].get("quote_block_reason") in ("", None)


def test_expired_cache_stays_zero_when_fetch_once_is_on(monkeypatch):
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)
    monkeypatch.setattr(config, "ODDS_CACHE_TTL_MINUTES", 20)
    ticket = _strong_ticket(
        **_quote_frame(quote_timestamp=(NOW - timedelta(hours=3)).isoformat()),
        kelly_stake_usd=20.0,
        suggested_stake=20.0,
    )
    out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
    assert float(out[0]["suggested_stake"]) == 0.0
    assert float(out[0]["kelly_stake_usd"]) == 0.0
    assert out[0]["quote_block_reason"] == "expired_quote"
    assert out[0]["bet_action"] == "NO BET"


def test_reading_cache_does_not_refresh_timestamp(tmp_path):
    path = tmp_path / "odds.csv"
    original = (
        "event_id,fighter_1,fighter_2,f1_odds,f2_odds,quote_timestamp,commence_time\n"
        "evt-old,Jon Jones,Stipe Miocic,1.4,2.8,2020-01-01T00:00:00+00:00,2020-02-01T00:00:00+00:00\n"
    )
    path.write_text(original, encoding="utf-8")
    loaded = read_quote_cache(path)
    assert str(loaded.loc[0, "quote_timestamp"]).startswith("2020-01-01")
    assert path.read_text(encoding="utf-8") == original


def test_wrong_event_rematch_is_not_attached_or_sized():
    market = pd.DataFrame(
        [
            {
                "event_id": "evt-old",
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "f1_odds": 1.40,
                "f2_odds": 3.20,
                "commence_time": "2024-11-16T00:00:00+00:00",
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
                "quote_timestamp": (NOW - timedelta(minutes=5)).isoformat(),
            }
        ]
    )
    selected = select_moneyline_quote(
        "Jon Jones",
        "Stipe Miocic",
        market,
        event_id="evt-new",
        event_date="2026-06-03",
    )
    assert selected["row"] is None
    assert selected["reason"] == "no_quote"
    assert _lookup_odds_row(
        "Jon Jones",
        "Stipe Miocic",
        market,
        event_id="evt-new",
        event_date="2026-06-03",
    ) is None
    ticket = _strong_ticket(
        **_quote_frame(
            fighter_1="Jon Jones",
            fighter_2="Stipe Miocic",
            quote_fighter_1="Jon Jones",
            quote_fighter_2="Stipe Miocic",
            pick="Jon Jones",
            fight="Jon Jones vs Stipe Miocic",
            card_event_id="evt-new",
            quote_event_id="evt-old",
        ),
        kelly_stake_usd=10.0,
    )
    out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
    assert float(out[0]["suggested_stake"]) == 0.0
    assert out[0]["quote_block_reason"] == "wrong_event"


def test_missing_malformed_and_future_timestamps_fail_closed():
    missing = _strong_ticket(**_quote_frame(quote_timestamp=None), kelly_stake_usd=8.0)
    malformed = _strong_ticket(**_quote_frame(quote_timestamp="not-a-time"), kelly_stake_usd=8.0)
    future = _strong_ticket(
        **_quote_frame(quote_timestamp=(NOW + timedelta(days=1)).isoformat()),
        kelly_stake_usd=8.0,
    )
    for ticket, reason in (
        (missing, "missing_timestamp"),
        (malformed, "invalid_timestamp"),
        (future, "future_timestamp"),
    ):
        assert live_quote_block_reason(ticket, now=NOW) == reason
        out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
        assert float(out[0]["suggested_stake"]) == 0.0
        assert float(out[0]["kelly_stake_usd"]) == 0.0


def test_already_started_event_stays_zero():
    ticket = _strong_ticket(
        **_quote_frame(commence_time=(NOW - timedelta(hours=1)).isoformat()),
        kelly_stake_usd=8.0,
        suggested_stake=8.0,
    )
    out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
    assert out[0]["quote_block_reason"] == "already_started"
    assert float(out[0]["suggested_stake"]) == 0.0


def test_same_surname_does_not_match_and_alias_and_reversed_do():
    assert not names_match_actionable("Jon Jones", "Arthur Jones")
    assert _names_match("Jon Jones", "Arthur Jones")
    market = pd.DataFrame(
        [
            {
                "event_id": "evt-card",
                "fighter_1": "Arthur Jones",
                "fighter_2": "Ciryl Gane",
                "f1_odds": 1.70,
                "f2_odds": 2.20,
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
            }
        ]
    )
    fuzzy = select_moneyline_quote("Jon Jones", "Ciryl Gane", market, event_id="evt-card")
    assert fuzzy["status"] == "advisory"
    assert fuzzy["quote_match"] == "advisory_fuzzy"
    fuzzy_ticket = _strong_ticket(
        **_quote_frame(
            fighter_1="Jon Jones",
            fighter_2="Ciryl Gane",
            quote_fighter_1="Arthur Jones",
            quote_fighter_2="Ciryl Gane",
            pick="Jon Jones",
            fight="Jon Jones vs Ciryl Gane",
            quote_match="advisory_fuzzy",
        ),
        kelly_stake_usd=9.0,
    )
    fuzzy_out = allocate_card_budget_pct(
        [fuzzy_ticket], 100.0, profile="live", inplace=False, as_of=NOW
    )
    assert float(fuzzy_out[0]["suggested_stake"]) == 0.0

    assert names_match_actionable("Ian Garry", "Ian Machado Garry")
    alias_market = pd.DataFrame(
        [
            {
                "event_id": "evt-card",
                "fighter_1": "Ian Machado Garry",
                "fighter_2": "Belal Muhammad",
                "f1_odds": 1.55,
                "f2_odds": 2.50,
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
                "quote_timestamp": (NOW - timedelta(minutes=5)).isoformat(),
            }
        ]
    )
    alias = select_moneyline_quote(
        "Ian Garry",
        "Belal Muhammad",
        alias_market,
        event_id="evt-card",
    )
    assert alias["status"] == "actionable"
    assert alias["quote_match"] == "alias"
    assert float(alias["row"]["f1_odds"]) == 1.55

    reversed_market = pd.DataFrame(
        [
            {
                "event_id": "evt-card",
                "fighter_1": "Stipe Miocic",
                "fighter_2": "Jon Jones",
                "f1_odds": 3.10,
                "f2_odds": 1.45,
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
                "quote_timestamp": (NOW - timedelta(minutes=5)).isoformat(),
            }
        ]
    )
    swapped = select_moneyline_quote(
        "Jon Jones",
        "Stipe Miocic",
        reversed_market,
        event_id="evt-card",
    )
    assert swapped["status"] == "actionable"
    assert swapped["quote_match"] == "reversed"
    assert str(swapped["row"]["fighter_1"]) == "Jon Jones"
    assert float(swapped["row"]["f1_odds"]) == 1.45
    assert float(swapped["row"]["f2_odds"]) == 3.10


def test_multiple_plausible_matches_fail_closed():
    market = pd.DataFrame(
        [
            {
                "event_id": "evt-a",
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "f1_odds": 1.40,
                "f2_odds": 3.00,
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
                "bookmaker": "DraftKings",
                "bookmaker_count": 1,
                "price_kind": "book",
            },
            {
                "event_id": "evt-b",
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "f1_odds": 1.80,
                "f2_odds": 2.10,
                "commence_time": (NOW + timedelta(days=2)).isoformat(),
                "bookmaker": "FanDuel",
                "bookmaker_count": 1,
                "price_kind": "book",
            },
        ]
    )
    selected = select_moneyline_quote("Jon Jones", "Stipe Miocic", market)
    assert selected["row"] is None
    assert selected["reason"] == "ambiguous_match"


def test_consensus_average_is_advisory_and_unsized_in_live():
    identity = classify_book_price(bookmaker_count=4, bookmaker="DraftKings")
    assert identity["bookmaker"] == "consensus"
    assert identity["price_kind"] == "consensus"
    single = classify_book_price(bookmaker_count=1, bookmaker="DraftKings")
    assert single["price_kind"] == "book"
    assert single["bookmaker"] == "DraftKings"
    ticket = _strong_ticket(
        **_quote_frame(
            bookmaker="consensus",
            odds_source="consensus",
            bookmaker_count=4,
            price_kind="consensus",
        ),
        kelly_stake_usd=12.0,
        suggested_stake=12.0,
    )
    out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
    assert float(out[0]["suggested_stake"]) == 0.0
    assert out[0]["quote_block_reason"] == "consensus_not_executable"
    assert out[0]["bet_action"] == "NO BET"


def test_fallback_keeps_real_source_identity():
    frame = pd.DataFrame(
        [
            {
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "f1_odds": 1.50,
                "f2_odds": 2.60,
                "bookmaker": "BetNow",
                "odds_source": "BetNow",
                "bookmaker_count": 1,
                "price_kind": "book",
            }
        ]
    )
    out = _normalize_odds_frame(frame, bookmaker="DraftKings")
    assert out.loc[0, "bookmaker"] == "BetNow"
    assert out.loc[0, "odds_source"] == "BetNow"


def test_cached_paper_to_live_and_narrative_cannot_revive_invalid_quote(monkeypatch):
    from src.grok_analysis import apply_grok_kelly_adjustments

    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.25)
    expired = _strong_ticket(
        **_quote_frame(quote_timestamp=(NOW - timedelta(hours=5)).isoformat()),
        kelly_stake_usd=25.0,
        suggested_stake=25.0,
        stake_usd=25.0,
    )
    fresh = _strong_ticket(**_quote_frame(), kelly_stake_usd=4.0, suggested_stake=4.0)
    first = allocate_card_budget_pct(
        [expired, fresh],
        100.0,
        profile="live",
        inplace=False,
        as_of=NOW,
    )
    assert float(first[0]["suggested_stake"]) == 0.0
    assert float(first[1]["suggested_stake"]) > 0.0
    assert float(first[1]["suggested_stake"]) <= 4.0 + 1e-9
    tilted = apply_grok_kelly_adjustments(
        first,
        {
            "ok": True,
            "picks": [
                {
                    "id": "ab",
                    "kelly_adjustment": 1.20,
                    "conviction": "high",
                    "narrative_edge": "noise",
                }
            ],
        },
    )
    assert float(tilted[0]["suggested_stake"]) == 0.0
    assert float(tilted[0]["kelly_stake_usd"]) == 0.0
    second = allocate_card_budget_pct(tilted, 100.0, profile="live", inplace=False, as_of=NOW)
    assert float(second[0]["suggested_stake"]) == 0.0
    assert float(second[0]["kelly_stake_usd"]) == 0.0
    assert float(second[1]["suggested_stake"]) <= 4.0 + 1e-9


def test_prop_lookup_rejects_surname_only():
    props = pd.DataFrame(
        [
            {
                "fighter_1": "Arthur Jones",
                "fighter_2": "Ciryl Gane",
                "prop_key": "over_1_5_rounds",
                "selection": "Over",
                "decimal_odds": 1.80,
            }
        ]
    )
    assert lookup_prop_odds_row("Jon Jones", "Ciryl Gane", "over_1_5_rounds", props) is None
    props.loc[0, "fighter_1"] = "Jon Jones"
    found = lookup_prop_odds_row("Jon Jones", "Ciryl Gane", "over_1_5_rounds", props)
    assert found is not None


def test_missing_quote_fighters_and_card_id_are_not_invented():
    missing_names = _strong_ticket(
        **_quote_frame(quote_fighter_1="", quote_fighter_2=""),
        kelly_stake_usd=5.0,
    )
    assert live_quote_block_reason(missing_names, now=NOW) == "missing_fighters"
    borrowed = _strong_ticket(
        **_quote_frame(quote_event_id="", event_id="evt-card", card_event_date=None, commence_time=None),
        kelly_stake_usd=5.0,
    )
    assert live_quote_block_reason(borrowed, now=NOW) == "missing_event_identity"
    out = allocate_card_budget_pct(
        [missing_names, borrowed], 100.0, profile="live", inplace=False, as_of=NOW
    )
    assert float(out[0]["suggested_stake"]) == 0.0
    assert float(out[1]["suggested_stake"]) == 0.0


def test_surname_price_is_not_bound_and_median_is_not_a_book():
    assert price_for_selection({"Arthur Jones": 1.7, "Ciryl Gane": 2.2}, "Jon Jones") is None
    assert price_for_selection({"Jon Jones": 1.45, "Stipe Miocic": 3.1}, "Jon Jones") == 1.45
    one_book = _pick_best_totals_price(
        [("DraftKings", [{"name": "Over", "price": 1.91, "point": 1.5}])],
        side="over",
        point=1.5,
    )
    assert one_book["bookmaker"] == "DraftKings"
    assert one_book["price_kind"] == "book"
    assert one_book["bookmaker_count"] == 1
    median = _pick_best_totals_price(
        [
            ("DraftKings", [{"name": "Over", "price": 1.80, "point": 1.5}]),
            ("FanDuel", [{"name": "Over", "price": 1.91, "point": 1.5}]),
            ("BetMGM", [{"name": "Over", "price": 2.05, "point": 1.5}]),
        ],
        side="over",
        point=1.5,
    )
    assert median["bookmaker"] == "consensus"
    assert median["price_kind"] == "consensus"
    assert median["bookmaker_count"] == 3
    ticket = _strong_ticket(
        **_quote_frame(
            bookmaker="consensus",
            odds_source="the_odds_api",
            price_kind="consensus",
            bookmaker_count=3,
            pick="Over 1.5",
            selection="Over 1.5",
        ),
        kelly_stake_usd=6.0,
    )
    out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False, as_of=NOW)
    assert out[0]["quote_block_reason"] == "consensus_not_executable"
    assert float(out[0]["suggested_stake"]) == 0.0


def test_two_prop_events_are_not_attached(monkeypatch, tmp_path):
    props = pd.DataFrame(
        [
            {
                "event_id": "evt-old",
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "prop_key": "over_1_5_rounds",
                "selection": "Over",
                "decimal_odds": 1.70,
                "commence_time": "2024-11-16T00:00:00+00:00",
            },
            {
                "event_id": "evt-new",
                "fighter_1": "Jon Jones",
                "fighter_2": "Stipe Miocic",
                "prop_key": "over_1_5_rounds",
                "selection": "Over",
                "decimal_odds": 1.90,
                "commence_time": "2026-06-03T00:00:00+00:00",
            },
        ]
    )
    assert (
        lookup_prop_odds_row(
            "Jon Jones",
            "Stipe Miocic",
            "over_1_5_rounds",
            props,
            event_id="evt-new",
            event_date="2026-06-03",
        )["event_id"]
        == "evt-new"
    )
    assert (
        lookup_prop_odds_row("Jon Jones", "Stipe Miocic", "over_1_5_rounds", props) is None
    )

    from src.odds_providers import draftkings as dk

    cache = tmp_path / "dk.csv"
    stale = (
        "event_id,fighter_1,fighter_2,f1_odds,f2_odds,quote_timestamp,commence_time,bookmaker\n"
        "evt-old,Jon Jones,Stipe Miocic,1.4,2.8,2020-01-01T00:00:00+00:00,2020-02-01T00:00:00+00:00,DraftKings\n"
    )
    cache.write_text(stale, encoding="utf-8")
    monkeypatch.setattr(dk, "DK_CACHE_PATH", cache)
    monkeypatch.setattr(config, "ODDS_API_KEY", "fake-odds-api-key")
    monkeypatch.setattr(config, "ODDS_FETCH_ONCE", True)

    def _boom(*_args, **_kwargs):
        raise AssertionError("ODDS_FETCH_ONCE must not download")

    monkeypatch.setattr(dk.requests, "get", _boom)
    loaded = dk.fetch_draftkings_odds(force_refresh=True)
    assert str(loaded.loc[0, "quote_timestamp"]).startswith("2020-01-01")
    assert cache.read_text(encoding="utf-8") == stale


def test_selection_price_must_match_the_quote():
    wrong_price = _strong_ticket(
        **_quote_frame(),
        decimal_odds=2.40,
        kelly_stake_usd=6.0,
    )
    assert live_quote_block_reason(wrong_price, now=NOW) == "price_mismatch"
    other_side = _strong_ticket(
        **_quote_frame(quote_f1_odds=1.55, quote_f2_odds=2.40),
        decimal_odds=2.40,
        pick="A",
        kelly_stake_usd=6.0,
    )
    assert live_quote_block_reason(other_side, now=NOW) == "price_mismatch"
    prop = _strong_ticket(
        **_quote_frame(
            pick="Over 1.5",
            selection="Over 1.5",
            quote_selection="Over 1.5",
            quote_decimal_odds=1.91,
            quote_f1_odds=None,
            quote_f2_odds=None,
        ),
        decimal_odds=1.91,
        kelly_stake_usd=4.0,
    )
    assert live_quote_block_reason(prop, now=NOW) == ""
    out = allocate_card_budget_pct(
        [wrong_price, prop], 100.0, profile="live", inplace=False, as_of=NOW
    )
    assert float(out[0]["suggested_stake"]) == 0.0
    assert float(out[0]["kelly_stake_usd"]) == 0.0
    assert 0.0 < float(out[1]["suggested_stake"]) <= 4.0 + 1e-9


def test_event_ids_use_namespace_and_utc_dates():
    same_namespace = _strong_ticket(
        **_quote_frame(
            card_event_id="evt-new",
            quote_event_id="evt-old",
            card_event_date=(NOW + timedelta(days=2)).isoformat(),
            commence_time=(NOW + timedelta(days=2)).isoformat(),
        ),
        kelly_stake_usd=5.0,
    )
    assert live_quote_block_reason(same_namespace, now=NOW) == "wrong_event"
    hex_id = "a" * 32
    cross = _strong_ticket(
        **_quote_frame(
            card_event_id="ufc:321",
            quote_event_id=hex_id,
            card_event_date="2026-06-03T04:00:00+00:00",
            commence_time="2026-06-02T23:00:00-05:00",
        ),
        kelly_stake_usd=5.0,
    )
    assert live_quote_block_reason(cross, now=NOW) == ""
    shifted = _strong_ticket(
        **_quote_frame(
            card_event_id="ufc:321",
            quote_event_id=hex_id,
            card_event_date="2026-06-03",
            commence_time="2026-06-03T22:00:00-04:00",
        ),
        kelly_stake_usd=5.0,
    )
    assert live_quote_block_reason(shifted, now=NOW) == "wrong_event"
    funded = allocate_card_budget_pct([cross], 100.0, profile="live", inplace=False, as_of=NOW)
    blocked = allocate_card_budget_pct(
        [same_namespace], 100.0, profile="live", inplace=False, as_of=NOW
    )
    assert 0.0 < float(funded[0]["suggested_stake"]) <= 5.0 + 1e-9
    assert float(blocked[0]["suggested_stake"]) == 0.0


def test_daily_loss_and_unreadable_state_clear_live_stakes(tmp_path, monkeypatch):
    from src.grok_analysis import apply_grok_kelly_adjustments
    from src.strategy import budget_aware_alerts

    monkeypatch.setattr(config, "CIRCUIT_BREAKER_STATE_PATH", tmp_path / "cb.json")
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    today = date.today().isoformat()
    (tmp_path / "cb.json").write_text(
        json.dumps(
            {
                today: {
                    "circuit_tripped": True,
                    "loss_pct": 0.05,
                    "session_date": today,
                    "bankroll_open": 1000,
                }
            }
        ),
        encoding="utf-8",
    )
    ticket = _strong_ticket(**_quote_frame(), kelly_stake_usd=8.0, suggested_stake=8.0, bankroll=1000)
    out = allocate_card_budget_pct(
        [ticket], 100.0, profile="live", inplace=False, as_of=NOW, bankroll=1000
    )
    assert float(out[0]["suggested_stake"]) == 0.0
    assert float(out[0]["kelly_stake_usd"]) == 0.0
    assert out[0]["bet_action"] == "NO BET"
    tilted = apply_grok_kelly_adjustments(
        out,
        {"ok": True, "picks": [{"id": "ab", "kelly_adjustment": 1.20, "conviction": "high"}]},
    )
    assert float(tilted[0]["suggested_stake"]) == 0.0
    second = allocate_card_budget_pct(
        tilted, 100.0, profile="live", inplace=False, as_of=NOW, bankroll=1000
    )
    assert float(second[0]["suggested_stake"]) == 0.0
    assert float(second[0]["kelly_stake_usd"]) == 0.0

    (tmp_path / "cb.json").write_text("{not-json", encoding="utf-8")
    fresh = _strong_ticket(**_quote_frame(), kelly_stake_usd=8.0, suggested_stake=8.0)
    blocked = allocate_card_budget_pct(
        [fresh], 100.0, profile="live", inplace=False, as_of=NOW, bankroll=1000
    )
    assert float(blocked[0]["suggested_stake"]) == 0.0
    assert "unreadable" in str(blocked[0].get("safety_block_reason") or "")
    cached = budget_aware_alerts(
        {"singles": [dict(fresh)], "parlays": [{"suggested_stake": 15.0, "stake_pct": 10.0, "kelly_stake_usd": 15.0}]},
        None,
        "DraftKings",
        profile="live",
    )
    assert float(cached["singles"][0]["suggested_stake"]) == 0.0
    assert float(cached["parlays"][0]["suggested_stake"]) == 0.0
    paper = allocate_card_budget_pct(
        [_strong_ticket(kelly_stake_usd=8.0)], 100.0, profile="paper", inplace=False
    )
    assert float(paper[0]["suggested_stake"]) > 0.0


def test_gane_panel_stays_advisory_at_zero(monkeypatch):
    from src.gane_foul_scenario import build_gane_foul_scenario

    monkeypatch.setattr(
        "src.gane_foul_scenario.fetch_live_prop_odds",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    card = pd.DataFrame(
        [
            {
                "fighter_1": "Alex Pereira",
                "fighter_2": "Ciryl Gane",
                "prob_f1_win": 0.62,
                "prob_f2_win": 0.38,
                "f2_ko_rate": 0.20,
            }
        ]
    )
    synthetic = build_gane_foul_scenario(overview_predictions=card, books={})
    assert synthetic["suggested_stake_usd"] == 0.0
    assert synthetic["kelly_stake_usd"] == 0.0
    assert synthetic["advisory"] is True
    assert synthetic["bet_action"] == "NO BET"
    assert synthetic["price_kind"] == "synthetic"

    quoted_card = card.copy()
    quoted_card["f2_odds"] = 3.40
    quoted = build_gane_foul_scenario(
        overview_predictions=quoted_card,
        books={"DraftKings": {"predictions": quoted_card}},
    )
    assert quoted["suggested_stake_usd"] == 0.0
    assert quoted["advisory"] is True
    assert quoted["bet_action"] == "NO BET"
    assert quoted["best_bet"]["decimal_odds"] == 3.40
    assert quoted["price_kind"] == "book"
