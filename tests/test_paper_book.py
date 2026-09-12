"""Tests for the automated theoretical paper book."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src import paper_book


@pytest.fixture
def book_paths(tmp_path, monkeypatch):
    csv_path = tmp_path / "paper_book.csv"
    state_path = tmp_path / "paper_book_state.json"
    monkeypatch.setattr(config, "PAPER_BOOK_CSV", csv_path, raising=False)
    monkeypatch.setattr(config, "PAPER_BOOK_STATE_JSON", state_path, raising=False)
    monkeypatch.setattr(config, "PAPER_BOOK_START_BANKROLL", 1000.0, raising=False)
    monkeypatch.setattr(config, "PAPER_BOOK_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "PAPER_BOOK_ALLOWED_BOOKS", ("odds api", ""), raising=False)
    return csv_path, state_path


def _ml(name: str, opp: str, *, tier="blue", odds=2.0, stake=20.0, book="the_odds_api", edge=8.0,
        fun=False):
    return {
        "event": "Test Card",
        "fight": f"{name} vs {opp}",
        "fighter_1": name,
        "fighter_2": opp,
        "market_type": "moneyline",
        "pick": name,
        "side": f"{name} over {opp}",
        "bet_tier": tier,
        "book": book,
        "decimal_odds": odds,
        "suggested_stake": stake,
        "stake_pct": 5.0,
        "edge_pct": edge,
        "fun_bet": fun,
        "advisory": fun,
    }


def _prop(name: str, opp: str, *, odds=1.5, stake=10.0, book="the_odds_api"):
    return {
        "event": "Test Card",
        "fight": f"{name} vs {opp}",
        "fighter_1": name,
        "fighter_2": opp,
        "market_type": "prop",
        "prop_key": "over_1_5_rounds",
        "market": "Over 1.5 Rounds",
        "label": "Over 1.5 Rounds",
        "side": f"{name} vs {opp} — Over 1.5 Rounds",
        "pick": "Over 1.5 Rounds",
        "bet_tier": "blue",
        "book": book,
        "decimal_odds": odds,
        "suggested_stake": stake,
        "edge_pct": 12.0,
        "fun_bet": False,
        "advisory": False,
    }


def test_empty_book_starts_at_1000(book_paths):
    s = paper_book.paper_book_summary()
    assert s["start_bankroll"] == 1000.0
    assert s["balance"] == 1000.0
    assert s["settled_count"] == 0
    assert s["open_count"] == 0


def test_places_only_money_tickets(book_paths):
    tickets = [
        _ml("Alpha", "Zulu", tier="blue"),
        _ml("Bravo", "Yankee", tier="sky_blue", stake=5.0),
        _ml("Charlie", "Xray", tier="green", fun=True),   # fun → skip
        _ml("Delta", "Whiskey", tier="blue", stake=0.0),  # no stake → skip
    ]
    res = paper_book.place_paper_tickets(tickets, event="Test Card")
    assert res["placed"] == 2
    assert res["considered"] == 2  # only the two actionable are considered
    opens = paper_book.open_positions()
    picks = sorted(r["pick"] for r in opens)
    assert picks == ["Alpha", "Bravo"]
    assert {r["tier"] for r in opens} == {"blue", "sky_blue"}


def test_skips_non_odds_api_book_and_no_odds(book_paths):
    tickets = [
        _ml("Alpha", "Zulu", book="MyBookie"),           # wrong book → skip
        _ml("Bravo", "Yankee", odds=None),               # no usable odds → skip
    ]
    # odds=None → decimal_odds missing
    tickets[1]["decimal_odds"] = None
    res = paper_book.place_paper_tickets(tickets, event="Test Card")
    assert res["placed"] == 0
    assert res["skipped_book"] == 1
    assert res["skipped_no_odds"] == 1
    assert paper_book.open_positions() == []


def test_placement_is_idempotent(book_paths):
    tickets = [_ml("Alpha", "Zulu")]
    first = paper_book.place_paper_tickets(tickets, event="Test Card")
    second = paper_book.place_paper_tickets(tickets, event="Test Card")
    assert first["placed"] == 1
    assert second["placed"] == 0
    assert second["skipped_existing"] == 1
    assert len(paper_book.open_positions()) == 1


def _fights(rows):
    return pd.DataFrame(rows)


def test_settle_moneyline_win_and_loss_updates_balance(book_paths):
    paper_book.place_paper_tickets(
        [
            _ml("Alpha", "Zulu", odds=2.0, stake=20.0),   # will WIN  → +20.0
            _ml("Bravo", "Yankee", odds=3.0, stake=10.0),  # will LOSE → -10.0
        ],
        event="Test Card",
    )
    hist = _fights(
        [
            {"fighter_1": "Alpha", "fighter_2": "Zulu", "winner": "Alpha", "method": "KO", "round": 2},
            {"fighter_1": "Bravo", "fighter_2": "Yankee", "winner": "Yankee", "method": "KO", "round": 1},
        ]
    )
    res = paper_book.settle_paper_book(historical=hist)
    assert res["settled"] == 2
    s = paper_book.paper_book_summary()
    # +20 (win at 2.0 on 20) and -10 (loss on 10) → 1000 + 10
    assert s["realized_pnl"] == 10.0
    assert s["balance"] == 1010.0
    assert s["settled_count"] == 2
    assert s["wins"] == 1
    assert s["hit_rate_pct"] == 50.0
    # ROI = 10 / (20+10) stake
    assert s["roi_pct"] == pytest.approx(33.33, abs=0.05)
    assert len(s["equity_curve"]) == 3  # start + 2 settled


def test_settle_prop_over_1_5_by_decision(book_paths):
    paper_book.place_paper_tickets([_prop("Klose", "Gantt", odds=1.5, stake=10.0)], event="Test Card")
    hist = _fights(
        [{"fighter_1": "Klose", "fighter_2": "Gantt", "winner": "Gantt", "method": "Decision", "round": 3}]
    )
    res = paper_book.settle_paper_book(historical=hist)
    assert res["settled"] == 1
    s = paper_book.paper_book_summary()
    assert s["wins"] == 1
    assert s["balance"] == pytest.approx(1005.0)  # +0.5*10


def test_prop_without_round_data_stays_open_fail_closed(book_paths):
    paper_book.place_paper_tickets([_prop("Klose", "Gantt", odds=1.5, stake=10.0)], event="Test Card")
    hist = _fights(
        [{"fighter_1": "Klose", "fighter_2": "Gantt", "winner": "Gantt", "method": "", "round": ""}]
    )
    res = paper_book.settle_paper_book(historical=hist)
    assert res["settled"] == 0
    assert paper_book.paper_book_summary()["open_count"] == 1


def test_settle_is_idempotent(book_paths):
    paper_book.place_paper_tickets([_ml("Alpha", "Zulu", odds=2.0, stake=20.0)], event="Test Card")
    hist = _fights([{"fighter_1": "Alpha", "fighter_2": "Zulu", "winner": "Alpha", "method": "KO", "round": 2}])
    paper_book.settle_paper_book(historical=hist)
    again = paper_book.settle_paper_book(historical=hist)
    assert again["settled"] == 0
    assert paper_book.paper_book_summary()["balance"] == 1020.0


def test_disabled_flag_places_nothing(book_paths, monkeypatch):
    monkeypatch.setattr(config, "PAPER_BOOK_ENABLED", False, raising=False)
    res = paper_book.place_paper_tickets([_ml("Alpha", "Zulu")], event="Test Card")
    assert res["placed"] == 0


def test_money_tickets_from_result_extracts_sized(book_paths):
    result = {
        "event_name": "Test Card",
        "alerts": {
            "singles": [_ml("Alpha", "Zulu"), _ml("Charlie", "Xray", tier="green", fun=True)],
            "prop_singles": [_prop("Klose", "Gantt")],
        },
    }
    tickets = paper_book.money_tickets_from_result(result)
    picks = sorted(t.get("pick") for t in tickets)
    assert picks == ["Alpha", "Over 1.5 Rounds"]
