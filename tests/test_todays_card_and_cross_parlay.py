"""Today's Card resolution + best parlay across all loaded cards."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data_loader import match_todays_card_index, select_upcoming_event_indices
from src.strategy import build_auto_parlay_recommendations, build_best_parlay_across_cards


def _events() -> list[dict[str, str]]:
    today = datetime.now(timezone.utc).date().isoformat()
    return [
        {"event_name": "UFC Today", "event_date": today, "event_path": "/event/today"},
        {"event_name": "UFC Next Week", "event_date": "2099-01-01", "event_path": "/event/next"},
        {"event_name": "UFC Later", "event_date": "2099-02-01", "event_path": "/event/later"},
    ]


def test_match_todays_card_index():
    events = _events()
    assert match_todays_card_index(events) == 0
    no_today = [
        {"event_name": "A", "event_date": "2099-01-01", "event_path": "/a"},
        {"event_name": "B", "event_date": "2099-02-01", "event_path": "/b"},
    ]
    assert match_todays_card_index(no_today) == 0


def test_select_todays_and_all_modes():
    events = _events()
    today_idxs = select_upcoming_event_indices(
        events, todays_card=True, include_adjacent_week=False
    )
    assert today_idxs == [0]
    assert select_upcoming_event_indices(
        events, "Today's Card", include_adjacent_week=False
    ) == [0]
    assert select_upcoming_event_indices(
        events, "Current Card", include_adjacent_week=False
    ) == [0]
    assert select_upcoming_event_indices(
        events, all_available=True, include_adjacent_week=False
    ) == [0, 1, 2]


def _card(event: str, start: int, n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        p = 0.88 - i * 0.04
        rows.append(
            {
                "fight_id": f"{event}-f{start + i}",
                "fighter_1": f"{event}_A{i}",
                "fighter_2": f"{event}_B{i}",
                "predicted_winner": f"{event}_A{i}",
                "prob_f1_win": p,
                "prob_f2_win": 1.0 - p,
                "predicted_prob": p,
                "confidence_label": "high",
                "event_name": event,
            }
        )
    return pd.DataFrame(rows)


def test_auto_parlay_marks_cross_card():
    frame = pd.concat([_card("CardA", 0), _card("CardB", 10)], ignore_index=True)
    recs = build_auto_parlay_recommendations(frame)
    assert recs
    two = next(r for r in recs if r["n_legs"] == 2)
    assert all("event_name" in leg for leg in two["legs"])
    events = {leg["event_name"] for leg in two["legs"]}
    if len(events) > 1:
        assert two["cross_card"] is True
        assert "across cards" in two["brief"]


def test_build_best_parlay_across_cards_from_card_list():
    cards = [
        {"event_name": "CardA", "predictions": _card("CardA", 0, 4)},
        {"event_name": "CardB", "predictions": _card("CardB", 10, 4)},
    ]
    recs = build_best_parlay_across_cards(cards)
    assert sorted(int(r["n_legs"]) for r in recs) == [2, 3]
    three = next(r for r in recs if r["n_legs"] == 3)
    leg_events = {leg.get("event_name") for leg in three["legs"]}
    assert leg_events <= {"CardA", "CardB"}
    assert three.get("advisory") is True
    assert float(three.get("stake_usd") or 0) == 0.0
