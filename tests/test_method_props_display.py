"""Live method props (KO/sub/decision) appear on Props display as research-only."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src.odds_providers.prop_odds_common import prop_row
from src.props import rank_prop_singles


@pytest.fixture(autouse=True)
def enable_props(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROPS", True)
    monkeypatch.setattr(config, "PROP_MARKETS", ["over_1_5_rounds"])
    monkeypatch.setattr(config, "PROP_MAX_RESULTS", 6)


def test_rank_prop_singles_shows_live_method_lines(monkeypatch):
    preds = pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "event_name": "UFC Test",
                "fighter_1": "Islam Makhachev",
                "fighter_2": "Ian Machado Garry",
                "prob_f1_win": 0.61,
                "prob_f2_win": 0.39,
                "f1_ko_rate": 0.12,
                "f2_ko_rate": 0.18,
                "f1_sub_avg": 0.8,
                "f2_sub_avg": 0.2,
                "f1_finish_rate": 0.35,
                "f2_finish_rate": 0.40,
                "ko_rate_diff": 0.0,
            }
        ]
    )
    prop_odds = pd.DataFrame(
        [
            prop_row(
                fighter_1="Islam Makhachev",
                fighter_2="Ian Machado Garry",
                prop_key="fighter_decision",
                selection="Islam Makhachev Yes",
                decimal_odds=2.15,
                bookmaker="MyBookie",
                odds_source="live",
                market_key="prop",
                american_odds=115.0,
            ),
            prop_row(
                fighter_1="Islam Makhachev",
                fighter_2="Ian Machado Garry",
                prop_key="fighter_sub",
                selection="Islam Makhachev Yes",
                decimal_odds=2.91,
                bookmaker="MyBookie",
                odds_source="live",
                market_key="prop",
                american_odds=191.0,
            ),
            prop_row(
                fighter_1="Islam Makhachev",
                fighter_2="Ian Machado Garry",
                prop_key="fighter_ko",
                selection="Ian Machado Garry Yes",
                decimal_odds=8.2,
                bookmaker="MyBookie",
                odds_source="live",
                market_key="prop",
                american_odds=720.0,
            ),
        ]
    )
    ranked, meta = rank_prop_singles(
        preds, book="MyBookie", prop_odds=prop_odds, max_results=6
    )
    method = [r for r in ranked if r["prop_key"] in {"fighter_ko", "fighter_sub", "fighter_decision"}]
    assert len(method) == 3
    assert all(r["odds_source"] == "live" for r in method)
    assert all(r.get("strict_qualified") is False for r in method)
    assert all((r.get("suggested_stake") in (0, 0.0, None)) for r in method)
    islam_dec = next(r for r in method if "Decision" in r["prop_short"] and "Islam" in r["fighter"])
    assert islam_dec["edge_pct"] is not None
    assert meta["shown"] >= 3
