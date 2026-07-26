"""Prop key aliases remain for Odds API Under 1.5 ↔ Round 1 finish quotes."""

from __future__ import annotations

import pandas as pd

from src.high_accuracy_strategy import ALLOWED_PROP_KEYS, prop_allowed
from src.odds_providers.prop_odds_common import lookup_prop_odds_row


def test_under_1_5_aliases_to_round_1_finish():
    odds = pd.DataFrame(
        [
            {
                "fighter_1": "Alice",
                "fighter_2": "Bob",
                "prop_key": "under_1_5_rounds",
                "selection": "Under 1.5",
                "decimal_odds": 1.85,
                "implied_prob": 1.0 / 1.85,
                "odds_source": "the_odds_api",
            }
        ]
    )
    row = lookup_prop_odds_row("Alice", "Bob", "round_1_finish", odds)
    assert row is not None
    assert float(row["decimal_odds"]) == 1.85


def test_only_over_1_5_ha_actionable():
    assert prop_allowed("over_1_5_rounds")
    assert not prop_allowed("ko_tko")
    assert ALLOWED_PROP_KEYS == frozenset({"over_1_5_rounds"})
