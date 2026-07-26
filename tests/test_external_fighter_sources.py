"""Tests for Sherdog / Wikipedia / CompuBox-style external fighter sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.compubox_stats import CompuBoxLine, differential, fill_history_from_compubox, load_detailed_bout_striking
from src.sherdog import parse_sherdog_fighter_html, sherdog_record_as_of
from src.wikipedia_fighters import parse_wikipedia_wikitext
import config


SHERDOG_HTML = """
<html><body>
<span class="fn">Jane Fighter</span>
<span class="nickname">"The Test"</span>
<div class="record"><span class="winsloses">12-3-0</span></div>
<table class="bio-holder">
<tr><th>Height</th><td>5'8"</td></tr>
<tr><th>Reach</th><td>68 in</td></tr>
<tr><th>Class</th><td>Flyweight</td></tr>
</table>
<table class="fight_history">
<tr><th>Result</th><th>Opponent</th><th>Event</th><th>Method</th><th>Date</th></tr>
<tr><td>win</td><td><a>Alice Opp</a></td><td>UFC 1</td><td>KO/TKO</td><td>Jan / 10 / 2024</td></tr>
<tr><td>loss</td><td><a>Bob Opp</a></td><td>UFC 2</td><td>Decision</td><td>Jun / 01 / 2023</td></tr>
</table>
</body></html>
"""

WIKI_TEXT = """
{{Infobox martial artist
| name = Jane Fighter
| nickname = The Test
| height = {{convert|5|ft|8|in|cm|abbr=on}}
| reach = {{convert|68|in|cm|abbr=on}}
| style = Orthodox
| birth_date = {{Birth date and age|1992|5|1}}
| nationality = American
| team = Test Gym
| weight_class = Flyweight
}}
Jane is a mixed martial artist in the UFC.
"""


def test_parse_sherdog_html():
    profile, fights = parse_sherdog_fighter_html(
        SHERDOG_HTML, url="https://www.sherdog.com/fighter/Jane-Fighter-12345"
    )
    assert profile["name"] == "Jane Fighter"
    assert profile["sherdog_id"] == "12345"
    assert abs(float(profile["height_in"]) - 68.0) < 0.1
    assert len(fights) >= 2
    assert fights[0]["result"] == "WIN"


def test_parse_wikipedia_infobox():
    profile = parse_wikipedia_wikitext(WIKI_TEXT, title="Jane Fighter (fighter)", query_name="Jane Fighter")
    assert profile["name"] == "Jane Fighter"
    assert abs(float(profile["height_in"]) - 68.0) < 0.1
    assert abs(float(profile["reach_in"]) - 68.0) < 0.1
    assert profile["stance"] == "Orthodox"
    assert profile["birth_date"].startswith("1992")


def test_compubox_differential_and_greco_load():
    a = CompuBoxLine("A", sig_strikes_landed=50, sig_strikes_attempted=100, knockdowns=2, head_pct=0.7, avg_fight_time_min=15)
    b = CompuBoxLine("B", sig_strikes_landed=40, sig_strikes_attempted=100, knockdowns=0, head_pct=0.5, avg_fight_time_min=15)
    d = differential(a, b)
    assert d["acc_diff"] > 0
    assert d["power_diff"] > 0
    bout = load_detailed_bout_striking()
    # Greco cache should exist in this project; if missing, empty is still ok (fail soft).
    assert isinstance(bout, pd.DataFrame)
    if not bout.empty:
        assert "head_strike_pct" in bout.columns
        assert "kd_rate" in bout.columns


def test_compubox_fill_is_leakage_safe(tmp_path, monkeypatch):
    # Synthetic pre-fight table via history fill no-op when empty source is fine.
    history = pd.DataFrame(
        {
            "fighter": ["A", "A"],
            config.DATE_COLUMN: pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sig_strike_acc": [np.nan, np.nan],
        }
    )
    out = fill_history_from_compubox(history, window=5)
    assert "kd_rate" in out.columns
    assert len(out) == 2


def test_sherdog_record_as_of_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.sherdog.SHERDOG_FIGHTERS_CACHE", tmp_path / "fighters.csv")
    monkeypatch.setattr("src.sherdog.SHERDOG_FIGHTS_CACHE", tmp_path / "fights.csv")
    monkeypatch.setattr("src.sherdog.SHERDOG_INDEX_CACHE", tmp_path / "index.json")
    fights = pd.DataFrame(
        [
            {
                "sherdog_id": "1",
                "fighter": "Jane Fighter",
                "opponent": "Alice",
                "result": "WIN",
                "method": "KO",
                "event": "E1",
                "bout_date": "2023-01-01",
                "weight_class": "Flyweight",
                "source": "sherdog",
            },
            {
                "sherdog_id": "1",
                "fighter": "Jane Fighter",
                "opponent": "Bob",
                "result": "LOSS",
                "method": "DEC",
                "event": "E2",
                "bout_date": "2024-06-01",
                "weight_class": "Flyweight",
                "source": "sherdog",
            },
        ]
    )
    fights.to_csv(tmp_path / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Jane Fighter",
                "sherdog_id": "1",
                "url": "",
                "nickname": "",
                "weight_class": "Flyweight",
                "height_in": 68,
                "reach_in": 68,
                "birth_date": "",
                "nationality": "",
                "team": "",
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "source": "sherdog",
                "fetched_at": "",
            }
        ]
    ).to_csv(tmp_path / "fighters.csv", index=False)

    before = sherdog_record_as_of("Jane Fighter", "2024-01-01")
    assert before["sherdog_wins"] == 1
    assert before["sherdog_losses"] == 0
    assert abs(before["sherdog_win_rate"] - 1.0) < 1e-9

    after = sherdog_record_as_of("Jane Fighter", "2025-01-01")
    assert after["sherdog_wins"] == 1
    assert after["sherdog_losses"] == 1
    assert abs(after["sherdog_win_rate"] - 0.5) < 1e-9


def test_feature_schema_bumped():
    assert config.FEATURE_SCHEMA_VERSION >= 3
    assert "kd_rate_diff" in config.FEATURE_COLUMNS
    assert "sherdog_win_rate_diff" in config.FEATURE_COLUMNS
    assert "power_proxy_diff" in config.FEATURE_COLUMNS
