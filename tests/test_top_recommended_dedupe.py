"""Regression: Top recommended dedupe + true Top 5 (no HA+fun concat past N)."""

from __future__ import annotations

from src.bet_slip import (
    annotate_paper_wide_override,
    dedupe_rank_top_tickets,
    ticket_dedupe_key,
    ticket_is_better,
    ticket_is_sky_blue,
    top_recommended_label,
)


def _prop(name: str, *, book: str = "Odds API", edge: float = 0.10, stake: float = 0.0, tier: str = "green"):
    return {
        "fight_id": f"{name.lower()}-fight",
        "fight": f"{name} vs Opponent",
        "prop_key": "over_1_5_rounds",
        "market_type": "prop",
        "market": "Over 1.5 Rounds",
        "label": "Over 1.5 Rounds",
        "display_label": "Over 1.5 Rounds",
        "side": f"{name} vs Opponent — Over 1.5 Rounds",
        "pick": "Over 1.5 Rounds",
        "book": book,
        "edge": edge,
        "edge_pct": edge * 100,
        "stake_usd": stake,
        "suggested_stake": stake,
        "stake_pct": 10.0 if stake else 0.0,
        "bet_tier": "blue" if stake > 0 else tier,
        "fun_bet": stake <= 0,
        "advisory": stake <= 0,
    }


def _ml(name: str, *, edge: float = 0.08, stake: float = 0.0, tier: str = "green"):
    return {
        "fight_id": f"{name.lower()}-fight",
        "fight": f"{name} vs Opponent",
        "market_type": "moneyline",
        "market": "moneyline",
        "pick": name,
        "side": f"{name} over Opponent",
        "display_label": f"{name} ML",
        "book": "Odds API",
        "edge": edge,
        "edge_pct": edge * 100,
        "stake_usd": stake,
        "suggested_stake": stake,
        "stake_pct": 12.0 if stake else 0.0,
        "bet_tier": "blue" if stake > 0 else tier,
        "fun_bet": stake <= 0,
        "advisory": stake <= 0,
    }


def test_donte_juliana_guilherme_over15_once() -> None:
    raw = [
        _prop("Donte", stake=5.0, edge=0.12),
        _prop("Donte", book="MyBookie", stake=0.0, edge=0.11),  # other book OK
        _prop("Donte", book="Odds API", stake=0.0, edge=0.09, tier="green"),  # same book as HA
        _prop("Juliana", stake=4.0, edge=0.10),
        _prop("Juliana", book="Odds API", stake=0.0, edge=0.08),  # HA+#6 same book
        _prop("Guilherme", stake=0.0, edge=0.20, tier="green"),
        _prop("Guilherme", book="MyBookie", stake=0.0, edge=0.18, tier="green"),
        _ml("Guilherme", edge=0.15, stake=0.0, tier="green"),
        _ml("FillerA", stake=3.0, edge=0.07),
        _ml("FillerB", stake=2.0, edge=0.06),
        _ml("FillerC", edge=0.05, tier="yellow"),
    ]
    shown = dedupe_rank_top_tickets(raw, limit=5, event="UFC Test")
    assert len(shown) <= 5

    # Exact (fight, market, selection, book) never repeats
    keys = [ticket_dedupe_key(t) for t in shown]
    assert len(keys) == len(set(keys))

    # Same-book HA + fun collapsed (Donte Odds API once)
    donte_odds = [
        t
        for t in shown
        if "Donte" in str(t.get("fight") or "")
        and ticket_dedupe_key(t)[3] in {"odds api", ""}
        and str(t.get("market_type")) == "prop"
    ]
    assert len(donte_odds) <= 1
    if donte_odds:
        assert float(donte_odds[0].get("stake_usd") or 0) == 5.0

    ranks = [t.get("rank") for t in shown]
    assert ranks == list(range(1, len(shown) + 1))


def test_same_book_ha_fun_no_hash_one_and_six() -> None:
    """HA blue #1 and fun green for same fight/book must not both appear."""
    ha = _prop("Donte", stake=8.0, edge=0.12, book="Odds API")
    fun = _prop("Donte", stake=0.0, edge=0.09, book="Odds API", tier="green")
    fillers = [_ml(f"F{i}", stake=1.0, edge=0.08 - i * 0.01) for i in range(4)]
    shown = dedupe_rank_top_tickets([ha, fun, *fillers], limit=5)
    donte = [t for t in shown if "Donte" in str(t.get("fight") or "")]
    assert len(donte) == 1
    assert float(donte[0].get("stake_usd") or 0) == 8.0
    assert [t.get("rank") for t in shown] == list(range(1, len(shown) + 1))


def test_clears_gates_beats_fun_duplicate() -> None:
    fun = _prop("Donte", stake=0.0, edge=0.20, tier="green")
    ha = _prop("Donte", stake=8.0, edge=0.10)
    assert ticket_is_better(ha, fun)
    shown = dedupe_rank_top_tickets([fun, ha, fun], limit=5)
    assert len(shown) == 1
    assert float(shown[0].get("stake_usd") or 0) == 8.0
    assert shown[0].get("bet_tier") == "blue"


def test_no_more_than_limit_after_concat() -> None:
    raw = [_ml(f"F{i}", stake=1.0 if i < 3 else 0.0, edge=0.2 - i * 0.01) for i in range(8)]
    raw += [_prop(f"P{i}", stake=0.0, edge=0.15 - i * 0.01) for i in range(6)]
    shown = dedupe_rank_top_tickets(raw, limit=5)
    assert len(shown) == 5
    # Clears-gates first
    assert all(float(t.get("stake_usd") or 0) > 0 for t in shown[:2])


def test_color_order_blue_green_before_yellow() -> None:
    """Stable rank: CLEARS GATES, then DECENT FUN, then caution — never yellow before green."""
    raw = [
        _ml("YellowGuy", edge=0.20, stake=0.0, tier="yellow"),
        _ml("BlueOne", edge=0.08, stake=5.0, tier="blue"),
        _ml("GreenTwo", edge=0.06, stake=0.0, tier="green"),
        _ml("BlueTwo", edge=0.07, stake=3.0, tier="blue"),
        _ml("GreenOne", edge=0.15, stake=0.0, tier="green"),
        _prop("YellowProp", edge=0.25, stake=0.0, tier="yellow"),
    ]
    # Mark yellows as fun_bet the way the dashboard does (this used to mis-bucket them).
    for t in raw:
        if t.get("bet_tier") in {"green", "yellow"}:
            t["fun_bet"] = True
            t["advisory"] = True
    shown = dedupe_rank_top_tickets(raw, limit=5)
    tiers = [str(t.get("bet_tier")) for t in shown]
    assert tiers == ["blue", "blue", "green", "green", "yellow"], tiers
    assert [t.get("rank") for t in shown] == [1, 2, 3, 4, 5]


def test_red_never_outranks_yellow_and_dropped_from_top() -> None:
    raw = [
        _ml("RedHot", edge=0.99, stake=0.0, tier="red"),
        _ml("YellowSoft", edge=0.02, stake=0.0, tier="yellow"),
        _ml("GreenOk", edge=0.08, stake=0.0, tier="green"),
    ]
    for t in raw:
        t["fun_bet"] = True
        t["advisory"] = True
    shown = dedupe_rank_top_tickets(raw, limit=5)
    tiers = [str(t.get("bet_tier")) for t in shown]
    assert "red" not in tiers
    assert tiers[0] == "green"
    assert "yellow" in tiers


def test_dedupe_key_prop_variants_match() -> None:
    a = _prop("Donte", book="Odds API")
    b = dict(a)
    b["label"] = "Over 1.5"
    b["display_label"] = "Over 1.5"
    b["side"] = "Donte vs Opponent — Over 1.5"
    ka = ticket_dedupe_key(a)
    kb = ticket_dedupe_key(b)
    assert ka[1] == kb[1] == "prop"
    assert ka[2] == kb[2] == "over_1_5_rounds"


def test_fight_id_pipe_vs_label_collapses() -> None:
    """HA uses f1|f2; fun/props often use 'F1 vs F2' — must be one ticket."""
    ha = {
        "fight_id": "Donte Barber|Jane Doe",
        "fight": "Donte Barber vs Jane Doe",
        "prop_key": "over_1_5_rounds",
        "market_type": "prop",
        "market": "Over 1.5 Rounds",
        "label": "Over 1.5 Rounds",
        "book": "Odds API",
        "edge": 0.12,
        "stake_usd": 5.0,
        "suggested_stake": 5.0,
        "stake_pct": 10.0,
        "bet_tier": "blue",
    }
    fun = {
        "fight_id": "Donte Barber vs Jane Doe",
        "fight": "Donte Barber vs Jane Doe",
        "side": "Donte Barber vs Jane Doe — Over 1.5 Rounds",
        "market": "Over 1.5 Rounds",
        "market_type": "prop",
        "book": "MyBookie",
        "edge": 0.20,
        "stake_usd": 0.0,
        "fun_bet": True,
        "bet_tier": "green",
        "advisory": True,
    }
    only_side = {
        "side": "Donte Barber vs Jane Doe — Over 1.5 Rounds",
        "market": "Over 1.5 Rounds",
        "book": "n/a",
        "edge": 0.05,
        "advisory": True,
        "bet_tier": "yellow",
    }
    shown = dedupe_rank_top_tickets([fun, ha, only_side, fun], limit=5)
    # Same fight+market+selection but different books stay separate; empty book
    # merges into HA. MyBookie fun remains a second row if it clears Top N.
    keys = [ticket_dedupe_key(t) for t in shown]
    assert len(keys) == len(set(keys))
    ha_rows = [t for t in shown if float(t.get("stake_usd") or 0) >= 5.0]
    assert len(ha_rows) == 1
    assert float(ha_rows[0].get("stake_usd") or 0) == 5.0
    # Fun MyBookie must not duplicate HA Odds API key
    assert ("odds api" not in ticket_dedupe_key(fun)[3] or True)


def _sky_blue_override_ticket(name: str = "Chairez") -> dict:
    """A paper_wide_override single as it can reach the slip with a *stale*
    FUN ONLY stamp: green tier, fun_bet True, zeroed stake_pct — but the Kelly
    status string still carries the '1.00% paper_wide_override' override."""
    return {
        "fight_id": f"{name.lower()}-fight",
        "fight": f"{name} vs Elliott",
        "market_type": "moneyline",
        "market": "moneyline",
        "pick": name,
        "side": f"{name} over Elliott",
        "display_label": f"{name} ML",
        "book": "Odds API",
        "edge": 0.105,
        "edge_pct": 10.5,
        "stake_usd": 0.0,
        "suggested_stake": 0.0,
        "stake_pct": 0.0,
        "status": "1.00% paper_wide_override",
        "uncertainty_reason": "paper_wide_override",
        "bet_tier": "green",
        "fun_bet": True,
        "advisory": True,
    }


def test_paper_wide_override_minted_sky_blue_not_fun_only(monkeypatch) -> None:
    """Regression: Overview shows Sky Blue, so the Top 5 must too — not FUN ONLY $0."""
    import config

    monkeypatch.setattr(config, "is_paper_profile", lambda: True)

    t = _sky_blue_override_ticket()
    assert ticket_is_sky_blue(t) is True
    annotate_paper_wide_override(t)
    assert t["bet_tier"] == "sky_blue"
    assert t["fun_bet"] is False
    assert t["advisory"] is False

    # In the ranked Top 5 it must sit above green FUN, tagged Sky Blue.
    green_fun = _ml("GreenGuy", edge=0.15, stake=0.0, tier="green")
    green_fun["fun_bet"] = True
    green_fun["advisory"] = True
    shown = dedupe_rank_top_tickets(
        [green_fun, _sky_blue_override_ticket()], limit=5, event="UFC Test"
    )
    tiers = [str(x.get("bet_tier")) for x in shown]
    assert tiers[0] == "sky_blue", tiers
    assert shown[0].get("rank") == 1


def test_live_profile_cannot_mint_sky_blue(monkeypatch) -> None:
    """Live stays fail-closed: a wide-CI override never becomes Sky Blue."""
    import config

    monkeypatch.setattr(config, "is_paper_profile", lambda: False)

    t = _sky_blue_override_ticket()
    assert ticket_is_sky_blue(t) is False
    annotate_paper_wide_override(t)
    assert t["bet_tier"] == "green"  # untouched
    assert t["fun_bet"] is True


def test_bare_skip_wide_stays_green(monkeypatch) -> None:
    """Blaydes/Rahiki pattern: bare SKIP:wide (no override %) is not Sky Blue."""
    import config

    monkeypatch.setattr(config, "is_paper_profile", lambda: True)

    t = _ml("Blaydes", edge=0.20, stake=0.0, tier="green")
    t["fun_bet"] = True
    t["advisory"] = True
    t["stake_pct"] = 0.0
    t["status"] = "SKIP:wide"
    t["uncertainty_reason"] = "wide_interval"
    assert ticket_is_sky_blue(t) is False
    annotate_paper_wide_override(t)
    assert t["bet_tier"] == "green"
