"""Tests for optional Grok/Ollama narrative tilt constraints (model-first)."""

from __future__ import annotations

import math

import pytest

from src.grok_analysis import (
    apply_grok_kelly_adjustments,
    build_grok_prompt,
    clamp_kelly_factor,
    normalize_grok_result,
    resolve_narrative_tilt,
    _extract_json_blob,
)


def test_clamp_kelly_factor_paper_pm10(monkeypatch):
    import config

    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.10)
    assert clamp_kelly_factor(1.0) == 1.0
    assert clamp_kelly_factor(2.0) == 1.10
    assert clamp_kelly_factor(0.1) == 0.90
    assert clamp_kelly_factor("bad") == 1.0
    assert clamp_kelly_factor(float("nan")) == 1.0


def test_clamp_kelly_live_tighter(monkeypatch):
    import config

    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "LIVE_NARRATIVE_KELLY_MIN", 0.95)
    monkeypatch.setattr(config, "LIVE_NARRATIVE_KELLY_MAX", 1.05)
    assert clamp_kelly_factor(1.20) == 1.05
    assert clamp_kelly_factor(0.80) == 0.95


def test_extract_json_blob_from_fenced_block():
    raw = 'Here is the analysis:\n```json\n{"summary": "test", "picks": []}\n```'
    parsed = _extract_json_blob(raw)
    assert parsed["summary"] == "test"


def test_normalize_grok_result_picks(monkeypatch):
    import config

    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.10)
    raw = {
        "summary": "Card leans grapplers",
        "picks": [
            {
                "id": "fight-1",
                "pick_type": "moneyline",
                "narrative_edge": "Wrestling edge",
                "crowd_positioning": "Public on favorite",
                "invalidation_risks": ["Bad weight cut"],
                "kelly_adjustment": 1.08,
                "conviction": "high",
            }
        ],
    }
    out = normalize_grok_result(raw, event_label="UFC 300")
    assert out["event"] == "UFC 300"
    assert len(out["picks"]) == 1
    assert out["picks"][0]["kelly_adjustment"] == 1.08
    assert out["picks"][0]["invalidation_risks"] == ["Bad weight cut"]


def test_build_grok_prompt_includes_constraints():
    inputs = {
        "event": "UFC Test",
        "profile": "paper",
        "bankroll": 100.0,
        "card_budget": 55.0,
        "total_stake_pct": 38.0,
        "total_stake_usd": 20.9,
        "tickets": [
            {
                "id": "f1",
                "side": "A over B",
                "market": "moneyline",
                "book": "DraftKings",
                "odds_display": "1.91",
                "stake_pct": 38.0,
                "stake_usd": 20.9,
                "prob": 0.62,
                "edge_pct": 4.2,
                "confidence": "High",
                "strength_score": 0.7,
                "uncertainty_action": "allow",
            }
        ],
        "skipped": [],
    }
    prompt = build_grok_prompt(inputs)
    assert "UFC Test" in prompt
    assert "A over B" in prompt
    assert "stake_pct=38.0%" in prompt
    assert "Do NOT invent" in prompt or "Do NOT invent, flip" in prompt
    assert "MODEL-FIRST" in prompt
    assert "conf/odds" in prompt.lower() or "FINAL from conf/odds" in prompt


def test_merge_ollama_reasons_keeps_ha_stakes():
    from src.grok_analysis import merge_ollama_reasons_into_slip

    tickets = [
        {
            "id": "f1",
            "side": "A over B",
            "market": "moneyline",
            "stake_pct": 40.0,
            "stake_usd": 22.0,
            "confidence": "high",
            "edge_pct": 8.0,
        }
    ]
    picks = [{"id": "f1", "reason": "High conf grappler edge", "conviction": "high"}]
    out = merge_ollama_reasons_into_slip(tickets, picks)
    assert out[0]["stake_pct"] == 40.0
    assert out[0]["stake_usd"] == 22.0
    assert "grappler" in out[0]["reason"]

def test_low_conviction_forces_one(monkeypatch):
    import config

    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "NARRATIVE_LOW_CONVICTION_FORCE_ONE", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    bet = {"fight_id": "f1", "pick": "A", "edge_pct": 5.0, "suggested_stake": 10.0}
    item = {"id": "f1", "kelly_adjustment": 1.08, "conviction": "low", "reason": "unsure"}
    dec = resolve_narrative_tilt(bet, item, grok_ok=True)
    assert dec.factor == 1.0
    assert dec.status == "rejected"
    assert dec.reason == "low_conviction"


def test_pick_flip_rejected(monkeypatch):
    import config

    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    bet = {"fight_id": "f1", "pick": "Jon Jones", "fighter_1": "Jon Jones", "fighter_2": "Stipe"}
    item = {"id": "f1", "pick": "Stipe", "kelly_adjustment": 1.05, "conviction": "high"}
    dec = resolve_narrative_tilt(bet, item, grok_ok=True)
    assert dec.factor == 1.0
    assert dec.reason == "pick_flip_rejected"


def test_edge_inflate_rejected(monkeypatch):
    import config

    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    bet = {"fight_id": "f1", "pick": "A", "edge_pct": 4.0}
    item = {
        "id": "f1",
        "kelly_adjustment": 1.05,
        "conviction": "high",
        "edge_adjustment": 0.03,
    }
    dec = resolve_narrative_tilt(bet, item, grok_ok=True)
    assert dec.factor == 1.0
    assert dec.reason == "edge_inflate_rejected"


def test_ollama_down_fail_closed(monkeypatch):
    import config

    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    bet = {"fight_id": "f1", "pick": "A", "suggested_stake": 10.0}
    dec = resolve_narrative_tilt(bet, {"kelly_adjustment": 1.1}, grok_ok=False)
    assert dec.factor == 1.0
    assert dec.status == "fail_closed"


def test_uncertainty_skip_blocks_tilt(monkeypatch):
    import config

    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    bet = {
        "fight_id": "f1",
        "pick": "A",
        "uncertainty_action": "skip",
        "skip_reason": "high_disagreement",
        "suggested_stake": 10.0,
    }
    item = {"id": "f1", "kelly_adjustment": 1.08, "conviction": "high"}
    dec = resolve_narrative_tilt(bet, item, grok_ok=True)
    assert dec.factor == 1.0
    assert dec.reason == "after_uncertainty_skip"


def test_apply_scales_stakes_not_edge(monkeypatch):
    import config

    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.10)
    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "NARRATIVE_LOW_CONVICTION_FORCE_ONE", True)
    bets = [
        {
            "fight_id": "f1",
            "pick": "A",
            "pick_line": "A over B",
            "edge": 0.05,
            "edge_pct": 5.0,
            "kelly_stake_usd": 10.0,
            "kelly_pct": 2.0,
            "max_safe_bet_usd": 5.0,
            "suggested_stake": 8.0,
        }
    ]
    grok = {
        "ok": True,
        "picks": [
            {
                "id": "f1",
                "kelly_adjustment": 1.08,
                "narrative_edge": "Solid wrestling",
                "conviction": "high",
            }
        ],
    }
    out = apply_grok_kelly_adjustments(bets, grok)
    assert out[0]["pick"] == "A"
    assert out[0]["edge_pct"] == 5.0
    assert out[0]["edge"] == 0.05
    assert out[0]["suggested_stake"] == pytest.approx(8.0 * 1.08, rel=1e-3)
    assert out[0]["grok_kelly_factor"] == 1.08
    assert out[0]["narrative_tilt_status"] == "applied"


def test_apply_low_conviction_no_scale(monkeypatch):
    import config

    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "NARRATIVE_LOW_CONVICTION_FORCE_ONE", True)
    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    bets = [
        {
            "fight_id": "f1",
            "pick": "A",
            "edge_pct": 5.0,
            "kelly_stake_usd": 10.0,
            "suggested_stake": 8.0,
        }
    ]
    grok = {
        "ok": True,
        "picks": [
            {
                "id": "f1",
                "kelly_adjustment": 0.8,
                "narrative_edge": "Thin edge",
                "conviction": "low",
            }
        ],
    }
    out = apply_grok_kelly_adjustments(bets, grok)
    assert out[0]["kelly_stake_usd"] == 10.0
    assert out[0]["suggested_stake"] == 8.0
    assert out[0]["grok_kelly_factor"] == 1.0
    assert out[0]["grok_narrative"] == "Thin edge"


def test_apply_noop_when_failed():
    bets = [{"fight_id": "f1", "pick": "A", "edge_pct": 4.0, "kelly_stake_usd": 10.0}]
    out = apply_grok_kelly_adjustments(bets, {"ok": False, "picks": []})
    assert out[0]["kelly_stake_usd"] == 10.0
    assert out[0]["edge_pct"] == 4.0
    assert out[0]["grok_kelly_factor"] == 1.0


def test_apply_none_result_unchanged():
    bets = [{"fight_id": "f1", "kelly_stake_usd": 10.0, "edge_pct": 3.0}]
    out = apply_grok_kelly_adjustments(bets, None)
    assert out[0]["kelly_stake_usd"] == 10.0


def test_strategy_wrapper(monkeypatch):
    import config
    from src.strategy import apply_narrative_tilt_after_model_sizing

    monkeypatch.setattr(config, "UFC_PROFILE", "paper")
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.10)
    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    bets = [{"fight_id": "f1", "pick": "A", "suggested_stake": 10.0, "edge_pct": 4.0}]
    grok = {
        "ok": True,
        "picks": [{"id": "f1", "kelly_adjustment": 1.05, "conviction": "high", "reason": "ok"}],
    }
    out = apply_narrative_tilt_after_model_sizing(bets, grok)
    assert out[0]["suggested_stake"] == pytest.approx(10.5, rel=1e-3)
    assert out[0]["edge_pct"] == 4.0
