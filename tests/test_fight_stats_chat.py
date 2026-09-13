"""Fight-stats chat answers from card data (no Ollama wait)."""

from __future__ import annotations

import pandas as pd

from src.grok_analysis import answer_ollama_chat, build_fight_stats_answer


def _sample_result() -> dict:
    return {
        "event": "UFC Fight Night",
        "predictions": pd.DataFrame(
            [
                {
                    "fighter_1": "Donte Johnson",
                    "fighter_2": "Eric McConico",
                    "p_f1": 0.71,
                    "p_f2": 0.29,
                    "edge_f1": -0.01,
                    "edge_f2": 0.12,
                    "edge_pct": -1.0,
                    "best_edge": -0.01,
                    "f1_odds": 1.45,
                    "f2_odds": 2.80,
                    "odds_matched": True,
                    "odds_source": "the_odds_api",
                    "interval_width": 0.83,
                    "disagreement": 0.05,
                    "predicted_winner": "Donte Johnson",
                }
            ]
        ),
        "fun_tiers": {
            "yellow": [
                {
                    "pick": "Donte Johnson",
                    "fight": "Donte Johnson vs Eric McConico",
                    "edge_pct": -1.0,
                    "bet_tier": "yellow",
                    "tier_reason": "wide_interval",
                }
            ]
        },
        "bet_slip": [],
        "skipped": [],
    }


def test_build_fight_stats_matches_last_name():
    text = build_fight_stats_answer(
        "tell me the stats on the mcconico fight",
        _sample_result(),
        event_label="UFC Fight Night",
    )
    assert text is not None
    assert "McConico" in text
    assert "Donte Johnson" in text
    assert "Model:" in text
    assert "Odds:" in text
    assert "Action:" in text
    assert "no Ollama wait" in text


def test_answer_ollama_chat_fight_stats_skips_llm(monkeypatch):
    called = {"ollama": False}

    def _boom(*_a, **_k):
        called["ollama"] = True
        raise AssertionError("ollama_complete should not run for fight stats")

    import src.ollama_client as oc

    monkeypatch.setattr(oc, "ollama_complete", _boom)
    monkeypatch.setattr(
        oc,
        "check_ollama_health",
        lambda force=False: {"reachable": True, "error_class": "ok"},
    )

    out = answer_ollama_chat(
        "stats on McConico",
        analysis_result=_sample_result(),
        event_label="UFC Fight Night",
    )
    assert out["source"] == "ha_fight_stats"
    assert out["model"] == "stats"
    assert "McConico" in out["answer"]
    assert called["ollama"] is False
