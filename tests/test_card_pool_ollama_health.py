"""Card-pool status line + Ollama health / no-odds fail-closed."""

from __future__ import annotations

from src.grok_analysis import analyze_card_with_ollama, books_have_usable_odds
from src.ollama_client import classify_ollama_error
from src.strategy import available_card_budget_text, format_card_allocation_status


def test_format_card_allocation_status_auto():
    line = format_card_allocation_status(
        auto_card_usd=55, allocated_usd=12.5, n_tickets=3
    )
    assert "Auto card $55.00" in line
    assert "Allocated $12.50" in line
    assert "(23%)" in line
    assert "Tickets 3" in line


def test_available_card_budget_text_no_legacy_pool_wording():
    txt = available_card_budget_text(
        {
            "total_bankroll": 100,
            "card_budget": 55,
            "use_betnow": True,
            "use_draftkings": True,
            "use_mybookie": True,
            "betnow_balance": 100,
            "draftkings_balance": 100,
            "mybookie_balance": 100,
        },
        profile="paper",
    )
    assert "Available this card" not in txt
    assert "Auto card" in txt


def test_classify_ollama_error_classes():
    assert classify_ollama_error("connection refused") == "offline"
    assert classify_ollama_error("timed out after 60s") == "timeout"
    assert classify_ollama_error("model missing: foo") == "model_missing"


def test_books_have_usable_odds():
    assert books_have_usable_odds({}) is False
    assert books_have_usable_odds({"BetNow": {"odds_matched": 0, "alerts": {}}}) is False
    assert books_have_usable_odds({"BetNow": {"odds_matched": 3}}) is True


def test_analyze_no_usable_odds_fail_closed(monkeypatch):
    import src.ollama_client as oc

    monkeypatch.setattr(
        oc,
        "check_ollama_health",
        lambda force=False: {
            "reachable": False,
            "error_class": "offline",
            "banner": "Ollama offline — showing HA tickets",
            "latency_ms": 12,
            "error": "unreachable",
            "resolved_model": None,
        },
    )
    result = analyze_card_with_ollama(
        {},
        {"total_bankroll": 100, "card_budget": 55},
        event_label="Test",
        use_cache=False,
    )
    assert result.get("no_usable_odds") is True
    assert "no usable odds" in str(result.get("summary") or "").lower()
    assert result.get("bet_slip") == []
    assert "showing ha tickets" in str(result.get("health_banner") or "").lower()
    assert result.get("ollama_error_class") == "offline"
    assert result.get("error_class") == "ok"
