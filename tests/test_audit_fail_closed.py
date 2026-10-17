"""Safety checks from the sizing / leakage audit."""

from __future__ import annotations

import logging
import sys
import types

import pandas as pd
import pytest
import requests

import config
from src.alerts import generate_alerts
from src.bet_tiers import TIER_BLUE, TIER_GREEN, zero_fun_only_stakes
from src.odds_providers.odds_api_client import odds_api_fail_closed_message, odds_api_get
from src.safe_io import redact_secrets
from src.strategy import (
    aggregate_overview_recommendations,
    allocate_card_budget_pct,
    build_auto_parlay_recommendations,
    budget_aware_alerts,
    edge_for_predicted_pick,
    strategy_from_profile,
)
from ufc_betting_bot.modules.dynamic_thresholds import ThresholdResult

pytestmark = pytest.mark.usefixtures("live_session")


def test_edge_for_predicted_pick_ignores_opponent_best_edge():
    row = pd.Series(
        {
            "fighter_1": "A",
            "fighter_2": "B",
            "predicted_winner": "A",
            "edge_f1": -0.04,
            "edge_f2": 0.15,
            "best_edge": 0.15,
            "best_edge_side": "f2",
            "edge_pct": -4.0,
        }
    )
    edge, pick = edge_for_predicted_pick(row)
    assert pick == "A"
    assert edge == -0.04


def test_dynamic_singles_edge_cannot_undercut_live_floor(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "HEALTH_FEEDBACK_ENABLED", False)

    class _Thresholds:
        alert_min_edge = 0.06
        parlay_min_edge = 0.05
        parlay_min_combined_prob = 0.30
        parlay_min_ev = 0.10

    monkeypatch.setattr(
        "ufc_betting_bot.modules.dynamic_thresholds.get_profile_thresholds",
        lambda *args, **kwargs: _Thresholds(),
    )
    strat = strategy_from_profile(bankroll=1000.0, use_dynamic_thresholds=True)
    assert strat.min_edge >= 0.09
    assert strat.parlay_min_edge >= 0.08


def test_redact_secrets_strips_api_key_and_session():
    raw = (
        "HTTP 401 for url: https://api.example/odds?apiKey=super-secret&regions=us "
        "session=abc123token"
    )
    cleaned = redact_secrets(raw)
    assert "super-secret" not in cleaned
    assert "abc123token" not in cleaned
    assert "apiKey=<redacted>" in cleaned
    assert "session=<redacted>" in cleaned


def test_fail_closed_message_redacts_api_key():
    msg = odds_api_fail_closed_message(
        status_code=500,
        detail="HTTP 500 for url: https://api.example/odds?apiKey=super-secret&regions=us",
    )
    assert "super-secret" not in msg
    assert "apiKey=<redacted>" in msg


def _fresh_quote(**extra) -> dict:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    fields = {
        "quote_timestamp": (now - timedelta(minutes=5)).isoformat(),
        "commence_time": (now + timedelta(days=3)).isoformat(),
        "card_event_id": "evt-card",
        "quote_event_id": "evt-card",
        "bookmaker": "DraftKings",
        "odds_source": "DraftKings",
        "bookmaker_count": 1,
        "price_kind": "book",
        "quote_match": "exact",
        "fighter_1": "A",
        "fighter_2": "B",
        "quote_fighter_1": "A",
        "quote_fighter_2": "B",
        "quote_f1_odds": 1.80,
        "quote_f2_odds": 2.50,
    }
    fields.update(extra)
    return fields


def _strong_ticket(**extra) -> dict:
    ticket = {
        "fight": "A vs B",
        "fight_id": "ab",
        "pick": "A",
        "edge": 0.20,
        "prob": 0.80,
        "confidence": "high",
        "decimal_odds": 1.80,
        "uncertainty_action": "allow",
    }
    ticket.update(extra)
    return ticket


def test_rejected_mismatch_and_zero_kelly_stay_unfunded():
    skipped = generate_alerts(
        pd.DataFrame(
            [
                {
                    "fight_id": "neg",
                    "event_name": "UFC Test",
                    "fighter_1": "Model Pick",
                    "fighter_2": "Opponent",
                    "predicted_winner": "Model Pick",
                    "predicted_prob": 0.74,
                    "prob_f1_win": 0.74,
                    "prob_f2_win": 0.26,
                    "confidence_label": "high",
                    "ensemble_disagreement": 0.02,
                    "interval_width": 0.10,
                    "f1_odds": 1.20,
                    "f2_odds": 4.50,
                    "edge_f1": -0.08,
                    "edge_f2": 0.30,
                    "best_edge": 0.30,
                    "best_edge_side": "f2",
                    "edge_pct": -8.0,
                }
            ]
        ),
        min_edge=0.05,
        risk_metrics={"available": False},
    )
    assert skipped["singles"] == []

    rejected = allocate_card_budget_pct(
        [_strong_ticket(uncertainty_action="skip", kelly_stake_usd=25.0)],
        100.0,
        profile="paper",
        inplace=False,
    )
    zero = allocate_card_budget_pct(
        [_strong_ticket(kelly_stake_usd=0)],
        100.0,
        profile="paper",
        inplace=False,
    )
    assert float(rejected[0]["suggested_stake"]) == 0.0
    assert float(zero[0]["suggested_stake"]) == 0.0
    assert float(zero[0]["stake_pct"]) == 0.0


def test_kelly_ceiling_missing_invalid_zero_and_resize():
    missing = allocate_card_budget_pct(
        [_strong_ticket()],
        100.0,
        profile="paper",
        inplace=False,
    )
    assert float(missing[0]["suggested_stake"]) > 0.0

    invalid = allocate_card_budget_pct(
        [_strong_ticket(kelly_stake_usd="not-a-ceiling")],
        100.0,
        profile="paper",
        inplace=False,
    )
    nan_ceiling = allocate_card_budget_pct(
        [_strong_ticket(kelly_stake_usd=float("nan"))],
        100.0,
        profile="paper",
        inplace=False,
    )
    assert float(invalid[0]["suggested_stake"]) == 0.0
    assert float(nan_ceiling[0]["suggested_stake"]) == 0.0

    first = allocate_card_budget_pct(
        [_strong_ticket(kelly_stake_usd=1.25)],
        100.0,
        profile="paper",
        inplace=False,
    )
    second = allocate_card_budget_pct(first, 100.0, profile="paper", inplace=False)
    assert float(first[0]["suggested_stake"]) <= 1.25 + 1e-9
    assert float(second[0]["suggested_stake"]) <= 1.25 + 1e-9
    assert float(first[0]["stake_pct"]) == round(
        100.0 * float(first[0]["suggested_stake"]) / 100.0, 1
    )


def test_live_requires_valid_kelly_ceiling():
    """Live does not invent a ceiling from a missing field or an old stake."""
    cases = [
        _strong_ticket(suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd=None, suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd="", suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd="not-a-ceiling", suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd=float("nan"), suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd=float("inf"), suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd=-3.0, suggested_stake=40.0),
        _strong_ticket(kelly_stake_usd=0, suggested_stake=40.0),
    ]
    for ticket in cases:
        out = allocate_card_budget_pct([ticket], 100.0, profile="live", inplace=False)
        assert float(out[0]["suggested_stake"]) == 0.0
        assert float(out[0]["stake_pct"]) == 0.0

    first = allocate_card_budget_pct(
        [_strong_ticket(**_fresh_quote(), kelly_stake_usd=1.25, suggested_stake=40.0)],
        100.0,
        profile="live",
        inplace=False,
    )
    second = allocate_card_budget_pct(first, 100.0, profile="live", inplace=False)
    assert 0.0 < float(first[0]["suggested_stake"]) <= 1.25 + 1e-9
    assert float(second[0]["suggested_stake"]) <= 1.25 + 1e-9


def test_narrative_resize_stays_under_kelly_ceiling(monkeypatch):
    from src.grok_analysis import apply_grok_kelly_adjustments

    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "NARRATIVE_TILT_ENABLED", True)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MIN", 0.90)
    monkeypatch.setattr(config, "GROK_KELLY_ADJ_MAX", 1.25)
    bets = [
        {
            "fight_id": "f1",
            "pick": "A",
            "fight": "A vs B",
            "edge": 0.12,
            "edge_pct": 12.0,
            "kelly_stake_usd": 4.0,
            "suggested_stake": 4.0,
            "card_pool_usd": 100.0,
            "stake_pct": 4.0,
            "decimal_odds": 1.80,
            **_fresh_quote(),
        }
    ]
    grok = {
        "ok": True,
        "picks": [
            {
                "id": "f1",
                "kelly_adjustment": 1.20,
                "conviction": "high",
                "narrative_edge": "ok",
            }
        ],
    }
    out = apply_grok_kelly_adjustments(bets, grok)
    assert float(out[0]["kelly_stake_usd"]) == 4.0
    assert float(out[0]["suggested_stake"]) <= 4.0 + 1e-9
    again = allocate_card_budget_pct(out, 100.0, profile="live", inplace=False)
    assert float(again[0]["suggested_stake"]) <= 4.0 + 1e-9


def test_paper_override_cap_is_card_pool_percent(monkeypatch):
    monkeypatch.setattr(config, "PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC", 0.01)
    out = allocate_card_budget_pct(
        [
            _strong_ticket(
                uncertainty_action="tighten",
                uncertainty_reason="paper_wide_override",
                kelly_stake_usd=80.0,
                bankroll=1000.0,
            )
        ],
        100.0,
        profile="paper",
        inplace=False,
    )
    stake = float(out[0]["suggested_stake"])
    assert stake <= 10.0 + 1e-9
    assert stake > 0.0
    assert float(out[0]["stake_pct"]) == round(100.0 * stake / 100.0, 1)
    assert float(out[0]["stake_pct"]) != round(100.0 * stake / 1000.0, 1)


def test_live_profile_switch_drops_cached_paper_override():
    """Same allocation the dashboard uses after Paper → Live re-render."""
    override = _strong_ticket(
        fight_id="override",
        fight="Paper vs Live",
        uncertainty_action="tighten",
        uncertainty_reason="paper_wide_override",
        kelly_stake_usd=25.0,
        suggested_stake=25.0,
    )
    blue = _strong_ticket(
        fight_id="blue",
        fight="Blue vs Stay",
        pick="Blue",
        kelly_stake_usd=4.0,
        **_fresh_quote(
            fighter_1="Blue",
            fighter_2="Stay",
            quote_fighter_1="Blue",
            quote_fighter_2="Stay",
            card_event_id="evt-blue",
            quote_event_id="evt-blue",
        ),
    )
    # One enabled book gives a positive card pool. It has no alerts, so the
    # Overview cache is what Live reallocation sees.
    budget = {
        "total_bankroll": 1000.0,
        "card_budget": 100.0,
        "use_betnow": False,
        "use_draftkings": True,
        "use_mybookie": False,
        "draftkings_balance": 100.0,
    }
    books = {"Overview": {"alerts": {"singles": [override, blue], "parlays": []}}}
    overview = aggregate_overview_recommendations(books, budget, profile="live")
    ids = {str(s.get("fight_id")) for s in overview["singles"]}
    assert "override" not in ids
    assert "blue" in ids
    assert all(float(s.get("suggested_stake") or 0) > 0 for s in overview["singles"])

    rendered = budget_aware_alerts(
        {"singles": [dict(override)], "parlays": [], "bankroll": 1000.0},
        {"total_bankroll": 1000.0, "card_budget": 100.0},
        "Odds API",
        profile="live",
    )
    assert rendered["singles"] == []


def test_dynamic_thresholds_respect_ha_floors(monkeypatch):
    monkeypatch.setattr(config, "UFC_PROFILE", "live")
    monkeypatch.setattr(config, "HEALTH_FEEDBACK_ENABLED", False)
    from src.parlay_builder import strategy_from_thresholds
    from src.predictor import resolve_edge_thresholds

    monkeypatch.setattr(
        "ufc_betting_bot.modules.dynamic_thresholds.get_profile_thresholds",
        lambda *args, **kwargs: ThresholdResult(
            alert_min_edge=0.06,
            parlay_min_edge=0.05,
            parlay_min_combined_prob=0.30,
            parlay_min_ev=0.10,
            profile="live",
        ),
    )
    resolved = resolve_edge_thresholds(
        pd.DataFrame([{"event_date": "2026-01-01", "prob_f1_win": 0.7}]),
        bankroll=1000.0,
        use_dynamic=True,
    )
    assert float(resolved["alert_min_edge"]) >= 0.09
    assert float(resolved["parlay_min_edge"]) >= 0.08

    strat = strategy_from_thresholds(
        ThresholdResult(
            alert_min_edge=0.06,
            parlay_min_edge=0.05,
            parlay_min_combined_prob=0.30,
            parlay_min_ev=0.10,
            profile="live",
        )
    )
    assert strat.min_edge >= 0.09
    assert strat.parlay_min_edge >= 0.08


def test_fun_only_and_auto_parlays_stay_zero_while_blue_remains():
    items = zero_fun_only_stakes(
        [
            {"bet_tier": TIER_GREEN, "suggested_stake": 12.0, "stake_usd": 12.0, "stake_pct": 8.0},
            {"bet_tier": TIER_BLUE, "suggested_stake": 6.0, "stake_usd": 6.0, "stake_pct": 4.0, "pick": "A"},
        ]
    )
    assert items[0]["suggested_stake"] == 0.0
    assert items[0]["stake_usd"] == 0.0
    assert items[0]["stake_pct"] == 0.0
    assert items[0]["fun_bet"] is True
    assert items[1]["suggested_stake"] == 6.0
    assert items[1]["stake_pct"] == 4.0

    card = pd.DataFrame(
        [
            {"fight_id": "1", "fighter_1": "A", "fighter_2": "B", "predicted_winner": "A", "predicted_prob": 0.80},
            {"fight_id": "2", "fighter_1": "C", "fighter_2": "D", "predicted_winner": "C", "predicted_prob": 0.76},
            {"fight_id": "3", "fighter_1": "E", "fighter_2": "F", "predicted_winner": "E", "predicted_prob": 0.72},
        ]
    )
    parlays = build_auto_parlay_recommendations(card)
    assert parlays
    assert all(float(p["suggested_stake"]) == 0.0 for p in parlays)
    assert all(float(p["stake_pct"]) == 0.0 for p in parlays)
    assert all(float(p["stake_usd"]) == 0.0 for p in parlays)

    kept = generate_alerts(
        pd.DataFrame(
            [
                {
                    "fight_id": "blue",
                    "event_name": "UFC Test",
                    "fighter_1": "Jon Jones",
                    "fighter_2": "Stipe Miocic",
                    "predicted_winner": "Jon Jones",
                    "predicted_prob": 0.74,
                    "prob_f1_win": 0.74,
                    "prob_f2_win": 0.26,
                    "confidence_label": "high",
                    "ensemble_disagreement": 0.02,
                    "interval_width": 0.10,
                    "f1_odds": 1.55,
                    "f2_odds": 2.50,
                    "edge_f1": 0.12,
                    "edge_f2": -0.10,
                    "best_edge": 0.12,
                    "best_edge_side": "f1",
                    "edge_pct": 12.0,
                }
            ]
        ),
        min_edge=0.05,
        risk_metrics={"available": False},
    )
    assert len(kept["singles"]) == 1
    assert kept["singles"][0]["pick"] == "Jon Jones"
    assert float(kept["singles"][0]["suggested_stake"]) > 0.0


def test_betnow_resolve_log_omits_token(caplog, monkeypatch):
    import src.odds_providers.betnow_scraper as betnow

    monkeypatch.setattr(betnow, "_ACTIVE_SESSION_TOKEN", None)
    monkeypatch.setattr(betnow, "_env_session_token", lambda: "fake-session-token-value")
    with caplog.at_level(logging.INFO):
        token = betnow.resolve_betnow_session_token(force_discover=True)
    assert token == "fake-session-token-value"
    assert "fake-session-token-value" not in caplog.text
    assert "source=env(BETNOW_SESSION)" in caplog.text


def test_betnow_discovery_and_fetch_logs_omit_token(caplog, monkeypatch):
    import src.odds_providers.betnow_scraper as betnow

    class _Resp:
        url = "https://example.test/ufc"
        headers: dict[str, str] = {}
        cookies: list = []
        text = "session=fake-discovered-token-xyz"
        history: list = []

        def raise_for_status(self) -> None:
            raise requests.HTTPError(
                "401 Client Error for url: https://example.test/odds?session=fake-fetch-token"
            )

    monkeypatch.setattr(betnow.requests, "get", lambda *a, **k: _Resp())
    with caplog.at_level(logging.DEBUG):
        found = betnow._discover_session_token({})
        betnow._log_auth_materials(
            "fake-auth-token-xyz",
            "",
            url="https://example.test/odds?session=fake-auth-token-xyz",
        )
        betnow._fetch_and_parse(
            "https://example.test/odds?session=fake-fetch-token",
            {},
            label="session",
        )
    assert found == "fake-discovered-token-xyz"
    for secret in (
        "fake-discovered-token-xyz",
        "fake-auth-token-xyz",
        "fake-fetch-token",
    ):
        assert secret not in caplog.text


def test_betnow_selenium_exception_log_omits_token(caplog, monkeypatch):
    import src.odds_providers.betnow_scraper as betnow

    selenium = types.ModuleType("selenium")
    webdriver = types.ModuleType("selenium.webdriver")
    chrome = types.ModuleType("selenium.webdriver.chrome")
    options_mod = types.ModuleType("selenium.webdriver.chrome.options")

    class _Options:
        def add_argument(self, *_args, **_kwargs):
            return None

    class _Chrome:
        def __init__(self, options=None):
            raise RuntimeError("nav https://example.test/ufc?session=fake-selenium-token")

    options_mod.Options = _Options
    webdriver.Chrome = _Chrome
    webdriver.chrome = chrome
    selenium.webdriver = webdriver
    saved = {name: sys.modules.get(name) for name in (
        "selenium",
        "selenium.webdriver",
        "selenium.webdriver.chrome",
        "selenium.webdriver.chrome.options",
    )}
    sys.modules["selenium"] = selenium
    sys.modules["selenium.webdriver"] = webdriver
    sys.modules["selenium.webdriver.chrome"] = chrome
    sys.modules["selenium.webdriver.chrome.options"] = options_mod
    monkeypatch.setattr(betnow, "_ACTIVE_SESSION_TOKEN", "already-set")
    monkeypatch.setattr(betnow, "resolve_betnow_session_token", lambda **k: "")
    try:
        with caplog.at_level(logging.WARNING):
            betnow._scrape_with_selenium()
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert "fake-selenium-token" not in caplog.text


def test_odds_api_request_exception_omits_key(monkeypatch):
    monkeypatch.setattr(
        "src.odds_providers.odds_api_client.refresh_odds_api_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.odds_providers.odds_api_client.ensure_live_odds_api_allowed",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "src.odds_providers.odds_api_client.normalize_odds_api_key",
        lambda raw: "fake-odds-api-key",
    )
    monkeypatch.setattr(config, "ODDS_API_KEY", "fake-odds-api-key")

    class _Session:
        def get(self, url, params=None, timeout=None):
            key = (params or {}).get("apiKey")
            raise requests.HTTPError(f"401 Client Error for url: {url}?apiKey={key}&regions=us")

    monkeypatch.setattr(
        "src.odds_providers.odds_api_client._fresh_session",
        lambda: _Session(),
    )
    with pytest.raises(requests.HTTPError) as exc_info:
        odds_api_get("/sports/mma_mixed_martial_arts/odds")
    text = str(exc_info.value)
    assert "fake-odds-api-key" not in text
    assert "apiKey=<redacted>" in text


def test_draftkings_fallback_logs_omit_key(caplog, monkeypatch):
    import src.odds_providers.draftkings as dk

    monkeypatch.setattr(dk, "_read_cache", lambda path: None)

    def _boom(**kwargs):
        raise RuntimeError("consensus failed apiKey=fake-dk-key")

    monkeypatch.setattr(dk, "_consensus_odds", _boom)

    def _scrape(**kwargs):
        raise RuntimeError("scrape failed session=fake-scrape-token")

    monkeypatch.setattr(
        "src.odds_providers.odds_fallback.fetch_book_scraper_odds",
        _scrape,
    )
    with caplog.at_level(logging.WARNING):
        out = dk._fallback_odds(force_refresh=False, reason="401 Unauthorized")
    assert out.empty
    assert "fake-dk-key" not in caplog.text
    assert "fake-scrape-token" not in caplog.text


def test_mybookie_odds_api_error_omits_key(monkeypatch):
    import src.odds_providers.mybookie_scraper as mb
    from src.predictor import OddsAPIError

    monkeypatch.setattr(config, "ODDS_API_KEY", "fake-mybookie-api-key")
    monkeypatch.setattr(config, "ODDS_API_BASE_URL", "https://example.test")
    monkeypatch.setattr(
        "src.odds_providers.odds_api_client.ensure_live_odds_api_allowed",
        lambda **kwargs: None,
    )

    def _raise(*_args, **_kwargs):
        raise requests.HTTPError(
            "403 Client Error for url: https://example.test/odds?apiKey=fake-mybookie-api-key"
        )

    monkeypatch.setattr(mb.requests, "get", _raise)
    with pytest.raises(OddsAPIError) as exc_info:
        mb._fetch_odds_api_fallback()
    text = str(exc_info.value)
    assert "fake-mybookie-api-key" not in text
    assert "apiKey=<redacted>" in text


def test_draftkings_prop_skip_log_omits_key(caplog, monkeypatch):
    import src.odds_providers.draftkings_props as props

    monkeypatch.setattr(config, "ENABLE_PROPS", True)
    monkeypatch.setattr(config, "ODDS_API_KEY", "fake-dk-prop-key")
    monkeypatch.setattr(config, "ODDS_API_BASE_URL", "https://example.test")
    monkeypatch.setattr(config, "ODDS_API_SPORT", "mma_mixed_martial_arts")
    monkeypatch.setattr(
        "src.odds_providers.odds_api_client.ensure_live_odds_api_allowed",
        lambda **kwargs: None,
    )

    class _Events:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, str]]:
            return [{"id": "e1", "home_team": "Alpha", "away_team": "Beta"}]

    def _get(url, params=None, timeout=None):
        if str(url).endswith("/events"):
            return _Events()
        key = (params or {}).get("apiKey")
        raise requests.HTTPError(f"500 Client Error for url: {url}?apiKey={key}")

    monkeypatch.setattr(props.requests, "get", _get)
    with caplog.at_level(logging.DEBUG):
        out = props.fetch_draftkings_prop_odds(force_refresh=True)
    assert out.empty
    assert "fake-dk-prop-key" not in caplog.text
    assert "apiKey=<redacted>" in caplog.text


def _overview_app(singles: list[dict], *, profile: str = "Live"):
    import config

    # Dashboard import may run bootstrap; keep the test safety file (never the real cache).
    safety_path = config.CIRCUIT_BREAKER_STATE_PATH
    import src.ufc_dashboard as dash

    config.CIRCUIT_BREAKER_STATE_PATH = safety_path
    dash.pd = pd

    class _Var:
        def get(self) -> str:
            return profile

    class _Payload:
        books = {
            "Overview": {
                "alerts": {"singles": singles, "parlays": []},
                "predictions": pd.DataFrame(),
            }
        }
        combined = pd.DataFrame()
        event_label = "UFC Test"

    class _App:
        _payload = _Payload()
        _grok_result = None
        _budget_state = {
            "total_bankroll": 1000.0,
            "card_budget": 100.0,
            "use_betnow": False,
            "use_draftkings": True,
            "use_mybookie": False,
            "draftkings_balance": 100.0,
        }
        profile_var = _Var()

        def _profile_from_menu(self, display: str) -> str:
            return "live" if "live" in str(display).lower() else "paper"

        def _current_budget_state(self) -> dict:
            return self._budget_state

        def _allowed_fight_keys(self, combined) -> set[str]:
            return set()

    return dash.UFCDashboardApp._overview_recommendations(_App())


def test_overview_aggregation_error_returns_no_sized_tickets(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError(
            "overview failed https://example.test?session=fake-overview-session"
        )

    monkeypatch.setattr("src.strategy.aggregate_overview_recommendations", _boom)
    override = _strong_ticket(
        fight_id="override",
        fight="Paper vs Live",
        uncertainty_action="tighten",
        uncertainty_reason="paper_wide_override",
        kelly_stake_usd=25.0,
        suggested_stake=25.0,
        stake_usd=25.0,
    )
    items = _overview_app([override], profile="Live")
    stake_keys = (
        "suggested_stake",
        "stake_usd",
        "stake_pct",
        "raw_stake",
        "kelly_stake_usd",
        "kelly_pct",
        "max_safe_bet_usd",
    )
    for ticket in items:
        for key in stake_keys:
            assert float(ticket.get(key) or 0) == 0.0
    assert all(float(ticket.get("suggested_stake") or 0) == 0.0 for ticket in items)


def test_overview_success_funds_valid_blue_ticket():
    override = _strong_ticket(
        fight_id="override",
        fight="Paper vs Live",
        uncertainty_action="tighten",
        uncertainty_reason="paper_wide_override",
        kelly_stake_usd=25.0,
        suggested_stake=25.0,
    )
    blue = _strong_ticket(
        fight_id="blue",
        fight="Blue vs Stay",
        pick="Blue",
        kelly_stake_usd=4.0,
        suggested_stake=4.0,
        **_fresh_quote(
            fighter_1="Blue",
            fighter_2="Stay",
            quote_fighter_1="Blue",
            quote_fighter_2="Stay",
            card_event_id="evt-blue",
            quote_event_id="evt-blue",
        ),
    )
    items = _overview_app([override, blue], profile="Live")
    funded = [
        ticket
        for ticket in items
        if str(ticket.get("fight_id")) == "blue" or str(ticket.get("pick")) == "Blue"
    ]
    assert funded
    assert float(funded[0].get("suggested_stake") or 0) > 0.0
    assert float(funded[0].get("suggested_stake") or 0) <= 4.0 + 1e-9
    for ticket in items:
        if "paper_wide_override" in str(ticket.get("uncertainty_reason") or ""):
            assert float(ticket.get("suggested_stake") or 0) == 0.0


def test_cookie_capture_and_fallback_logs_omit_credentials(caplog, monkeypatch):
    import src.odds_providers.action_network as action
    import src.odds_providers.cookie_capture as cookies
    import src.odds_providers.odds_fallback as fallback

    selenium = types.ModuleType("selenium")
    webdriver = types.ModuleType("selenium.webdriver")
    chrome = types.ModuleType("selenium.webdriver.chrome")
    options_mod = types.ModuleType("selenium.webdriver.chrome.options")

    class _Options:
        def add_argument(self, *_args, **_kwargs):
            return None

        def add_experimental_option(self, *_args, **_kwargs):
            return None

    class _Chrome:
        def __init__(self, options=None):
            raise RuntimeError(
                "nav https://example.test/ufc?session=fake-selenium-session "
                "cookie=fake-selenium-cookie"
            )

    options_mod.Options = _Options
    webdriver.Chrome = _Chrome
    webdriver.chrome = chrome
    selenium.webdriver = webdriver
    saved = {
        name: sys.modules.get(name)
        for name in (
            "selenium",
            "selenium.webdriver",
            "selenium.webdriver.chrome",
            "selenium.webdriver.chrome.options",
        )
    }
    sys.modules["selenium"] = selenium
    sys.modules["selenium.webdriver"] = webdriver
    sys.modules["selenium.webdriver.chrome"] = chrome
    sys.modules["selenium.webdriver.chrome.options"] = options_mod
    spec = cookies.BookCaptureSpec(
        key="betnow",
        label="BetNow",
        start_url="https://example.test/ufc?session=fake-selenium-session",
        cookie_env="BETNOW_COOKIE",
        session_env="BETNOW_SESSION",
    )
    secrets = (
        "fake-selenium-session",
        "fake-selenium-cookie",
        "fake-cookie-header",
        "fake-fallback-key",
        "fake-fallback-session",
        "fake-action-key",
        "fake-action-cookie",
    )
    try:
        with caplog.at_level(logging.DEBUG):
            captured = cookies._capture_with_selenium(spec, timeout_sec=1, progress=None)
            monkeypatch.setattr(
                cookies.webbrowser,
                "open",
                lambda *_a, **_k: (_ for _ in ()).throw(
                    RuntimeError("open failed cookie=fake-cookie-header")
                ),
            )
            cookies._capture_fallback_open_only(spec, progress=None)

            def _outer(*_a, **_k):
                raise RuntimeError(
                    "wrapper https://example.test/?session=fake-selenium-session "
                    "Cookie: auth=fake-selenium-cookie"
                )

            monkeypatch.setattr(cookies, "_backend_available", lambda: "selenium")
            monkeypatch.setattr(cookies, "_capture_with_selenium", _outer)
            wrapped = cookies.capture_book_login(spec, force=True, timeout_sec=1)

            def _action_get(*_a, **_k):
                raise requests.HTTPError(
                    "401 for url: https://example.test/scoreboard?apiKey=fake-action-key "
                    "Cookie: session=fake-action-cookie"
                )

            monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", True)
            monkeypatch.setattr(action.requests, "get", _action_get)
            action_out = action.fetch_action_network_odds(force_refresh=True)

            def _odds_boom(*_a, **_k):
                raise RuntimeError(
                    "403 for url: https://example.test/odds?apiKey=fake-fallback-key"
                    "&session=fake-fallback-session Cookie: session=fake-cookie-header"
                )

            monkeypatch.setattr(config, "ACTION_NETWORK_ENABLED", False)
            monkeypatch.setattr(config, "DRAFTKINGS_ENABLED", False)
            monkeypatch.setattr(config, "BETNOW_ENABLED", False)
            monkeypatch.setattr(config, "MYBOOKIE_ENABLED", False)
            monkeypatch.setattr("src.predictor.fetch_ufc_odds", _odds_boom)
            frame, err = fallback._try_fetch("OddsAPI", _odds_boom, force_refresh=True)
            odds_frame, meta = fallback.fetch_best_available_odds(force_refresh=True)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    blob = "\n".join(
        [
            caplog.text,
            captured.message,
            wrapped.message,
            err or "",
            str(meta.get("warning") or ""),
        ]
    )
    for secret in secrets:
        assert secret not in blob
    assert action_out.empty
    assert frame.empty
    assert odds_frame.empty
    assert meta.get("fail_closed") is True
    assert "apiKey=<redacted>" in blob
    assert "session=<redacted>" in blob or "Cookie: <redacted>" in blob
