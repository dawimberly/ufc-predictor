"""End-to-end card pipeline: score → bank → size → alert."""

from __future__ import annotations

from typing import Any

from sports_bot.alerts.telegram import format_pick_alert, send_telegram
from sports_bot.bank.ledger import log_prediction
from sports_bot.bank.learning import lessons_prompt_block
from sports_bot.core import config
from sports_bot.core.confidence import confidence_label
from sports_bot.core.kelly import KellyConfig, kelly_stake
from sports_bot.data.odds import OddsLine, active_providers
from sports_bot.sports.base import ModelPick, SportAdapter
from sports_bot.sports.ufc.adapter import UfcAdapter


def _best_odds(selection: str, lines: list[OddsLine]) -> OddsLine | None:
    hits = [ln for ln in lines if ln.selection.lower() == selection.lower()]
    if not hits:
        return None
    return max(hits, key=lambda x: x.decimal_odds)


def run_card(
    adapter: SportAdapter | None = None,
    *,
    bankroll: float | None = None,
    send_alerts: bool = False,
) -> list[dict[str, Any]]:
    """
    Score upcoming matchups, attach odds/Kelly/confidence, log to bank.

    Returns list of actionable pick dicts.
    """
    config.ensure_dirs()
    adapter = adapter or UfcAdapter()
    bankroll = float(bankroll if bankroll is not None else config.INITIAL_BANKROLL)
    kelly_cfg = KellyConfig(
        fraction=config.KELLY_FRACTION,
        min_edge=config.MIN_EDGE,
        max_bet_fraction=config.MAX_BET_FRACTION,
    )

    lessons = lessons_prompt_block()
    lines: list[OddsLine] = []
    for provider in active_providers():
        try:
            lines.extend(provider.fetch_event_odds(""))
        except Exception:
            continue

    out: list[dict[str, Any]] = []
    for matchup in adapter.upcoming_matchups():
        pick: ModelPick = adapter.score_matchup(matchup)
        odds_line = _best_odds(pick.selection, lines)
        decimal = odds_line.decimal_odds if odds_line else None
        edge = (pick.prob - (1.0 / decimal)) if decimal and decimal > 1 else None
        stake = (
            kelly_stake(bankroll, prob=pick.prob, decimal_odds=decimal, config=kelly_cfg)
            if decimal
            else 0.0
        )
        conf = confidence_label(pick.prob)
        reasons = "; ".join(pick.reasons)
        if lessons:
            reasons = f"{reasons} | lessons_applied"

        if config.PREDICTION_BANK_AUTO_LOG:
            log_prediction(
                sport=adapter.sport,
                event=matchup.event,
                selection=pick.selection,
                opponent=matchup.selection_b
                if pick.selection == matchup.selection_a
                else matchup.selection_a,
                prob=pick.prob,
                odds=decimal,
                edge=edge,
                confidence=conf,
                stake=stake,
                reasons=reasons,
            )

        payload = {
            "event": matchup.event,
            "selection": pick.selection,
            "prob": pick.prob,
            "odds": decimal,
            "edge": edge,
            "stake": stake,
            "confidence": conf,
            "reasons": reasons,
            "book": odds_line.book if odds_line else None,
        }
        out.append(payload)

        if send_alerts and stake > 0:
            send_telegram(
                format_pick_alert(
                    event=matchup.event,
                    selection=pick.selection,
                    prob=pick.prob,
                    odds=decimal,
                    stake=stake,
                    confidence=conf,
                    reasons=reasons,
                )
            )
    return out
