"""Telegram alerts (skeleton)."""

from __future__ import annotations

from typing import Any

import requests

from sports_bot.core import config


def telegram_configured() -> bool:
    return bool(
        config.TELEGRAM_ENABLED and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID
    )


def send_telegram(text: str, *, parse_mode: str | None = None) -> dict[str, Any]:
    """Send a message to the configured chat. No-op when disabled."""
    if not telegram_configured():
        return {"ok": False, "skipped": True, "reason": "telegram_disabled_or_unconfigured"}
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text[:4000],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    resp = requests.post(url, json=payload, timeout=30)
    try:
        body = resp.json()
    except Exception:
        body = {"ok": False, "status_code": resp.status_code, "text": resp.text[:200]}
    return body


def format_pick_alert(
    *,
    event: str,
    selection: str,
    prob: float,
    odds: float | None,
    stake: float,
    confidence: str,
    reasons: str = "",
) -> str:
    odds_txt = f"{odds:.2f}" if odds else "-"
    return (
        f"🎯 {event}\n"
        f"Pick: {selection} ({prob:.0%}) @ {odds_txt}\n"
        f"Conf: {confidence} | Stake: ${stake:.2f}\n"
        f"{reasons[:300]}"
    )
