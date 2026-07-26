"""Compact 'what bets to make' formatting for dashboard, CLI, and Ollama."""

from __future__ import annotations

from typing import Any


def short_name(fighter: str) -> str:
    """Prefer last name for slip-style lines (Chandler, not Michael Chandler)."""
    text = " ".join(str(fighter or "").strip().split())
    if not text:
        return "-"
    parts = text.split()
    if len(parts) == 1:
        return parts[0]
    # Keep hyphenated last names; drop common suffixes already attached.
    return parts[-1]


def short_reason(bet: dict[str, Any], *, max_len: int = 90) -> str:
    """One-line reason only — prefer brief / grok short text over long narratives."""
    for key in (
        "reason",
        "brief",
        "description",
        "grok_narrative",
        "narrative_edge",
        "reasoning",
    ):
        text = str(bet.get(key) or "").strip()
        if text:
            text = " ".join(text.replace("\n", " ").split())
            if len(text) > max_len:
                return text[: max_len - 1].rstrip() + "…"
            return text
    edge = bet.get("edge_pct")
    if edge is not None:
        try:
            return f"{float(edge):+.1f}% edge"
        except (TypeError, ValueError):
            pass
    return ""


def _american(bet: dict[str, Any]) -> str:
    am = str(bet.get("american_odds") or "").strip()
    if am and am not in ("-", "None", "nan"):
        return am
    dec = bet.get("decimal_odds") or bet.get("combined_odds")
    try:
        from src.parlay_builder import decimal_to_american

        d = float(dec)
        if d > 1:
            return decimal_to_american(d)
    except Exception:
        pass
    return ""


def _stake_usd(bet: dict[str, Any]) -> float:
    try:
        return max(0.0, float(bet.get("suggested_stake") or 0))
    except (TypeError, ValueError):
        return 0.0


def format_single_line(bet: dict[str, Any]) -> str:
    """e.g. $2 on Chandler -140 — strong wrestling edge at this price"""
    stake = _stake_usd(bet)
    pick = short_name(str(bet.get("pick") or bet.get("display_label") or bet.get("pick_line") or "-"))
    # Strip trailing " ML" from display labels
    if pick.upper().endswith(" ML"):
        pick = pick[:-3].strip()
    am = _american(bet)
    stake_txt = f"${stake:.0f}" if stake == int(stake) else f"${stake:.2f}"
    core = f"{stake_txt} on {pick}"
    if am:
        core = f"{core} {am}"
    reason = short_reason(bet)
    return f"{core} — {reason}" if reason else core


def format_parlay_line(parlay: dict[str, Any]) -> str:
    """e.g. $5 on 3-leg +650 (Chandler + Oliveira + …) — stacked favorites with edge"""
    stake = _stake_usd(parlay)
    stake_txt = f"${stake:.0f}" if stake == int(stake) else f"${stake:.2f}"
    n_legs = parlay.get("n_legs")
    if n_legs is None:
        bt = str(parlay.get("bet_type") or "")
        if "-Leg" in bt:
            n_legs = bt.split("-Leg")[0].strip()
    try:
        legs_n = int(n_legs) if n_legs is not None else 0
    except (TypeError, ValueError):
        legs_n = 0
    am = _american(parlay)
    legs = str(parlay.get("pick_line") or parlay.get("picks") or parlay.get("display_label") or "").strip()
    # Shorten long leg lists
    if len(legs) > 70:
        legs = legs[:67].rstrip() + "…"
    label = f"{legs_n}-leg" if legs_n else "parlay"
    core = f"{stake_txt} on {label}"
    if am:
        core = f"{core} {am}"
    if legs:
        core = f"{core} ({legs})"
    reason = short_reason(parlay)
    return f"{core} — {reason}" if reason else core


def format_bet_slip_block(
    singles: list[dict[str, Any]] | None,
    parlays: list[dict[str, Any]] | None = None,
    *,
    max_singles: int = 5,
    max_parlays: int = 2,
    title: str = "WHAT TO BET",
) -> str:
    """Plain-text slip for CLI / logs."""
    lines = [title, ""]
    singles = list(singles or [])[:max_singles]
    parlays = list(parlays or [])[:max_parlays]
    if singles:
        lines.append("Singles")
        for i, bet in enumerate(singles, 1):
            lines.append(f"  {i}. {format_single_line(bet)}")
        lines.append("")
    if parlays:
        lines.append("Parlays")
        for i, bet in enumerate(parlays, 1):
            lines.append(f"  {i}. {format_parlay_line(bet)}")
        lines.append("")
    if not singles and not parlays:
        lines.append("  No qualifying bets.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def split_overview_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a mixed overview list into singles vs parlays."""
    singles: list[dict[str, Any]] = []
    parlays: list[dict[str, Any]] = []
    for item in items or []:
        if item.get("is_parlay") or "parlay" in str(item.get("bet_type") or "").lower():
            parlays.append(item)
        else:
            singles.append(item)
    return singles, parlays
