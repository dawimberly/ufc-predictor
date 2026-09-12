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
