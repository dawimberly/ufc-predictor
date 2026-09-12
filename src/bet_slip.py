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
