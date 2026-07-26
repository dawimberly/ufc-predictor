"""Fractional Kelly sizing with hard caps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyConfig:
    fraction: float = 0.25
    min_edge: float = 0.03
    max_bet_fraction: float = 0.02
    min_stake: float = 1.0


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def raw_kelly(prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction of bankroll for a binary bet."""
    if decimal_odds <= 1.0 or not (0.0 < prob < 1.0):
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob
    edge = prob * b - q
    if edge <= 0:
        return 0.0
    return edge / b


def kelly_stake(
    bankroll: float,
    *,
    prob: float,
    decimal_odds: float,
    config: KellyConfig | None = None,
) -> float:
    """
    Return stake in currency units.

    Applies fractional Kelly, min-edge gate, and max bankroll fraction.
    """
    cfg = config or KellyConfig()
    if bankroll <= 0:
        return 0.0
    edge = prob - implied_prob(decimal_odds)
    if edge < cfg.min_edge:
        return 0.0
    frac = raw_kelly(prob, decimal_odds) * cfg.fraction
    frac = max(0.0, min(frac, cfg.max_bet_fraction))
    stake = bankroll * frac
    if stake < cfg.min_stake:
        return 0.0
    return round(stake, 2)
