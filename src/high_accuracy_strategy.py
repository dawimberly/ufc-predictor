"""
Hard-coded high-accuracy / low-volume betting strategy.

Singles are the primary product. Props are Over 1.5 only (reliability study:
other markets underperform under HA gates). Parlays are 2-leg only when both
legs are strong. Round robins are off. Ticket count per card is capped (1–4,
default 3 Paper / 2 Live).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- Immutable strategy identity ---
STRATEGY_NAME = "high_accuracy_low_volume"
STRATEGY_VERSION = "2026-07-26-over15-props"

# Only Over 1.5 is HA-actionable (prop reliability: Use; others Avoid).
ALLOWED_PROP_KEYS: frozenset[str] = frozenset({"over_1_5_rounds"})

# Round robins explicitly disabled (no builder ships them either).
ROUND_ROBINS_ENABLED = False

# Parlays: exactly 2 legs, both must clear strong floors.
PARLAY_MAX_LEGS = 2
PROP_PARLAYS_ENABLED = False  # no prop/mixed parlays under this strategy

# Ticket budget per card (singles + parlays count as tickets).
MAX_TICKETS_MIN = 1
MAX_TICKETS_MAX = 4
DEFAULT_MAX_TICKETS_PAPER = 3
DEFAULT_MAX_TICKETS_LIVE = 2

# Paper vs Live thresholds (Live stricter).
_PAPER = {
    "singles_min_model_prob": 0.70,
    "singles_min_confidence": "medium",
    "singles_min_edge": 0.06,
    "parlay_min_leg_prob": 0.68,
    "parlay_min_leg_edge": 0.055,
    "parlay_min_combined_prob": 0.40,
    "parlay_min_ev": 0.10,
    "parlay_min_leg_confidence": "high",
    "prop_min_model_prob": 0.78,
    "prop_min_live_edge": 0.05,
    "max_tickets_per_card": DEFAULT_MAX_TICKETS_PAPER,
    "require_live_prop_odds": True,
    # Path-risk / drawdown controls
    "max_parlay_share": 0.40,  # max fraction of card budget on parlays
    "stake_power": 0.95,  # used as soft concentration hint; conf/odds sizing is primary
    "drawdown_soft_pct": 0.25,
    "drawdown_hard_pct": 0.40,
    "drawdown_soft_mult": 0.70,  # card risk × after soft DD
    "drawdown_hard_mult": 0.45,
    # Confidence + odds-aware compounding (absolute % of card; sum ≤ 100%)
    "sizing_max_ticket_pct": 0.48,  # max share of card for one ticket
    "sizing_curve_gamma": 1.20,  # strength**gamma → stake (lower = more aggressive)
    "sizing_parlay_mult": 0.72,  # 2-leg discount vs singles
    "sizing_prop_mult": 0.92,
    "sizing_tighten_mult": 0.55,  # elevated uncertainty shrink
    "sizing_min_util_strength": 0.0,  # do not force full card spend
}

_LIVE = {
    "singles_min_model_prob": 0.72,
    "singles_min_confidence": "high",
    "singles_min_edge": 0.09,
    "parlay_min_leg_prob": 0.70,
    "parlay_min_leg_edge": 0.08,
    "parlay_min_combined_prob": 0.45,
    "parlay_min_ev": 0.18,
    "parlay_min_leg_confidence": "high",
    "prop_min_model_prob": 0.80,
    "prop_min_live_edge": 0.06,
    "max_tickets_per_card": DEFAULT_MAX_TICKETS_LIVE,
    "require_live_prop_odds": True,
    "max_parlay_share": 0.30,
    "stake_power": 0.70,
    "drawdown_soft_pct": 0.20,
    "drawdown_hard_pct": 0.35,
    "drawdown_soft_mult": 0.55,
    "drawdown_hard_mult": 0.30,
    # Live: less aggressive absolute stake curve
    "sizing_max_ticket_pct": 0.32,
    "sizing_curve_gamma": 1.45,
    "sizing_parlay_mult": 0.60,
    "sizing_prop_mult": 0.85,
    "sizing_tighten_mult": 0.40,
    "sizing_min_util_strength": 0.0,
}


def is_live() -> bool:
    try:
        import config

        return bool(config.is_live_profile())
    except Exception:
        return False


def profile_rules() -> dict[str, Any]:
    """Active Paper/Live numeric rules for this strategy."""
    return dict(_LIVE if is_live() else _PAPER)


def max_parlay_budget_share(*, live: bool | None = None) -> float:
    """Max fraction of a card's stake pool that may go to parlays."""
    rules = dict(_LIVE if (is_live() if live is None else live) else _PAPER)
    return float(rules.get("max_parlay_share") or (0.30 if live else 0.40))


def stake_allocation_power(*, live: bool | None = None) -> float:
    """
    Legacy concentration exponent (kept for diagnostics). Conf/odds sizing uses
    ``sizing_curve_gamma`` / ``sizing_max_ticket_pct`` instead.
    """
    rules = dict(_LIVE if (is_live() if live is None else live) else _PAPER)
    default = 0.70 if (is_live() if live is None else live) else 0.95
    return float(rules.get("stake_power") or default)


def sizing_curve_params(*, live: bool | None = None) -> dict[str, float]:
    """Paper/Live absolute stake-curve parameters for conf+odds compounding."""
    rules = dict(_LIVE if (is_live() if live is None else live) else _PAPER)
    use_live = bool(is_live() if live is None else live)
    return {
        "max_ticket_pct": float(
            rules.get("sizing_max_ticket_pct") or (0.32 if use_live else 0.48)
        ),
        "gamma": float(rules.get("sizing_curve_gamma") or (1.45 if use_live else 1.20)),
        "parlay_mult": float(rules.get("sizing_parlay_mult") or (0.60 if use_live else 0.72)),
        "prop_mult": float(rules.get("sizing_prop_mult") or (0.85 if use_live else 0.92)),
        "tighten_mult": float(
            rules.get("sizing_tighten_mult") or (0.40 if use_live else 0.55)
        ),
    }


def card_risk_drawdown_multiplier(drawdown_pct: float, *, live: bool | None = None) -> float:
    """
    Scale card risk after peak-to-trough drawdown.

    ``drawdown_pct`` is in [0, 1] (e.g. 0.25 = 25% off peak).
    """
    try:
        dd = max(0.0, float(drawdown_pct))
    except (TypeError, ValueError):
        return 1.0
    rules = dict(_LIVE if (is_live() if live is None else live) else _PAPER)
    hard = float(rules.get("drawdown_hard_pct") or 0.40)
    soft = float(rules.get("drawdown_soft_pct") or 0.25)
    hard_m = float(rules.get("drawdown_hard_mult") or 0.45)
    soft_m = float(rules.get("drawdown_soft_mult") or 0.70)
    if dd >= hard:
        return max(0.05, hard_m)
    if dd >= soft:
        # Linear blend soft→hard
        span = max(hard - soft, 1e-9)
        t = min(1.0, (dd - soft) / span)
        return max(0.05, soft_m + t * (hard_m - soft_m))
    return 1.0


def parlay_min_leg_confidence(*, live: bool | None = None) -> str:
    rules = dict(_LIVE if (is_live() if live is None else live) else _PAPER)
    return str(rules.get("parlay_min_leg_confidence") or "high")


def clamp_max_tickets(n: int | None) -> int:
    """Hard clamp ticket budget to 1–4."""
    try:
        v = int(n) if n is not None else profile_rules()["max_tickets_per_card"]
    except (TypeError, ValueError):
        v = int(profile_rules()["max_tickets_per_card"])
    return max(MAX_TICKETS_MIN, min(MAX_TICKETS_MAX, v))


def prop_allowed(prop_key: str) -> bool:
    return str(prop_key or "").strip().lower() in ALLOWED_PROP_KEYS


def strategy_rules_summary() -> dict[str, Any]:
    """
    Dashboard / decision_layer payload: human-readable active rules.
    """
    r = profile_rules()
    return {
        "name": STRATEGY_NAME,
        "version": STRATEGY_VERSION,
        "profile": "live" if is_live() else "paper",
        "focus": "singles",
        "singles": {
            "min_model_prob": float(r["singles_min_model_prob"]),
            "min_confidence": str(r["singles_min_confidence"]),
            "min_edge": float(r["singles_min_edge"]),
            "require_low_uncertainty": True,
        },
        "parlays": {
            "enabled": True,
            "max_legs": PARLAY_MAX_LEGS,
            "min_leg_prob": float(r["parlay_min_leg_prob"]),
            "min_leg_edge": float(r["parlay_min_leg_edge"]),
            "min_combined_prob": float(r["parlay_min_combined_prob"]),
            "min_ev": float(r["parlay_min_ev"]),
            "min_leg_confidence": str(r.get("parlay_min_leg_confidence") or "high"),
            "both_legs_strong": True,
            "max_share_of_card": float(r.get("max_parlay_share") or 0.4),
        },
        "path_risk": {
            "stake_power": float(r.get("stake_power") or 0.95),
            "drawdown_soft_pct": float(r.get("drawdown_soft_pct") or 0.25),
            "drawdown_hard_pct": float(r.get("drawdown_hard_pct") or 0.40),
            "drawdown_soft_mult": float(r.get("drawdown_soft_mult") or 0.70),
            "drawdown_hard_mult": float(r.get("drawdown_hard_mult") or 0.45),
        },
        "conf_odds_sizing": {
            "max_ticket_pct": float(r.get("sizing_max_ticket_pct") or 0.48),
            "curve_gamma": float(r.get("sizing_curve_gamma") or 1.20),
            "parlay_mult": float(r.get("sizing_parlay_mult") or 0.72),
            "prop_mult": float(r.get("sizing_prop_mult") or 0.92),
            "tighten_mult": float(r.get("sizing_tighten_mult") or 0.55),
            "mode": "absolute_strength_pct_sum_le_100",
        },
        "props": {
            "allowed_markets": sorted(ALLOWED_PROP_KEYS),
            "only_over_1_5": True,
            "min_model_prob": float(r["prop_min_model_prob"]),
            "min_live_edge": float(r["prop_min_live_edge"]),
            "require_live_odds": bool(r["require_live_prop_odds"]),
            "prop_parlays_enabled": PROP_PARLAYS_ENABLED,
        },
        "round_robins_enabled": ROUND_ROBINS_ENABLED,
        "max_tickets_per_card": clamp_max_tickets(r["max_tickets_per_card"]),
        "max_tickets_range": [MAX_TICKETS_MIN, MAX_TICKETS_MAX],
    }


def format_strategy_rules_line() -> str:
    """One-line status for book tabs / mode banner."""
    s = strategy_rules_summary()
    singles = s["singles"]
    props = s["props"]
    return (
        f"HA strategy: singles focus | "
        f"prob>={100 * float(singles['min_model_prob']):.0f}% "
        f"edge>={100 * float(singles['min_edge']):.0f}% "
        f"conf>={singles['min_confidence']} | "
        f"2-leg only (both strong) | "
        f"prop=Over1.5 @{100 * float(props['min_model_prob']):.0f}%+"
        f"/{100 * float(props['min_live_edge']):.0f}%+ live | "
        f"RR=off | max {s['max_tickets_per_card']} tickets/card"
    )


def format_strategy_rules_block() -> str:
    """Multi-line block for Risk tab."""
    s = strategy_rules_summary()
    singles = s["singles"]
    parlays = s["parlays"]
    props = s["props"]
    return "\n".join(
        [
            f"HIGH-ACCURACY STRATEGY [{s['profile'].upper()}] v{s['version']}",
            f"  Singles (primary): model>={100 * float(singles['min_model_prob']):.0f}% "
            f"| conf>={singles['min_confidence']} "
            f"| edge>={100 * float(singles['min_edge']):.0f}% "
            f"| low uncertainty required",
            f"  Parlays: {parlays['max_legs']}-leg only when both legs strong "
            f"(leg prob>={100 * float(parlays['min_leg_prob']):.0f}% "
            f"edge>={100 * float(parlays['min_leg_edge']):.0f}% "
            f"conf>={parlays.get('min_leg_confidence', 'high')}) "
            f"| max {100 * float(parlays.get('max_share_of_card') or 0):.0f}% of card budget",
            f"  Path risk: flatter stakes (power={float((s.get('path_risk') or {}).get('stake_power') or 0):.2f}) "
            f"| DD soft {100 * float((s.get('path_risk') or {}).get('drawdown_soft_pct') or 0):.0f}%→"
            f"×{float((s.get('path_risk') or {}).get('drawdown_soft_mult') or 1):.2f} "
            f"| hard {100 * float((s.get('path_risk') or {}).get('drawdown_hard_pct') or 0):.0f}%→"
            f"×{float((s.get('path_risk') or {}).get('drawdown_hard_mult') or 1):.2f}",
            f"  Props: Over 1.5 only | model>={100 * float(props['min_model_prob']):.0f}% "
            f"| live edge>={100 * float(props['min_live_edge']):.0f}% "
            f"| other props disabled | prop parlays off",
            f"  Round robins: disabled | Max tickets/card: {s['max_tickets_per_card']} "
            f"(range {s['max_tickets_range'][0]}–{s['max_tickets_range'][1]})",
        ]
    )


def log_strategy_block(
    reason: str,
    *,
    context: str = "",
    fight: str = "",
    prop_key: str = "",
    detail: str = "",
    **extra: Any,
) -> None:
    """Log when a potential bet is blocked by high-accuracy rules."""
    parts = [f"HA block [{reason}]"]
    if context:
        parts.append(f"ctx={context}")
    if fight:
        parts.append(f"fight={fight}")
    if prop_key:
        parts.append(f"prop={prop_key}")
    if detail:
        parts.append(detail)
    for k, v in extra.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


def apply_hardcoded_profile_defaults(profile: dict[str, Any], *, live: bool) -> dict[str, Any]:
    """
    Overlay hard-coded HA defaults onto a profile dict (mutates copy).
    Env overrides may still raise bars further; they cannot loosen below HA floors
    for the critical keys listed here.
    """
    out = dict(profile)
    rules = _LIVE if live else _PAPER
    # Floor: take the max of existing vs HA (never looser than strategy)
    out["alert_min_edge"] = max(float(out.get("alert_min_edge") or 0), float(rules["singles_min_edge"]))
    out["singles_min_model_prob"] = max(
        float(out.get("singles_min_model_prob") or 0),
        float(rules["singles_min_model_prob"]),
    )
    # Confidence: high > medium > low
    rank = {"low": 0, "medium": 1, "high": 2}
    need = str(rules["singles_min_confidence"])
    have = str(out.get("singles_min_confidence") or "low").lower()
    if rank.get(have, 0) < rank.get(need, 0):
        out["singles_min_confidence"] = need
    out["max_bets_per_card"] = clamp_max_tickets(
        out.get("max_bets_per_card", rules["max_tickets_per_card"])
    )
    out["parlay_min_edge"] = max(float(out.get("parlay_min_edge") or 0), float(rules["parlay_min_leg_edge"]))
    out["parlay_min_combined_prob"] = max(
        float(out.get("parlay_min_combined_prob") or 0),
        float(rules["parlay_min_combined_prob"]),
    )
    out["parlay_min_ev"] = max(float(out.get("parlay_min_ev") or 0), float(rules["parlay_min_ev"]))
    out["parlay_max_legs"] = PARLAY_MAX_LEGS
    out["prop_min_model_prob"] = max(
        float(out.get("prop_min_model_prob") or 0),
        float(rules["prop_min_model_prob"]),
    )
    out["prop_min_edge"] = max(float(out.get("prop_min_edge") or 0), float(rules["prop_min_live_edge"]))
    out["alert_max_parlays"] = min(int(out.get("alert_max_parlays") or 2), 2)
    out["max_parlays_show"] = min(int(out.get("max_parlays_show") or 2), 2)
    out["parlay_min_leg_prob"] = float(rules["parlay_min_leg_prob"])
    out["max_parlay_share"] = float(rules.get("max_parlay_share") or (0.30 if live else 0.40))
    out["stake_power"] = float(rules.get("stake_power") or (0.70 if live else 0.95))
    return out
