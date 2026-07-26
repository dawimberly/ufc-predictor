"""Ensemble disagreement / conformal-width betting gates.

When disagreement is high or the prediction interval is wide:
  - ``skip`` — do not bet
  - ``tighten`` — raise min-edge and cut Kelly
  - ``allow`` — no change

Fail-closed: missing uncertainty metrics → treat as high uncertainty (skip).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

import config

logger = logging.getLogger(__name__)

GateAction = Literal["allow", "tighten", "skip"]

SKIP_HIGH_DISAGREEMENT = "high_disagreement"
SKIP_WIDE_INTERVAL = "wide_interval"
SKIP_MISSING = "missing_uncertainty"
TIGHTEN_DISAGREEMENT = "elevated_disagreement"
TIGHTEN_INTERVAL = "elevated_interval_width"


@dataclass
class UncertaintyGateResult:
    action: GateAction = "allow"
    reasons: list[str] = field(default_factory=list)
    disagreement: float | None = None
    interval_width: float | None = None
    edge_bump: float = 0.0
    kelly_mult: float = 1.0
    primary_reason: str = ""

    @property
    def skip(self) -> bool:
        return self.action == "skip"

    @property
    def tighten(self) -> bool:
        return self.action == "tighten"

    def reason_label(self) -> str:
        if self.primary_reason:
            return self.primary_reason
        return ",".join(self.reasons) if self.reasons else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reasons": list(self.reasons),
            "primary_reason": self.reason_label(),
            "disagreement": self.disagreement,
            "interval_width": self.interval_width,
            "edge_bump": self.edge_bump,
            "kelly_mult": self.kelly_mult,
        }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def read_uncertainty_metrics(row: pd.Series | dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Return (ensemble_disagreement, interval_width); None means missing."""
    if row is None:
        return None, None
    if isinstance(row, dict):
        row = pd.Series(row)
    disagree = _safe_float(row.get("ensemble_disagreement"))
    width = _safe_float(row.get("interval_width"))
    if width is None:
        lo = _safe_float(row.get("prob_ci_low"))
        hi = _safe_float(row.get("prob_ci_high"))
        if lo is not None and hi is not None and hi >= lo:
            width = hi - lo
    return disagree, width


def evaluate_uncertainty_gate(
    row: pd.Series | dict[str, Any] | None = None,
    *,
    disagreement: float | None = None,
    interval_width: float | None = None,
    settings: dict[str, Any] | None = None,
) -> UncertaintyGateResult:
    """
    Decide allow / tighten / skip from disagreement + conformal width.

    Fail-closed: either metric missing → skip with ``missing_uncertainty``.
    """
    try:
        cfg = settings or config.uncertainty_gate_settings()
    except Exception:
        # Absolute fail-closed
        return UncertaintyGateResult(
            action="skip",
            reasons=[SKIP_MISSING],
            primary_reason=SKIP_MISSING,
            edge_bump=0.05,
            kelly_mult=0.0,
        )

    if not cfg.get("enabled", True):
        d, w = (
            (disagreement, interval_width)
            if disagreement is not None or interval_width is not None
            else read_uncertainty_metrics(row)
        )
        return UncertaintyGateResult(
            action="allow",
            disagreement=d,
            interval_width=w,
            kelly_mult=1.0,
        )

    if disagreement is None and interval_width is None:
        disagreement, interval_width = read_uncertainty_metrics(row)
    else:
        disagreement = _safe_float(disagreement)
        interval_width = _safe_float(interval_width)

    reasons: list[str] = []
    action: GateAction = "allow"
    edge_bump = 0.0
    kelly_mult = 1.0

    # Fail-closed on missing metrics
    if disagreement is None or interval_width is None:
        return UncertaintyGateResult(
            action="skip",
            reasons=[SKIP_MISSING],
            primary_reason=SKIP_MISSING,
            disagreement=disagreement,
            interval_width=interval_width,
            edge_bump=float(cfg.get("edge_bump") or 0.0),
            kelly_mult=0.0,
        )

    d_skip = float(cfg["disagreement_skip"])
    d_tight = float(cfg["disagreement_tighten"])
    w_skip = float(cfg["interval_width_skip"])
    w_tight = float(cfg["interval_width_tighten"])
    bump = float(cfg.get("edge_bump") or 0.0)
    k_mult = float(cfg.get("kelly_mult") or 1.0)
    # Ensure tighten ≤ skip
    if d_tight > d_skip:
        d_tight = d_skip
    if w_tight > w_skip:
        w_tight = w_skip

    if disagreement >= d_skip:
        action = "skip"
        reasons.append(SKIP_HIGH_DISAGREEMENT)
    elif disagreement >= d_tight:
        action = "tighten"
        reasons.append(TIGHTEN_DISAGREEMENT)

    if interval_width >= w_skip:
        action = "skip"
        reasons.append(SKIP_WIDE_INTERVAL)
    elif interval_width >= w_tight:
        if action != "skip":
            action = "tighten"
        reasons.append(TIGHTEN_INTERVAL)

    if action == "skip":
        edge_bump = bump
        kelly_mult = 0.0
        # Prefer canonical skip reason names for logging
        primary = SKIP_HIGH_DISAGREEMENT if SKIP_HIGH_DISAGREEMENT in reasons else (
            SKIP_WIDE_INTERVAL if SKIP_WIDE_INTERVAL in reasons else reasons[0]
        )
    elif action == "tighten":
        edge_bump = bump
        kelly_mult = max(0.0, min(1.0, k_mult))
        primary = reasons[0] if reasons else TIGHTEN_DISAGREEMENT
    else:
        primary = ""

    return UncertaintyGateResult(
        action=action,
        reasons=reasons,
        primary_reason=primary,
        disagreement=disagreement,
        interval_width=interval_width,
        edge_bump=edge_bump,
        kelly_mult=kelly_mult,
    )


def effective_min_edge(base_min_edge: float, gate: UncertaintyGateResult) -> float:
    if gate.action == "allow":
        return float(base_min_edge)
    return float(base_min_edge) + float(gate.edge_bump or 0.0)


def apply_uncertainty_kelly_mult(
    kelly_or_stake: float,
    gate: UncertaintyGateResult,
) -> float:
    if gate.action == "skip":
        return 0.0
    if gate.action == "tighten":
        return float(kelly_or_stake) * float(gate.kelly_mult)
    return float(kelly_or_stake)


def log_uncertainty_skip(
    row: pd.Series | dict[str, Any] | None,
    gate: UncertaintyGateResult,
    *,
    event: str = "",
    context: str = "signal",
) -> None:
    """Append skip to bet journal + scorecard (canonical reason codes)."""
    if gate.action != "skip":
        return
    if isinstance(row, dict):
        data = row
    elif row is None:
        data = {}
    else:
        data = row  # Series supports .get
    f1 = str(data.get("fighter_1") or data.get("fighter1") or "")
    f2 = str(data.get("fighter_2") or data.get("fighter2") or "")
    pick = str(data.get("predicted_winner") or data.get("pick") or "")
    fight = f"{f1} vs {f2}" if f1 or f2 else ""
    reason = gate.reason_label() or SKIP_MISSING
    logger.info("Uncertainty gate SKIP %s (%s)", fight or pick or "?", reason)
    try:
        from src.skip_scorecard import log_skip

        ev = event
        if not ev:
            ev = str(data.get("event_name") or data.get("event") or "")
        edge = data.get("best_edge", data.get("edge"))
        edge_pct = ""
        try:
            if edge is not None and str(edge).strip() != "" and not (
                isinstance(edge, float) and pd.isna(edge)
            ):
                edge_pct = f"{float(edge) * 100:+.1f}"
        except (TypeError, ValueError):
            edge_pct = ""
        log_skip(
            reason,
            event=ev,
            fight=fight,
            pick=pick,
            edge_pct=edge_pct,
            context=context,
            disagreement=gate.disagreement,
            interval_width=gate.interval_width,
        )
    except Exception as exc:
        logger.debug("uncertainty skip scorecard failed: %s", exc)


def gate_summary_for_prompt(gate: UncertaintyGateResult) -> str:
    """Compact string for Grok/Ollama prompts."""
    d = f"{gate.disagreement:.3f}" if gate.disagreement is not None else "missing"
    w = f"{gate.interval_width:.3f}" if gate.interval_width is not None else "missing"
    if gate.action == "allow":
        return f"gate=allow disagree={d} width={w}"
    return (
        f"gate={gate.action} reason={gate.reason_label()} "
        f"disagree={d} width={w} edge_bump={gate.edge_bump:.3f} "
        f"kelly_mult={gate.kelly_mult:.2f}"
    )
