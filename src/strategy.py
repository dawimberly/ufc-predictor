"""Betting strategy: fractional Kelly, card risk caps, same-card parlays."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ufc_betting_bot.modules.edge import fight_decimal_odds, market_probs, raw_kelly_fraction

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Conservative defaults: high-accuracy / low-volume (Paper vs Live via profile)."""

    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02
    min_bet_fraction: float = 0.005
    max_card_risk_fraction: float = 0.08
    min_edge: float = 0.06
    min_model_prob: float = 0.70
    min_confidence: str = "medium"  # low | medium | high
    max_bets_per_card: int = 3
    flat_stake: float = 10.0
    parlay_min_edge: float = 0.055
    parlay_min_combined_prob: float = 0.40
    parlay_max_legs: int = 2
    parlay_min_leg_prob: float = 0.68
    unrealistic_roi_threshold_pct: float = 500.0


@dataclass
class BetCandidate:
    fight_id: str
    event_key: str
    bet_side: str
    prob: float
    decimal_odds: float
    edge: float
    kelly_full: float
    expected_value: float
    fighter1_name: str = ""
    fighter2_name: str = ""
    pick_name: str = ""
    winner_name: str = ""
    market_type: str = "moneyline"
    prop_key: str = ""
    display_label: str = ""
    odds_source: str = "synthetic"


@dataclass
class ParlayCandidate:
    legs: list[BetCandidate]
    combined_prob: float
    combined_odds: float
    expected_value: float
    min_leg_edge: float


def kelly_stake(
    bankroll: float,
    *,
    prob: float,
    decimal_odds: float,
    edge: float,
    config: StrategyConfig,
    rating_mult: float | None = None,
    row: pd.Series | dict[str, Any] | None = None,
    prop_type: str | None = None,
    market_type: str = "moneyline",
    uncertainty_kelly_mult: float | None = None,
) -> float:
    """Fractional Kelly stake with per-bet cap, segment rating, and uncertainty gates."""
    if edge < config.min_edge or bankroll <= 0:
        return 0.0

    unc_mult = 1.0
    if uncertainty_kelly_mult is not None:
        unc_mult = float(uncertainty_kelly_mult)
    elif row is not None:
        try:
            from src.uncertainty_gates import evaluate_uncertainty_gate

            gate = evaluate_uncertainty_gate(row)
            if gate.skip:
                return 0.0
            # Tighten also requires edge vs bumped min — caller usually checks;
            # still cut Kelly here.
            unc_mult = float(gate.kelly_mult)
            if gate.tighten and edge < config.min_edge + gate.edge_bump:
                return 0.0
        except Exception:
            # Fail-closed: missing gate machinery → no stake
            return 0.0

    if unc_mult <= 0:
        return 0.0

    kelly_frac = config.kelly_fraction
    try:
        from src.strategy_rating import apply_rating_to_kelly_fraction

        kelly_frac = apply_rating_to_kelly_fraction(
            kelly_frac,
            rating_mult=rating_mult,
            row=row,
            decimal_odds=decimal_odds,
            prop_type=prop_type,
            market_type=market_type,
        )
    except Exception:
        pass
    kelly = raw_kelly_fraction(prob, decimal_odds) * kelly_frac * unc_mult
    kelly = min(kelly, config.max_bet_fraction)
    if kelly < config.min_bet_fraction:
        return 0.0
    return float(min(bankroll * kelly, bankroll * config.max_bet_fraction))


def effective_card_risk_cap(
    config: StrategyConfig,
    mc_card_risk: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Resolve per-card risk cap, optionally adjusted by Monte Carlo card assessment."""
    if not mc_card_risk:
        return config.max_card_risk_fraction, []
    try:
        from src.risk_manager import recommended_card_risk_fraction

        return recommended_card_risk_fraction(mc_card_risk, config.max_card_risk_fraction)
    except ImportError:
        return config.max_card_risk_fraction, []


def apply_card_risk_cap(
    stakes: list[float],
    bankroll: float,
    *,
    max_card_fraction: float,
    mc_card_risk: dict[str, Any] | None = None,
) -> tuple[list[float], float, list[str]]:
    """
    Scale down stakes so total card exposure <= max_card_fraction * bankroll.

    When ``mc_card_risk`` is provided, may lower the cap via Monte Carlo guidance.
    Returns (capped_stakes, effective_cap_fraction, warnings).
    """
    warnings: list[str] = []
    cap_fraction = max_card_fraction
    if mc_card_risk:
        try:
            from src.risk_manager import recommended_card_risk_fraction

            cap_fraction, cap_warnings = recommended_card_risk_fraction(mc_card_risk, max_card_fraction)
            warnings.extend(cap_warnings)
        except ImportError:
            pass

    if not stakes or bankroll <= 0:
        return stakes, cap_fraction, warnings
    total = sum(stakes)
    cap = bankroll * cap_fraction
    if total <= cap or total <= 0:
        return stakes, cap_fraction, warnings
    scale = cap / total
    return [s * scale for s in stakes], cap_fraction, warnings


def bet_expected_value(prob: float, decimal_odds: float) -> float:
    """EV per $1 staked."""
    if decimal_odds <= 1 or not np.isfinite(prob):
        return 0.0
    return prob * (decimal_odds - 1.0) - (1.0 - prob)


def extract_bet_candidates(
    row: pd.Series,
    *,
    config: StrategyConfig,
    apply_uncertainty_gates: bool = True,
) -> BetCandidate | None:
    """Single-fight value bet candidate when odds and edge exist (uncertainty-gated)."""
    from src.high_accuracy_strategy import log_strategy_block

    min_edge = float(config.min_edge)
    unc_mult = 1.0
    f1 = str(row.get("fighter_1", row.get("fighter1_name", row.get("fighter1", "")))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2_name", row.get("fighter2", "")))).strip()
    fight_lbl = f"{f1} vs {f2}" if f1 or f2 else str(row.get("fight_id", ""))

    try:
        from src.fighter_flags import should_skip_fight

        skip_flag, flag_detail = should_skip_fight(f1, f2)
        if skip_flag:
            log_strategy_block(
                "fighter_integrity_flag",
                context="single",
                fight=fight_lbl,
                detail=flag_detail,
            )
            return None
    except Exception:
        pass

    try:
        from src.controversy import should_skip_for_referee

        ref = row.get("referee") or row.get("REFEREE")
        skip_ref, ref_detail = should_skip_for_referee(ref)
        if skip_ref:
            log_strategy_block(
                "controversial_referee_flag",
                context="single",
                fight=fight_lbl,
                detail=ref_detail,
            )
            return None
    except Exception:
        pass

    if apply_uncertainty_gates:
        try:
            from src.uncertainty_gates import evaluate_uncertainty_gate, effective_min_edge

            gate = evaluate_uncertainty_gate(row)
            if gate.skip:
                log_strategy_block(
                    gate.reason_label() or "uncertainty_skip",
                    context="single",
                    fight=fight_lbl,
                )
                return None
            min_edge = effective_min_edge(min_edge, gate)
            unc_mult = float(gate.kelly_mult)
        except Exception:
            log_strategy_block("uncertainty_fail_closed", context="single", fight=fight_lbl)
            return None  # fail-closed

    market = market_probs(row)
    decimal = fight_decimal_odds(row)
    if market is None or decimal is None:
        return None

    m1, m2 = market
    p1 = float(row.get("prob_f1_win", 0.5))
    p2 = float(row.get("prob_f2_win", 1.0 - p1))
    edge_f1 = p1 - m1
    edge_f2 = p2 - m2

    if edge_f1 >= edge_f2 and edge_f1 >= min_edge:
        side, prob, odds, edge = "f1", p1, decimal[0], edge_f1
    elif edge_f2 > edge_f1 and edge_f2 >= min_edge:
        side, prob, odds, edge = "f2", p2, decimal[1], edge_f2
    else:
        return None

    if float(prob) < float(getattr(config, "min_model_prob", 0.0) or 0.0):
        log_strategy_block(
            "low_model_prob",
            context="single",
            fight=fight_lbl,
            detail=f"prob={float(prob):.3f}<{float(config.min_model_prob):.3f}",
        )
        return None
    conf_label = str(row.get("confidence_label") or "").strip().lower()
    if not confidence_meets_minimum(conf_label or "low", getattr(config, "min_confidence", "low")):
        log_strategy_block(
            "low_confidence",
            context="single",
            fight=fight_lbl,
            detail=f"have={conf_label or 'low'} need={config.min_confidence}",
        )
        return None

    pick_name = f1 if side == "f1" else f2

    return BetCandidate(
        fight_id=str(row.get("fight_id", "")),
        event_key=str(row.get("event_name", row.get("event", ""))),
        bet_side=side,
        prob=prob,
        decimal_odds=odds,
        edge=edge,
        kelly_full=raw_kelly_fraction(prob, odds) * unc_mult,
        expected_value=bet_expected_value(prob, odds),
        fighter1_name=f1,
        fighter2_name=f2,
        pick_name=pick_name,
        winner_name=pick_name,
    )


def build_parlay_candidates(
    card_rows: pd.DataFrame,
    *,
    config: StrategyConfig,
) -> list[ParlayCandidate]:
    """
    High-accuracy parlays: exactly 2 legs, both strong (high prob + real edge).

    Legs are uncertainty-gated via ``extract_bet_candidates``.
    """
    from src.high_accuracy_strategy import PARLAY_MAX_LEGS, log_strategy_block

    max_legs = min(int(getattr(config, "parlay_max_legs", PARLAY_MAX_LEGS) or PARLAY_MAX_LEGS), PARLAY_MAX_LEGS)
    if max_legs < 2:
        return []

    leg_prob_floor = float(getattr(config, "parlay_min_leg_prob", 0.0) or 0.0)
    if leg_prob_floor <= 0:
        try:
            from src.high_accuracy_strategy import profile_rules

            leg_prob_floor = float(profile_rules().get("parlay_min_leg_prob") or config.min_model_prob)
        except Exception:
            leg_prob_floor = float(config.min_model_prob)

    legs: list[BetCandidate] = []
    for _, row in card_rows.iterrows():
        cand = extract_bet_candidates(row, config=config, apply_uncertainty_gates=True)
        if cand is None:
            continue
        if cand.edge < config.parlay_min_edge:
            log_strategy_block(
                "parlay_leg_weak_edge",
                context="parlay",
                fight=f"{cand.fighter1_name} vs {cand.fighter2_name}",
                detail=f"edge={cand.edge:.3f}<{config.parlay_min_edge:.3f}",
            )
            continue
        if cand.prob < leg_prob_floor:
            log_strategy_block(
                "parlay_leg_weak_prob",
                context="parlay",
                fight=f"{cand.fighter1_name} vs {cand.fighter2_name}",
                detail=f"prob={cand.prob:.3f}<{leg_prob_floor:.3f}",
            )
            continue
        # Path-risk: parlays only when both legs are high-confidence
        try:
            from src.high_accuracy_strategy import parlay_min_leg_confidence

            need_conf = parlay_min_leg_confidence()
        except Exception:
            need_conf = "high"
        conf_label = str(row.get("confidence_label") or row.get("confidence") or "").strip().lower() or "low"
        if not confidence_meets_minimum(conf_label, need_conf):
            log_strategy_block(
                "parlay_leg_low_confidence",
                context="parlay",
                fight=f"{cand.fighter1_name} vs {cand.fighter2_name}",
                detail=f"have={conf_label} need={need_conf}",
            )
            continue
        legs.append(cand)

    if len(legs) < 2:
        return []

    parlays: list[ParlayCandidate] = []
    for combo in combinations(legs, 2):
        combined_prob = float(np.prod([c.prob for c in combo]))
        if combined_prob < config.parlay_min_combined_prob:
            log_strategy_block(
                "parlay_combined_prob",
                context="parlay",
                detail=f"combined={combined_prob:.3f}<{config.parlay_min_combined_prob:.3f}",
            )
            continue
        combined_odds = float(np.prod([c.decimal_odds for c in combo]))
        ev = combined_prob * (combined_odds - 1.0) - (1.0 - combined_prob)
        parlays.append(
            ParlayCandidate(
                legs=list(combo),
                combined_prob=combined_prob,
                combined_odds=combined_odds,
                expected_value=ev,
                min_leg_edge=min(c.edge for c in combo),
            )
        )

    parlays.sort(key=lambda p: p.expected_value, reverse=True)
    return parlays


def strategy_from_profile(
    *,
    min_edge: float | None = None,
    bankroll: float | None = None,
    recent_win_rate: float | None = None,
    model_confidence: float | None = None,
    hours_to_event: float | None = None,
    use_dynamic_thresholds: bool | None = None,
) -> StrategyConfig:
    """Build StrategyConfig from active UFC_PROFILE thresholds (optionally dynamic)."""
    import config as _cfg

    enabled = (
        _cfg.DYNAMIC_THRESHOLDS_ENABLED if use_dynamic_thresholds is None else use_dynamic_thresholds
    )
    if enabled and bankroll is not None:
        from ufc_betting_bot.modules.dynamic_thresholds import get_profile_thresholds

        health = None
        if getattr(_cfg, "HEALTH_FEEDBACK_ENABLED", True):
            try:
                from src.strategy_performance import segment_health

                health = segment_health(profile=_cfg.UFC_PROFILE)
            except Exception:
                health = {"complete": False, "fail_closed": True, "trade_count": 0}
        thresholds = get_profile_thresholds(
            bankroll,
            recent_win_rate,
            model_confidence,
            hours_to_event=hours_to_event,
            profile=_cfg.UFC_PROFILE,
            segment_health=health,
        )
        edge = thresholds.alert_min_edge if min_edge is None else min_edge
        ps_dyn = _cfg.profile_settings()
        return StrategyConfig(
            kelly_fraction=_cfg.profile_value("kelly_fraction"),
            max_bet_fraction=_cfg.profile_value("max_bet_fraction"),
            max_card_risk_fraction=_cfg.effective_max_card_risk_fraction(bankroll),
            min_edge=edge,
            min_model_prob=float(_cfg.profile_value("singles_min_model_prob") or 0.70),
            min_confidence=str(_cfg.profile_value("singles_min_confidence") or "medium"),
            max_bets_per_card=int(_cfg.profile_value("max_bets_per_card") or 3),
            parlay_min_edge=max(float(thresholds.parlay_min_edge), float(ps_dyn.get("parlay_min_edge") or 0)),
            parlay_min_combined_prob=max(
                float(thresholds.parlay_min_combined_prob),
                float(ps_dyn.get("parlay_min_combined_prob") or 0),
            ),
            parlay_max_legs=2,
            parlay_min_leg_prob=float(ps_dyn.get("parlay_min_leg_prob") or 0.68),
            flat_stake=_cfg.FLAT_STAKE,
        )

    ps = _cfg.profile_settings()
    card_frac = _cfg.effective_max_card_risk_fraction(bankroll)
    edge = ps["alert_min_edge"] if min_edge is None else min_edge
    return StrategyConfig(
        kelly_fraction=ps["kelly_fraction"],
        max_bet_fraction=ps["max_bet_fraction"],
        max_card_risk_fraction=card_frac,
        min_edge=edge,
        min_model_prob=float(ps.get("singles_min_model_prob") or 0.70),
        min_confidence=str(ps.get("singles_min_confidence") or "medium"),
        max_bets_per_card=int(ps.get("max_bets_per_card") or 3),
        parlay_min_edge=ps["parlay_min_edge"],
        parlay_min_combined_prob=ps["parlay_min_combined_prob"],
        parlay_max_legs=2,
        parlay_min_leg_prob=float(ps.get("parlay_min_leg_prob") or 0.68),
        flat_stake=_cfg.FLAT_STAKE,
    )


def _pick_model_prob(row: pd.Series) -> tuple[str, float, str]:
    """Return (pick, model_prob, fight_label) for a prediction row."""
    f1 = str(row.get("fighter_1", row.get("fighter1", "")))
    f2 = str(row.get("fighter_2", row.get("fighter2", "")))
    pick = str(row.get("predicted_winner", ""))
    prob = row.get("predicted_prob", row.get("prob_f1_win"))
    if pd.isna(prob):
        if pick == f2 and pd.notna(row.get("prob_f2_win")):
            prob = float(row["prob_f2_win"])
        elif pick == f1 and pd.notna(row.get("prob_f1_win")):
            prob = float(row["prob_f1_win"])
        else:
            p1 = float(row.get("prob_f1_win", 0.5))
            prob = p1 if pick == f1 else 1.0 - p1
    return pick, float(prob), f"{f1} vs {f2}"


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "": 0, "unknown": 0}


def confidence_meets_minimum(label: str | None, minimum: str | None) -> bool:
    """True when confidence label is at least the profile minimum."""
    need = str(minimum or "low").strip().lower()
    have = str(label or "low").strip().lower()
    return _CONFIDENCE_RANK.get(have, 0) >= _CONFIDENCE_RANK.get(need, 0)


def single_quality_score(single: dict[str, Any]) -> float:
    """
    Rank singles for the per-card cap: high model prob + clear edge + low uncertainty.
    """
    try:
        prob = float(single.get("prob") or 0.5)
    except (TypeError, ValueError):
        prob = 0.5
    try:
        edge = float(single.get("edge") or 0.0)
    except (TypeError, ValueError):
        edge = 0.0
    unc = str(single.get("uncertainty_action") or "allow").strip().lower()
    unc_bonus = 1.0 if unc == "allow" else (0.65 if unc == "tighten" else 0.0)
    width = single.get("interval_width")
    disagree = single.get("ensemble_disagreement")
    unc_pen = 0.0
    try:
        if width is not None and str(width).strip() != "":
            unc_pen += min(1.0, float(width) / 0.60) * 0.55
    except (TypeError, ValueError):
        pass
    try:
        if disagree is not None and str(disagree).strip() != "":
            unc_pen += min(1.0, float(disagree) / 0.12) * 0.45
    except (TypeError, ValueError):
        pass
    edge_term = min(max(edge, 0.0), 0.20) / 0.20
    return 0.50 * prob + 0.30 * edge_term + 0.20 * unc_bonus - 0.12 * unc_pen


def apply_max_bets_per_card(
    singles: list[dict[str, Any]],
    *,
    max_bets: int,
    event_key: str = "event_name",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Keep top ``max_bets`` singles per card by quality score.

    Ticket budget is hard-clamped to 1–4 (high-accuracy strategy).
    Returns (kept, overflow_skipped).
    """
    try:
        from src.high_accuracy_strategy import clamp_max_tickets

        cap = clamp_max_tickets(max_bets)
    except Exception:
        cap = max(1, min(4, int(max_bets or 3)))
    if len(singles) <= cap and len({str(s.get(event_key) or s.get("event") or "") for s in singles}) <= 1:
        # Fast path when already within cap for a single card
        ranked = sorted(singles, key=single_quality_score, reverse=True)
        if len(ranked) <= cap:
            for i, s in enumerate(ranked, start=1):
                s["quality_score"] = round(single_quality_score(s), 4)
                s["card_rank"] = i
            return ranked, []

    by_event: dict[str, list[dict[str, Any]]] = {}
    for s in singles:
        ev = str(s.get(event_key) or s.get("event") or s.get("event_key") or "card")
        by_event.setdefault(ev, []).append(s)

    kept: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for ev, group in by_event.items():
        ranked = sorted(group, key=single_quality_score, reverse=True)
        for i, s in enumerate(ranked, start=1):
            s["quality_score"] = round(single_quality_score(s), 4)
            s["card_rank"] = i
            if i <= cap:
                kept.append(s)
            else:
                overflow.append(s)
    kept.sort(key=single_quality_score, reverse=True)
    return kept, overflow


def apply_max_tickets_per_card(
    singles: list[dict[str, Any]],
    parlays: list[dict[str, Any]],
    *,
    max_tickets: int,
    event_key: str = "event_name",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Cap total tickets (singles + parlays) per card. Prefer singles, then best parlays.

    Returns (kept_singles, kept_parlays, overflow_singles, overflow_parlays).
    """
    try:
        from src.high_accuracy_strategy import clamp_max_tickets, log_strategy_block

        cap = clamp_max_tickets(max_tickets)
    except Exception:
        cap = max(1, min(4, int(max_tickets or 3)))

        def log_strategy_block(*_a, **_k):  # type: ignore
            return None

    # Rank singles globally by quality; assign per event slots preferring singles
    by_event: dict[str, dict[str, list]] = {}
    for s in singles:
        ev = str(s.get(event_key) or s.get("event") or s.get("event_key") or "card")
        by_event.setdefault(ev, {"singles": [], "parlays": []})["singles"].append(s)
    for p in parlays:
        legs = p.get("legs") or []
        ev = "card"
        if legs and isinstance(legs[0], dict):
            # Best-effort: use first leg fight's event if present
            ev = str(p.get(event_key) or p.get("event") or p.get("event_name") or "card")
        by_event.setdefault(ev, {"singles": [], "parlays": []})["parlays"].append(p)

    # If parlays lack event, pool under single key with all singles of one card
    if len(by_event) == 1 or (parlays and all(not (p.get(event_key) or p.get("event") or p.get("event_name")) for p in parlays)):
        # Merge everything under one budget when event keys are missing on parlays
        all_s = sorted(singles, key=single_quality_score, reverse=True)
        all_p = sorted(parlays, key=lambda x: float(x.get("expected_value") or 0), reverse=True)
        kept_s, kept_p, ov_s, ov_p = [], [], [], []
        slots = cap
        for s in all_s:
            if slots > 0:
                kept_s.append(s)
                slots -= 1
            else:
                ov_s.append(s)
                log_strategy_block("ticket_cap", context="single", detail=f"max={cap}")
        for p in all_p:
            if slots > 0:
                kept_p.append(p)
                slots -= 1
            else:
                ov_p.append(p)
                log_strategy_block("ticket_cap", context="parlay", detail=f"max={cap}")
        return kept_s, kept_p, ov_s, ov_p

    kept_s, kept_p, ov_s, ov_p = [], [], [], []
    for ev, bucket in by_event.items():
        s_ranked = sorted(bucket["singles"], key=single_quality_score, reverse=True)
        p_ranked = sorted(bucket["parlays"], key=lambda x: float(x.get("expected_value") or 0), reverse=True)
        slots = cap
        for s in s_ranked:
            if slots > 0:
                kept_s.append(s)
                slots -= 1
            else:
                ov_s.append(s)
                log_strategy_block("ticket_cap", context="single", detail=f"event={ev} max={cap}")
        for p in p_ranked:
            if slots > 0:
                kept_p.append(p)
                slots -= 1
            else:
                ov_p.append(p)
                log_strategy_block("ticket_cap", context="parlay", detail=f"event={ev} max={cap}")
    return kept_s, kept_p, ov_s, ov_p


def build_model_only_parlay_candidates(
    card_rows: pd.DataFrame,
    *,
    min_pick_prob: float = 0.52,
    parlay_max_legs: int = 3,
    parlay_min_combined_prob: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Research helper: rank same-card parlays by model probability only (no odds).

    Uses highest-confidence picks per fight; does not imply +EV without market lines.
    """
    legs: list[dict[str, Any]] = []
    for _, row in card_rows.iterrows():
        pick, prob, fight = _pick_model_prob(row)
        if prob < min_pick_prob:
            continue
        legs.append(
            {
                "fight": fight,
                "pick": pick,
                "prob": prob,
                "fight_id": str(row.get("fight_id", fight)),
            }
        )
    if len(legs) < 2:
        return []

    out: list[dict[str, Any]] = []
    max_legs = min(parlay_max_legs, len(legs))
    for n in range(2, max_legs + 1):
        for combo in combinations(legs, n):
            combined_prob = float(np.prod([leg["prob"] for leg in combo]))
            if combined_prob < parlay_min_combined_prob:
                continue
            picks_txt = " + ".join(f"{leg['pick']} ({leg['prob']:.0%})" for leg in combo)
            out.append(
                {
                    "n_legs": n,
                    "legs": list(combo),
                    "combined_prob": combined_prob,
                    "picks": picks_txt,
                    "model_only": True,
                }
            )
    out.sort(key=lambda x: x["combined_prob"], reverse=True)
    return out[:5]


def build_auto_parlay_recommendations(
    card_rows: pd.DataFrame | None,
    *,
    ha_singles: list[dict[str, Any]] | None = None,
    min_pick_prob: float = 0.58,
    min_combined_prob_2: float = 0.30,
    min_combined_prob_3: float = 0.18,
    leg_counts: tuple[int, ...] = (2, 3),
) -> list[dict[str, Any]]:
    """
    Automatically pick one best 2-leg and one best 3-leg research parlay.

    Prefer HA-cleared singles as legs when available; otherwise highest model probs.
    Advisory only — not Live HA-sized tickets (especially 3-leg).
    """
    if card_rows is None or getattr(card_rows, "empty", True):
        return []

    # Preferred fight ids from HA singles (moneyline)
    preferred: set[str] = set()
    for s in ha_singles or []:
        if s.get("is_parlay") or int(s.get("n_legs") or 1) >= 2:
            continue
        stake = float(s.get("suggested_stake") or s.get("stake_usd") or 0)
        stake_pct = float(s.get("stake_pct") or 0)
        if stake <= 0 and stake_pct <= 0:
            continue
        for k in (str(s.get("fight_id") or "").strip(), str(s.get("fight") or "").strip()):
            if k:
                preferred.add(k)

    legs: list[dict[str, Any]] = []
    for _, row in card_rows.iterrows():
        pick, prob, fight = _pick_model_prob(row)
        if not pick or float(prob) < float(min_pick_prob):
            continue
        fid = str(row.get("fight_id") or fight).strip()
        conf = str(row.get("confidence_label") or row.get("confidence") or "").strip().lower()
        from src.props import event_from_record

        legs.append(
            {
                "fight": fight,
                "pick": str(pick),
                "prob": float(prob),
                "fight_id": fid,
                "confidence": conf or "-",
                "ha_leg": bool(fid in preferred or fight in preferred),
                "weight_class": str(row.get("weight_class") or ""),
                "event": event_from_record(row),
                "event_name": event_from_record(row),
            }
        )

    # Prefer HA legs first, then by prob
    legs.sort(key=lambda L: (0 if L["ha_leg"] else 1, -L["prob"]))
    # Cap candidate pool so combinations stay small
    pool = legs[:10]
    if len(pool) < 2:
        return []

    def _best_for_n(n: int, min_comb: float) -> dict[str, Any] | None:
        if len(pool) < n:
            return None
        best: dict[str, Any] | None = None
        best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
        for combo in combinations(pool, n):
            # One leg per fight
            fids = [c["fight_id"] for c in combo]
            if len(set(fids)) != n:
                continue
            combined = float(np.prod([c["prob"] for c in combo]))
            if combined < min_comb:
                continue
            ha_count = sum(1 for c in combo if c["ha_leg"])
            # Rank: more HA legs, then combined prob, then min leg prob
            key = (float(ha_count), combined, min(c["prob"] for c in combo))
            if key > best_key:
                best_key = key
                picks_txt = " + ".join(f"{c['pick']} ({c['prob']:.0%})" for c in combo)
                best = {
                    "id": f"parlay-{n}leg-" + "-".join(sorted(fids))[:80],
                    "n_legs": n,
                    "legs": [
                        {
                            "fight_id": c["fight_id"],
                            "fight": c["fight"],
                            "pick": c["pick"],
                            "prob": c["prob"],
                            "confidence": c["confidence"],
                            "ha_leg": c["ha_leg"],
                            "event": c.get("event") or "",
                            "event_name": c.get("event_name") or c.get("event") or "",
                        }
                        for c in combo
                    ],
                    "combined_prob": combined,
                    "picks": picks_txt,
                    "pick_line": picks_txt,
                    "display_label": picks_txt,
                    "market": f"{n}-leg parlay",
                    "market_type": "parlay",
                    "is_parlay": True,
                    "model_only": True,
                    "advisory": True,
                    "fun_bet": False,
                    "suggested_stake": 0.0,
                    "stake_usd": 0.0,
                    "stake_pct": 0.0,
                    "ha_legs": ha_count,
                    "ha_qualified": n == 2 and ha_count == n,
                    "event": next((str(c.get("event") or "") for c in combo if c.get("event")), ""),
                    "event_name": next((str(c.get("event") or "") for c in combo if c.get("event")), ""),
                    "reason": "",
                    "brief": (
                        f"Auto {n}-leg · combined {combined:.0%}"
                        + (" · HA legs" if ha_count == n else " · research")
                    ),
                }
        return best

    floors = {2: float(min_combined_prob_2), 3: float(min_combined_prob_3)}
    out: list[dict[str, Any]] = []
    for n in leg_counts:
        if int(n) < 2:
            continue
        rec = _best_for_n(int(n), floors.get(int(n), 0.15))
        if rec is not None:
            out.append(rec)
    return out


def compute_equity_metrics(equity: pd.Series) -> dict[str, float]:
    """Max drawdown % and longest win streak from chronological equity curve."""
    if equity.empty:
        return {"max_drawdown_pct": 0.0, "max_win_streak": 0.0}

    eq = equity.astype(float).reset_index(drop=True)
    peak = eq.cummax()
    drawdown = np.where(peak > 0, (peak - eq) / peak, 0.0)
    max_dd = float(np.max(drawdown) * 100.0) if len(drawdown) else 0.0

    return {"max_drawdown_pct": max_dd, "max_win_streak": 0.0}


def compute_trade_streaks(won: pd.Series) -> dict[str, float]:
    if won.empty:
        return {"max_win_streak": 0.0, "max_loss_streak": 0.0}
    best_win = best_loss = cur_win = cur_loss = 0
    for w in won.astype(int):
        if w:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        best_win = max(best_win, cur_win)
        best_loss = max(best_loss, cur_loss)
    return {"max_win_streak": float(best_win), "max_loss_streak": float(best_loss)}


def warn_unrealistic_roi(roi_pct: float, *, threshold: float = 500.0) -> str | None:
    if not np.isfinite(roi_pct) or roi_pct <= threshold:
        return None
    return (
        f"ROI {roi_pct:.1f}% exceeds {threshold:.0f}% — likely overfitting, odds leakage, "
        "or compounding artifacts. Treat as diagnostic only."
    )


def enrich_summary_with_risk_metrics(
    trades: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Add drawdown / streak stats to a backtest summary dict."""
    if trades.empty:
        summary.update(max_drawdown_pct=0.0, max_win_streak=0.0, max_loss_streak=0.0)
        return summary
    if "equity" in trades.columns:
        summary.update(compute_equity_metrics(trades["equity"]))
    if "won" in trades.columns:
        summary.update(compute_trade_streaks(trades["won"]))
    warning = warn_unrealistic_roi(float(summary.get("roi_pct", 0)))
    if warning:
        summary["roi_warning"] = warning
    return summary


# --- Budget manager (per-book card allocation) --------------------------------


def resolve_budget_for_calculations(
    budget_state: dict[str, Any] | None,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """
    Normalize budget and apply profile card cap for stake / allocation math.

    Live mode clamps card_budget to the live USD cap (default $12).
    """
    import config as _cfg

    state = _cfg.normalize_budget_state(budget_state)
    card_eff, _ = effective_card_budget_usd(state, profile=profile)
    resolved = dict(state)
    resolved["card_budget"] = card_eff
    return resolved


def effective_card_budget_usd(
    budget_state: dict[str, Any],
    *,
    profile: str | None = None,
) -> tuple[float, list[str]]:
    """
    Resolve user card budget capped by profile safe limits.

    Live mode hard-caps at LIVE_MAX_CARD_STAKE_USD (default $12).
    """
    import config as _cfg

    warnings: list[str] = []
    br = max(float(budget_state.get("total_bankroll") or _cfg.DEFAULT_TOTAL_BANKROLL), 1.0)
    raw = float(budget_state.get("card_budget") or 0.0)
    if raw <= 0:
        raw = _cfg.max_card_stake_cap(br)

    safe_cap = _cfg.max_card_stake_cap(br)
    capped = min(raw, safe_cap)

    live = _cfg.is_live_profile() if profile is None else _cfg.normalize_profile(profile) == "live"
    if live:
        live_usd = float(
            _cfg.profile_settings().get("max_card_stake_usd") or _cfg.LIVE_MAX_CARD_BUDGET_USD
        )
        if raw > live_usd:
            warnings.append(
                f"Card budget ${raw:,.2f} exceeds Live cap ${live_usd:,.2f}; "
                f"allocations use ${min(capped, live_usd):,.2f}."
            )
        capped = min(capped, live_usd)
    elif raw > safe_cap:
        warnings.append(
            f"Card budget ${raw:,.2f} exceeds profile safe cap ${safe_cap:,.2f}; "
            f"allocations use ${capped:,.2f}."
        )

    return max(capped, 0.0), warnings


def allocate_card_budget_per_book(
    budget_state: dict[str, Any],
    *,
    profile: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Split effective card budget across enabled books.

    Proportional to positive balances when any exist; otherwise equal split.
    Each book allocation is capped at that book's balance.
    """
    import config as _cfg

    card_budget, _ = effective_card_budget_usd(budget_state, profile=profile)
    enabled: list[tuple[str, float]] = []
    for book in _cfg.BUDGET_BOOKS:
        use_key = _cfg.BUDGET_USE_KEYS[book]
        bal_key = _cfg.BUDGET_BALANCE_KEYS[book]
        if not budget_state.get(use_key, True):
            continue
        enabled.append((book, max(float(budget_state.get(bal_key) or 0.0), 0.0)))

    result: dict[str, dict[str, Any]] = {}
    for book in _cfg.BUDGET_BOOKS:
        use_key = _cfg.BUDGET_USE_KEYS[book]
        bal_key = _cfg.BUDGET_BALANCE_KEYS[book]
        balance = max(float(budget_state.get(bal_key) or 0.0), 0.0)
        enabled_flag = bool(budget_state.get(use_key, True))
        result[book] = {
            "balance": balance,
            "enabled": enabled_flag,
            "allocation": 0.0,
            "share_pct": 0.0,
        }

    if not enabled or card_budget <= 0:
        return result

    with_balance = [(b, bal) for b, bal in enabled if bal > 0]
    if with_balance:
        total_bal = sum(bal for _, bal in with_balance)
        shares = {book: card_budget * (bal / total_bal) for book, bal in with_balance}
    else:
        share = card_budget / len(enabled)
        shares = {book: share for book, _ in enabled}

    for book, _ in enabled:
        raw_alloc = shares.get(book, 0.0)
        balance = result[book]["balance"]
        alloc = min(raw_alloc, balance) if balance > 0 else raw_alloc
        result[book]["allocation"] = float(alloc)
        result[book]["share_pct"] = (alloc / card_budget * 100.0) if card_budget > 0 else 0.0

    return result


def distribute_stakes_to_pool(singles: list[dict[str, Any]], pool: float) -> list[float]:
    """Scale suggested singles stakes to fit within a book's card allocation pool."""
    if not singles or pool <= 0:
        return [0.0] * len(singles)
    raw = [max(float(s.get("suggested_stake") or 0.0), 0.0) for s in singles]
    total = sum(raw)
    if total <= 0:
        even = pool / len(singles)
        return [even] * len(singles)
    if total <= pool:
        return raw
    scale = pool / total
    return [r * scale for r in raw]


def _ticket_confidence_score(ticket: dict[str, Any]) -> float:
    conf = str(
        ticket.get("confidence")
        or ticket.get("confidence_label")
        or ticket.get("min_confidence")
        or ""
    ).strip().lower()
    return {"high": 1.0, "medium": 0.78, "low": 0.5, "": 0.65}.get(conf, 0.65)


def _ticket_confidence_label(ticket: dict[str, Any]) -> str:
    conf = str(
        ticket.get("confidence")
        or ticket.get("confidence_label")
        or ticket.get("min_confidence")
        or ""
    ).strip().lower()
    if conf in {"high", "medium", "low"}:
        return conf
    return "medium" if _ticket_confidence_score(ticket) >= 0.7 else "low"


def _ticket_edge_fraction(ticket: dict[str, Any]) -> float:
    frac = ticket_max_edge_fraction(ticket)
    if frac is None:
        return 0.0
    return max(0.0, float(frac))


def _ticket_prob(ticket: dict[str, Any]) -> float:
    for key in ("prob", "combined_prob", "predicted_prob"):
        if ticket.get(key) is not None:
            try:
                return float(np.clip(float(ticket[key]), 0.01, 0.99))
            except (TypeError, ValueError):
                pass
    return 0.55


def _ticket_decimal_odds(ticket: dict[str, Any]) -> float | None:
    """Best available decimal price on a ticket (singles or parlays)."""
    for key in ("decimal_odds", "combined_odds", "odds", "opening_odds"):
        if ticket.get(key) is None:
            continue
        raw = ticket.get(key)
        odds_f = sanitize_decimal_odds(raw)
        if odds_f is not None and float(odds_f) > 1.01:
            return float(odds_f)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        # Allow parlay combined prices above the single-fight sanitize cap (15)
        if 1.01 < v <= 100.0:
            return v
    # Parlay legs may carry per-leg odds when top-level combined is missing
    legs = ticket.get("legs")
    if isinstance(legs, list) and legs:
        prod = 1.0
        ok = 0
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            lo = _ticket_decimal_odds({k: leg.get(k) for k in ("decimal_odds", "odds", "opening_odds")})
            if lo is None:
                return None
            prod *= lo
            ok += 1
        if ok >= 2 and prod > 1.01:
            return float(prod)
    return None


def _ticket_kelly_fraction(ticket: dict[str, Any]) -> float:
    """Best available Kelly-style fraction for weighting."""
    for key in ("kelly_pct", "kelly_fraction"):
        if ticket.get(key) is not None:
            try:
                v = float(ticket[key])
                if key == "kelly_pct" and v > 1.0:
                    v = v / 100.0
                return max(0.0, min(v, 0.5))
            except (TypeError, ValueError):
                pass
    if ticket.get("kelly_stake_usd") is not None and ticket.get("card_pool_usd"):
        try:
            pool = float(ticket["card_pool_usd"])
            if pool > 0:
                return max(0.0, min(float(ticket["kelly_stake_usd"]) / pool, 0.5))
        except (TypeError, ValueError):
            pass
    # Approximate from prob + decimal odds
    prob = _ticket_prob(ticket)
    odds_f = _ticket_decimal_odds(ticket) or 0.0
    if odds_f > 1.01:
        full = (prob * odds_f - 1.0) / (odds_f - 1.0)
        return max(0.0, min(full * 0.25, 0.5))
    # Fallback: edge-scaled
    return min(0.5, _ticket_edge_fraction(ticket) * 0.8)


def _ticket_edge_vs_market(ticket: dict[str, Any]) -> tuple[float, float | None]:
    """
    Edge vs market implied probability.

    Returns (edge_fraction, market_implied_or_None).
    Prefer explicit ticket edge; else model_prob - 1/odds.
    """
    odds = _ticket_decimal_odds(ticket)
    implied = (1.0 / odds) if odds and odds > 1.01 else None
    stored = _ticket_edge_fraction(ticket)
    if stored > 0:
        return stored, implied
    if implied is None:
        return 0.0, None
    return max(0.0, _ticket_prob(ticket) - float(implied)), implied


def _uncertainty_penalty(ticket: dict[str, Any], *, live: bool, tighten_mult: float) -> tuple[float, bool, str]:
    """
    Uncertainty shrink for stake strength.

    Fail-closed: skip / missing uncertainty → no inflate (penalty 0).
    Continuous shrink when disagreement / interval_width look elevated.
    """
    unc = str(ticket.get("uncertainty_action") or "allow").strip().lower()
    if unc in {"skip", "block", "missing", "missing_uncertainty"}:
        return 0.0, True, unc or "skip"

    disagree = ticket.get("ensemble_disagreement")
    width = ticket.get("interval_width")
    try:
        d = float(disagree) if disagree is not None and str(disagree) != "" else None
    except (TypeError, ValueError):
        d = None
    try:
        w = float(width) if width is not None and str(width) != "" else None
    except (TypeError, ValueError):
        w = None

    # Tickets that already cleared gates usually have metrics; if both missing,
    # do not inflate (conservative) — treat as no-inflate with tiny residual strength.
    if d is None and w is None and unc == "allow" and ticket.get("uncertainty_action") is None:
        # Historical tickets often omit metrics after gating; allow but don't boost.
        base = 0.85 if not live else 0.75
    elif unc == "tighten":
        base = float(tighten_mult)
    else:
        base = 1.0

    # Soft continuous penalties (do not invent skip here — gates already ran)
    if d is not None:
        if d >= 0.18:
            base *= 0.55 if not live else 0.40
        elif d >= 0.12:
            base *= 0.75 if not live else 0.60
    if w is not None:
        if w >= 0.45:
            base *= 0.55 if not live else 0.40
        elif w >= 0.30:
            base *= 0.78 if not live else 0.62

    no_inflate = base <= 0.05 or unc == "tighten" and live
    return max(0.0, min(1.0, base)), bool(no_inflate and base < 0.5), unc or "allow"


def compute_ticket_strength(ticket: dict[str, Any], *, live: bool) -> dict[str, Any]:
    """
    Per-ticket strength for confidence- and odds-aware compounding.

    Components:
      - model probability / confidence
      - edge vs market implied probability
      - uncertainty penalty (high disagreement / wide interval → shrink)
      - slight discount for 2-leg parlays vs singles

    Fail-closed: missing odds or skip-level uncertainty → strength 0 (no inflate).
    """
    try:
        from src.high_accuracy_strategy import sizing_curve_params

        curve = sizing_curve_params(live=live)
    except Exception:
        curve = {
            "max_ticket_pct": 0.32 if live else 0.48,
            "gamma": 1.45 if live else 1.20,
            "parlay_mult": 0.60 if live else 0.72,
            "prop_mult": 0.85 if live else 0.92,
            "tighten_mult": 0.40 if live else 0.55,
        }

    odds = _ticket_decimal_odds(ticket)
    conf_label = _ticket_confidence_label(ticket)
    conf_score = _ticket_confidence_score(ticket)
    prob = _ticket_prob(ticket)
    edge, implied = _ticket_edge_vs_market(ticket)
    if ticket_edge_exceeds_actionable_cap(ticket) or abs(float(edge or 0)) > MAX_ACTIONABLE_EDGE:
        return {
            "strength": 0.0,
            "strength_score": 0.0,
            "target_stake_pct": 0.0,
            "edge": float(edge or 0.0),
            "confidence": conf_label,
            "confidence_score": float(conf_score),
            "model_prob": float(prob),
            "decimal_odds": odds,
            "market_implied": implied,
            "prob_score": 0.0,
            "edge_score": 0.0,
            "odds_score": 0.0,
            "uncertainty_penalty": 0.0,
            "uncertainty_action": "skip",
            "type_mult": 1.0,
            "is_parlay": bool(ticket.get("is_parlay")),
            "no_inflate": True,
            "fail_closed_reason": "suspect_edge",
            "sizing_mode": "conf_odds",
            "curve_gamma": float(curve["gamma"]),
            "max_ticket_pct": float(curve["max_ticket_pct"]),
            "profile": "live" if live else "paper",
        }
    unc_pen, unc_no_inflate, unc_action = _uncertainty_penalty(
        ticket, live=live, tighten_mult=float(curve["tighten_mult"])
    )

    kind = str(ticket.get("bet_type") or ticket.get("market_type") or "").lower()
    is_parlay = bool(ticket.get("is_parlay")) or "parlay" in kind or int(ticket.get("n_legs") or 1) >= 2
    is_prop = (
        str(ticket.get("market_type") or "").lower() == "prop"
        or str(ticket.get("prop_key") or "") == "over_1_5_rounds"
        or "prop" in kind
    )
    type_mult = float(curve["parlay_mult"]) if is_parlay else (
        float(curve["prop_mult"]) if is_prop else 1.0
    )

    fail_closed_reason = ""
    no_inflate = False
    if odds is None:
        # Missing odds → never inflate stake
        strength = 0.0
        fail_closed_reason = "missing_odds"
        no_inflate = True
        prob_score = 0.0
        edge_score = 0.0
        odds_score = 0.0
    elif unc_pen <= 0.0:
        strength = 0.0
        fail_closed_reason = f"uncertainty:{unc_action}"
        no_inflate = True
        prob_score = float(np.clip((prob - 0.55) / 0.35, 0.0, 1.0))
        edge_score = float(np.clip(edge / 0.20, 0.0, 1.0))
        odds_score = 0.0
    else:
        # Model-prob score: HA singles live near ≥0.70; map 0.55→0 … 0.90→1
        prob_score = float(np.clip((prob - 0.55) / 0.35, 0.0, 1.0))
        # Edge vs market: 0→0, 20%+ → 1
        edge_score = float(np.clip(edge / 0.20, 0.0, 1.0))
        # Favorable odds: reward positive EV prices; do not boost longshots without edge
        # Soft peak around -150 to +150 when edge is present
        if implied is not None and edge > 0:
            # Prefer prices where model is clearly above market; scale by edge quality
            odds_score = float(np.clip(0.35 + 0.65 * edge_score, 0.0, 1.0))
            if odds >= 2.40 and edge < 0.12:
                odds_score *= 0.70  # speculative dog without strong edge
            if odds < 1.25 and edge < 0.08:
                odds_score *= 0.75  # chalk with thin edge
        else:
            odds_score = 0.25 * edge_score

        raw = (
            0.34 * prob_score
            + 0.22 * conf_score
            + 0.28 * edge_score
            + 0.16 * odds_score
        ) * float(unc_pen) * float(type_mult)
        # Live: slightly less aggressive raw strength
        if live:
            raw *= 0.90
        strength = float(np.clip(raw, 0.0, 1.0))
        no_inflate = bool(unc_no_inflate or strength < 0.12)

    target_frac = float(curve["max_ticket_pct"]) * (strength ** float(curve["gamma"]))
    target_pct = 100.0 * max(0.0, target_frac)

    return {
        "strength": strength,
        "strength_score": strength,
        "target_stake_pct": target_pct,
        "edge": float(edge),
        "confidence": conf_label,
        "confidence_score": float(conf_score),
        "model_prob": float(prob),
        "decimal_odds": odds,
        "market_implied": implied,
        "prob_score": float(prob_score),
        "edge_score": float(edge_score),
        "odds_score": float(odds_score),
        "uncertainty_penalty": float(unc_pen),
        "uncertainty_action": unc_action,
        "type_mult": float(type_mult),
        "is_parlay": bool(is_parlay),
        "no_inflate": bool(no_inflate),
        "fail_closed_reason": fail_closed_reason,
        "sizing_mode": "conf_odds",
        "curve_gamma": float(curve["gamma"]),
        "max_ticket_pct": float(curve["max_ticket_pct"]),
        "profile": "live" if live else "paper",
    }


def ticket_allocation_weight(ticket: dict[str, Any], *, live: bool) -> float:
    """
    Relative strength for ranking / diagnostics.

    Primary sizing uses ``compute_ticket_strength`` absolute % mapping.
    """
    details = compute_ticket_strength(ticket, live=live)
    return max(1e-12, float(details["strength"]))


def _is_parlay_ticket(ticket: dict[str, Any]) -> bool:
    kind = str(ticket.get("bet_type") or ticket.get("market_type") or "").lower()
    return bool(ticket.get("is_parlay")) or "parlay" in kind or int(ticket.get("n_legs") or 1) >= 2


def _apply_parlay_share_cap(
    tickets: list[dict[str, Any]],
    pool: float,
    *,
    live: bool,
) -> list[dict[str, Any]]:
    """
    Cap total parlay stake to max_parlay_share of card pool; free % goes to singles first.
    Residual after singles may stay unallocated (do not force into parlays).
    Never redistribute into no-inflate / fail-closed tickets.
    """
    if not tickets or pool <= 0:
        return tickets
    try:
        from src.high_accuracy_strategy import max_parlay_budget_share

        max_share = float(max_parlay_budget_share(live=live))
    except Exception:
        max_share = 0.30 if live else 0.40
    max_share = max(0.0, min(1.0, max_share))
    max_parlay_pct = 100.0 * max_share

    parlays = [t for t in tickets if _is_parlay_ticket(t)]
    singles = [
        t
        for t in tickets
        if not _is_parlay_ticket(t) and not bool(t.get("sizing_no_inflate"))
    ]
    if not parlays:
        return tickets

    parlay_pct = sum(float(t.get("stake_pct") or 0) for t in parlays)
    if parlay_pct <= max_parlay_pct + 1e-6:
        return tickets

    scale = max_parlay_pct / parlay_pct if parlay_pct > 0 else 0.0
    freed = 0.0
    for t in parlays:
        old = float(t.get("stake_pct") or 0)
        new = round(old * scale, 1)
        freed += old - new
        t["stake_pct"] = new
        t["suggested_stake"] = round(pool * (new / 100.0), 2)
        t["parlay_share_capped"] = True

    if singles and freed > 0.05:
        s_weights = [
            max(float(t.get("strength_score") or t.get("stake_pct") or 0), 0.01)
            for t in singles
        ]
        s_total = sum(s_weights) or 1.0
        for t, w in zip(singles, s_weights):
            add = freed * (w / s_total)
            new_pct = float(t.get("stake_pct") or 0) + add
            # Do not push any single above Paper/Live max ticket curve
            max_pct = 100.0 * float(t.get("sizing_max_ticket_pct") or (0.32 if live else 0.48))
            new_pct = min(new_pct, max_pct)
            t["stake_pct"] = round(new_pct, 1)
            t["suggested_stake"] = round(pool * (float(t["stake_pct"]) / 100.0), 2)

        # If still over 100% after redistribution, scale all tickets down (never up)
        all_pct = [float(t.get("stake_pct") or 0) for t in tickets]
        total = sum(all_pct)
        if total > 100.0 + 1e-6:
            scale_all = 100.0 / total
            for t in tickets:
                t["stake_pct"] = round(float(t.get("stake_pct") or 0) * scale_all, 1)
                t["suggested_stake"] = round(pool * (float(t["stake_pct"]) / 100.0), 2)

    return tickets


def _largest_remainder_pct(weights: list[float], *, decimals: int = 1) -> list[float]:
    """Convert weights → percentages that sum exactly to 100."""
    n = len(weights)
    if n == 0:
        return []
    total_w = sum(weights)
    if total_w <= 0:
        even = round(100.0 / n, decimals)
        pcts = [even] * n
        pcts[-1] = round(100.0 - sum(pcts[:-1]), decimals)
        return pcts

    scale = 10**decimals
    target = 100 * scale  # integer hundredths or tenths
    exact = [w / total_w * target for w in weights]
    floors = [int(x) for x in exact]
    rem = target - sum(floors)
    order = sorted(range(n), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[: max(0, rem)]:
        floors[i] += 1
    return [f / scale for f in floors]


def allocate_card_budget_pct(
    tickets: list[dict[str, Any]],
    pool_usd: float,
    *,
    profile: str | None = None,
    inplace: bool = True,
) -> list[dict[str, Any]]:
    """
    Allocate card budget via confidence- and odds-aware strength scores.

    Each ticket gets an absolute target % of card budget from strength.
    Targets are scaled down only when the sum exceeds 100% (never inflated).
    Path-risk caps (parlay share) still apply afterward.

    Sets on each ticket:
      - stake_pct (0–100)
      - suggested_stake (USD = pct/100 * pool)
      - card_pool_usd
      - strength_score, edge, confidence, sizing diagnostics
    """
    import config as _cfg

    if not tickets:
        return []

    live = False
    if profile is not None:
        live = str(profile).strip().lower() == "live"
    else:
        live = bool(_cfg.is_live_profile())

    pool = max(0.0, float(pool_usd or 0.0))
    details_list = [compute_ticket_strength(t, live=live) for t in tickets]
    raw_pcts = [float(d["target_stake_pct"]) for d in details_list]
    total_raw = sum(raw_pcts)
    # Scale down only — leave residual unallocated when tickets are weak
    scale = (100.0 / total_raw) if total_raw > 100.0 + 1e-9 else 1.0
    pcts = [round(p * scale, 1) for p in raw_pcts]
    if scale < 1.0 and pcts:
        # Fix rounding so scaled sum does not exceed 100.0
        drift = round(sum(pcts) - 100.0, 1)
        if drift > 0:
            idx = max(range(len(pcts)), key=lambda i: pcts[i])
            pcts[idx] = round(pcts[idx] - drift, 1)

    out: list[dict[str, Any]] = []
    for i, (ticket, pct, details) in enumerate(zip(tickets, pcts, details_list), start=1):
        row = ticket if inplace else dict(ticket)
        if ticket_edge_exceeds_actionable_cap(row) or str(details.get("fail_closed_reason") or "") == "suspect_edge":
            dollars = 0.0
            pct = 0.0
            row["stake_pct"] = 0.0
            row["suggested_stake"] = 0.0
            row["stake_usd"] = 0.0
            row["advisory"] = True
            row["sizing_no_inflate"] = True
            row["sizing_fail_closed"] = "suspect_edge"
            row["card_pool_usd"] = pool
            row["allocation_rank"] = i
            out.append(row)
            continue
        dollars = round(pool * (pct / 100.0), 2) if pool > 0 else 0.0
        row["stake_pct"] = float(pct)
        row["suggested_stake"] = dollars
        row["card_pool_usd"] = pool
        row["book_pool_usd"] = row.get("book_pool_usd", pool)
        row["allocation_rank"] = i
        row["strength_score"] = float(details["strength_score"])
        row["edge"] = float(details["edge"])
        row["confidence"] = details["confidence"]
        row["confidence_score"] = float(details["confidence_score"])
        row["model_prob"] = float(details["model_prob"])
        row["sizing_mode"] = "conf_odds"
        row["sizing_no_inflate"] = bool(details["no_inflate"])
        row["sizing_fail_closed"] = str(details.get("fail_closed_reason") or "")
        row["sizing_target_pct"] = float(details["target_stake_pct"])
        row["sizing_max_ticket_pct"] = float(details["max_ticket_pct"])
        row["uncertainty_penalty"] = float(details["uncertainty_penalty"])
        row["uncertainty_action"] = details.get("uncertainty_action") or row.get("uncertainty_action")
        if details.get("decimal_odds") is not None and row.get("decimal_odds") is None:
            row["decimal_odds"] = details["decimal_odds"]
        logger.info(
            "HA stake size | strength=%.3f edge=%.3f conf=%s prob=%.3f → %.1f%% ($%.2f) "
            "unc_pen=%.2f no_inflate=%s%s",
            float(details["strength_score"]),
            float(details["edge"]),
            details["confidence"],
            float(details["model_prob"]),
            float(pct),
            dollars,
            float(details["uncertainty_penalty"]),
            bool(details["no_inflate"]),
            f" fail={details['fail_closed_reason']}" if details.get("fail_closed_reason") else "",
        )
        out.append(row)

    out = _apply_parlay_share_cap(out, pool, live=live)

    if out and pool > 0:
        spent = sum(float(t["suggested_stake"]) for t in out)
        # Only absorb tiny rounding drift when we intended full utilization (≥99.5%)
        util = sum(float(t.get("stake_pct") or 0) for t in out)
        drift = round(pool - spent, 2)
        if util >= 99.5 and 0.01 <= abs(drift) <= 0.05:
            singles = [
                t
                for t in out
                if not _is_parlay_ticket(t) and not bool(t.get("sizing_no_inflate"))
            ]
            target = singles or out
            idx = max(range(len(target)), key=lambda j: float(target[j]["suggested_stake"]))
            target[idx]["suggested_stake"] = round(float(target[idx]["suggested_stake"]) + drift, 2)

    return out


def allocate_alerts_card_budget_pct(
    alerts: dict[str, Any],
    pool_usd: float,
    *,
    profile: str | None = None,
    prop_singles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Rank and allocate % across singles + Over 1.5 props + 2-leg parlays in an alert payload.
    """
    out = dict(alerts)
    singles = [dict(s) for s in (alerts.get("singles") or [])]
    parlays = [dict(p) for p in (alerts.get("parlays") or [])]
    props = [dict(s) for s in (prop_singles or alerts.get("prop_singles") or [])]
    props = [
        p
        for p in props
        if str(p.get("prop_key") or "").strip().lower() == "over_1_5_rounds"
        or "over 1.5" in str(p.get("prop_type") or p.get("label") or "").lower()
    ]
    for s in singles:
        s["is_parlay"] = False
        s.setdefault("market_type", "moneyline")
    for p in parlays:
        p["is_parlay"] = True
        if p.get("prob") is None and p.get("combined_prob") is not None:
            p["prob"] = p["combined_prob"]
        if p.get("decimal_odds") is None and p.get("combined_odds") is not None:
            p["decimal_odds"] = p["combined_odds"]
    for p in props:
        p["is_parlay"] = False
        p["market_type"] = "prop"
        p["bet_type"] = p.get("bet_type") or "Over 1.5 Rounds"
        p["prop_key"] = "over_1_5_rounds"

    tickets = [*singles, *props, *parlays]
    if not tickets:
        out["singles"] = singles
        out["parlays"] = parlays
        out["prop_singles"] = props
        out["stake_allocation"] = {"pool_usd": float(pool_usd or 0), "n_tickets": 0, "sum_pct": 0.0}
        return out

    import config as _cfg

    is_live = (
        str(profile).strip().lower() == "live"
        if profile is not None
        else bool(_cfg.is_live_profile())
    )
    tickets_sorted = sorted(
        tickets,
        key=lambda t: ticket_allocation_weight(t, live=is_live),
        reverse=True,
    )
    allocated = allocate_card_budget_pct(tickets_sorted, pool_usd, profile=profile, inplace=True)
    for i, t in enumerate(allocated, start=1):
        t["rank"] = i

    out_singles = [
        t
        for t in allocated
        if not t.get("is_parlay") and str(t.get("prop_key") or "") == ""
    ]
    out_props = [
        t
        for t in allocated
        if str(t.get("prop_key") or "") == "over_1_5_rounds"
        or str(t.get("market_type") or "") == "prop"
    ]
    out_parlays = [t for t in allocated if t.get("is_parlay")]

    out["singles"] = out_singles
    out["parlays"] = out_parlays
    out["prop_singles"] = out_props
    out["stake_allocation"] = {
        "pool_usd": float(pool_usd or 0),
        "n_tickets": len(allocated),
        "sum_pct": round(sum(float(t.get("stake_pct") or 0) for t in allocated), 1),
        "mode": "live" if is_live else "paper",
        "sizing": "conf_odds",
    }
    return out


def format_stake_pct_dollars(ticket: dict[str, Any] | float, stake: float | None = None) -> str:
    """Display '38% | $4.56' for a ticket or (pct, dollars)."""
    if isinstance(ticket, dict):
        pct = ticket.get("stake_pct")
        dollars = stake if stake is not None else ticket.get("suggested_stake")
    else:
        pct = ticket
        dollars = stake
    try:
        pct_f = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        pct_f = None
    try:
        dol_f = float(dollars or 0)
    except (TypeError, ValueError):
        dol_f = 0.0
    if pct_f is None:
        if dol_f <= 0:
            return "— · $0.00"
        return f"${dol_f:.2f}"
    pct_txt = f"{pct_f:.0f}%" if abs(pct_f - round(pct_f)) < 0.05 else f"{pct_f:.1f}%"
    # ASCII separator (Windows ttk / code-page safe)
    return f"{pct_txt} | ${dol_f:.2f}"


def prop_may_receive_ha_stake(prop: dict[str, Any]) -> bool:
    """True only for live Over 1.5 that still clears HA edge/odds (not a 26% scrape)."""
    if str(prop.get("prop_key") or "").strip().lower() != "over_1_5_rounds":
        return False
    if prop.get("strict_qualified") is False:
        return False
    try:
        from src.props import is_live_prop_odds_source

        if not is_live_prop_odds_source(str(prop.get("odds_source") or "")):
            return False
    except Exception:
        if str(prop.get("odds_source") or "").strip().lower() not in {"live", "the_odds_api"}:
            return False
    if ticket_edge_exceeds_actionable_cap(prop):
        return False
    edge_f = ticket_max_edge_fraction(prop)
    if edge_f is None:
        return False
    odds = prop.get("decimal_odds") or prop.get("odds")
    try:
        odds_f = float(odds) if odds is not None else None
    except (TypeError, ValueError):
        odds_f = None
    prob = prop.get("prob")
    try:
        prob_f = float(prob) if prob is not None else None
    except (TypeError, ValueError):
        prob_f = None
    return edge_is_actionable(
        edge_f,
        decimal_odds=odds_f,
        model_prob=prob_f,
        edge_suspect=bool(prop.get("edge_suspect")),
    )


def attach_prop_stakes(
    singles: list[dict[str, Any]],
    budget_state: dict[str, Any] | None,
    book: str,
    *,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """Allocate Over 1.5 (and any remaining prop) stakes as % of the book's card pool."""
    if not singles:
        return []
    if not budget_state:
        return [{**s, "suggested_stake": 0.0, "stake_pct": 0.0} for s in singles]

    resolved = resolve_budget_for_calculations(budget_state, profile=profile)
    if book == "Odds API":
        pool = available_card_budget_usd(resolved, profile=profile)
        if pool <= 0:
            pool, _ = resolve_display_card_budget(resolved, profile=profile)
    else:
        plan = allocate_card_budget_per_book(resolved, profile=profile)
        info = plan.get(book, {})
        if not info.get("enabled"):
            return [{**s, "suggested_stake": 0.0, "stake_pct": 0.0, "book_disabled": True} for s in singles]

        pool = float(info.get("allocation") or 0)
        if pool <= 0 and info.get("enabled"):
            pool = available_card_budget_usd(resolved, profile=profile) / max(
                1,
                sum(1 for b in plan.values() if b.get("enabled")),
            )

    # Only Over 1.5 that still pass HA edge/odds gates get card-budget size.
    tickets = []
    zeroed: list[dict[str, Any]] = []
    for s in singles:
        row = dict(s)
        row["market_type"] = "prop"
        row["is_parlay"] = False
        if not prop_may_receive_ha_stake(row):
            row["suggested_stake"] = 0.0
            row["stake_usd"] = 0.0
            row["stake_pct"] = 0.0
            row["advisory"] = True
            zeroed.append(row)
            continue
        tickets.append(row)
    if not tickets:
        return zeroed
    allocated = allocate_card_budget_pct(tickets, pool, profile=profile, inplace=True)
    return allocated + zeroed


def budget_summary_text(budget_state: dict[str, Any]) -> str:
    """One-line budget summary for the dashboard."""
    import config as _cfg

    card, _ = effective_card_budget_usd(budget_state)
    parts = [
        f"Total Budget: ${float(budget_state.get('total_bankroll') or _cfg.DEFAULT_TOTAL_BANKROLL):,.0f}",
        f"Card Budget: ${card:,.0f}",
    ]
    alloc = allocate_card_budget_per_book(budget_state)
    for book in _cfg.BUDGET_BOOKS:
        info = alloc.get(book, {})
        if not info.get("enabled"):
            continue
        short = book.replace(".eu", "")
        parts.append(f"{short}: ${float(info.get('allocation') or 0):,.2f}")
    return " | ".join(parts)


def live_card_budget_warning(budget_state: dict[str, Any]) -> str | None:
    """Live-only warning when user card budget exceeds recommended safe limits."""
    import config as _cfg

    if not _cfg.is_live_profile():
        return None
    br = max(float(budget_state.get("total_bankroll") or _cfg.DEFAULT_TOTAL_BANKROLL), 1.0)
    raw = float(budget_state.get("card_budget") or 0.0)
    safe_cap = _cfg.max_card_stake_cap(br)
    live_usd = float(_cfg.profile_settings().get("max_card_stake_usd") or _cfg.DEFAULT_CARD_BUDGET)
    cap = min(safe_cap, live_usd)
    if raw <= cap:
        return None
    pct = raw / br * 100.0
    safe_pct = cap / br * 100.0
    return (
        f"Card budget ${raw:,.2f} ({pct:.0f}% of bankroll) exceeds recommended safe limit "
        f"${cap:,.2f} ({safe_pct:.0f}%) for Live mode."
    )


def book_display_name(book: str) -> str:
    return book.replace(".eu", "")


def available_card_budget_usd(
    budget_state: dict[str, Any],
    *,
    profile: str | None = None,
) -> float:
    """Total dollars allocatable this card across enabled books."""
    plan = allocate_card_budget_per_book(budget_state, profile=profile)
    return float(
        sum(float(info.get("allocation") or 0) for info in plan.values() if info.get("enabled"))
    )


def resolve_display_card_budget(
    budget_state: dict[str, Any] | None,
    *,
    profile: str | None = None,
) -> tuple[float, bool]:
    """
    Display card budget (single source of truth for status lines).

    Auto mode → bankroll × profile card-risk % (Live USD cap included in default).
    Override mode → user card_budget from Advanced.
    """
    import config as _cfg

    state = _cfg.normalize_budget_state(budget_state)
    br = float(state.get("total_bankroll") or _cfg.DEFAULT_TOTAL_BANKROLL)
    auto = float(_cfg.default_card_budget_usd(br, profile=profile))
    overridden = bool((budget_state or {}).get("card_budget_overridden"))
    if overridden:
        return float(state.get("card_budget") or auto), True
    return auto, False


def format_card_allocation_status(
    *,
    auto_card_usd: float,
    allocated_usd: float,
    n_tickets: int,
    overridden: bool = False,
    card_budget_usd: float | None = None,
) -> str:
    """Auto card $X · Allocated $Y (Z%) · Tickets N (or Card $X when overridden)."""
    if overridden and card_budget_usd is not None:
        card = max(0.0, float(card_budget_usd))
        label = "Card"
    else:
        card = max(0.0, float(auto_card_usd))
        label = "Auto card"
    allocated = max(0.0, float(allocated_usd or 0.0))
    pct = (100.0 * allocated / card) if card > 0 else 0.0
    return (
        f"{label} ${card:,.2f} · Allocated ${allocated:,.2f} "
        f"({pct:.0f}%) · Tickets {int(n_tickets)}"
    )


def available_card_budget_text(
    budget_state: dict[str, Any],
    *,
    profile: str | None = None,
) -> str:
    """Human label from auto card budget (no leftover 'Available $12' pool wording)."""
    import config as _cfg

    card, overridden = resolve_display_card_budget(budget_state, profile=profile)
    enabled = sum(
        1 for book in _cfg.BUDGET_BOOKS if budget_state.get(_cfg.BUDGET_USE_KEYS[book], True)
    )
    if enabled == 0:
        return "Auto card $0.00 (no books selected)"
    label = "Card" if overridden else "Auto card"
    return f"{label} ${card:,.2f}"


MAX_SAFE_BANKROLL_FRACTION = 0.005  # 0.5% hard cap for "max safe" single bet


def budget_availability_badge_style(
    total_usd: float,
    *,
    books_enabled: bool,
) -> tuple[str, str]:
    """Background and text colors for the Available-this-card badge."""
    if not books_enabled:
        return "#451a1a", "#fca5a5"
    if total_usd > 50:
        return "#14532d", "#86efac"
    if total_usd >= 20:
        return "#713f12", "#fde047"
    return "#451a1a", "#fca5a5"


def bet_sizing_metrics(
    bankroll: float,
    *,
    prob: float | None,
    decimal_odds: float | None,
    edge: float,
    config: StrategyConfig,
    row: pd.Series | dict[str, Any] | None = None,
    prop_type: str | None = None,
    market_type: str = "moneyline",
    rating_mult: float | None = None,
) -> dict[str, float | str]:
    """Kelly stake, Kelly % of bankroll, and max-safe bet (min 0.5% bankroll vs Kelly)."""
    kelly_usd = 0.0
    kelly_pct = 0.0
    half_pct_cap = max(bankroll * MAX_SAFE_BANKROLL_FRACTION, 0.0)
    applied_mult = 1.0
    unc_mult = 1.0
    unc_reason = ""
    unc_action = "allow"
    if (
        prob is not None
        and decimal_odds is not None
        and bankroll > 0
        and float(edge) >= config.min_edge
    ):
        try:
            from src.strategy_rating import kelly_multiplier_for_context

            applied_mult = (
                float(rating_mult)
                if rating_mult is not None
                else kelly_multiplier_for_context(
                    row=row,
                    decimal_odds=decimal_odds,
                    prop_type=prop_type,
                    market_type=market_type,
                )
            )
        except Exception:
            applied_mult = 1.0
        try:
            from src.uncertainty_gates import evaluate_uncertainty_gate, effective_min_edge

            gate = evaluate_uncertainty_gate(row)
            unc_action = gate.action
            unc_reason = gate.reason_label()
            if gate.skip:
                unc_mult = 0.0
            else:
                if gate.tighten and float(edge) < effective_min_edge(config.min_edge, gate):
                    unc_mult = 0.0
                    unc_action = "skip"
                    unc_reason = unc_reason or "below_tightened_min_edge"
                else:
                    unc_mult = float(gate.kelly_mult)
        except Exception:
            unc_mult = 0.0
            unc_action = "skip"
            unc_reason = "missing_uncertainty"

        if unc_mult > 0:
            kelly_usd = kelly_stake(
                bankroll,
                prob=float(prob),
                decimal_odds=float(decimal_odds),
                edge=float(edge),
                config=config,
                rating_mult=applied_mult,
                row=row,
                prop_type=prop_type,
                market_type=market_type,
                uncertainty_kelly_mult=unc_mult,
            )
            if kelly_usd > 0:
                kelly_pct = kelly_usd / bankroll * 100.0
    max_safe = min(half_pct_cap, kelly_usd) if kelly_usd > 0 else half_pct_cap
    return {
        "kelly_stake_usd": round(kelly_usd, 2),
        "kelly_pct": round(kelly_pct, 2),
        "max_safe_bet_usd": round(max_safe, 2),
        "strategy_rating_mult": round(float(applied_mult), 4),
        "uncertainty_kelly_mult": round(float(unc_mult), 4),
        "uncertainty_action": unc_action,
        "uncertainty_reason": unc_reason,
    }


def bankroll_from_budget(budget_state: dict[str, Any] | None) -> float:
    import config as _cfg

    if not budget_state:
        return float(_cfg.INITIAL_BANKROLL)
    return float(budget_state.get("total_bankroll") or _cfg.INITIAL_BANKROLL)


def scale_alerts_to_book_pool(
    alerts: dict[str, Any],
    budget_state: dict[str, Any] | None,
    book: str,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Allocate singles + Over 1.5 props + parlays as % of the book's card pool (sum 100%)."""
    if not alerts or not budget_state:
        return alerts

    resolved = resolve_budget_for_calculations(budget_state, profile=profile)

    if book in ("Overview", "Odds API"):
        pool = available_card_budget_usd(resolved, profile=profile)
        if book == "Odds API" and pool <= 0:
            pool, _ = resolve_display_card_budget(resolved, profile=profile)
    else:
        plan = allocate_card_budget_per_book(resolved, profile=profile)
        info = plan.get(book, {})
        if not info.get("enabled"):
            return {**alerts, "singles": [], "parlays": [], "book_disabled": True}
        pool = float(info.get("allocation") or 0)

    props = list(alerts.get("prop_singles") or [])
    return allocate_alerts_card_budget_pct(
        alerts,
        pool,
        profile=profile,
        prop_singles=props,
    )


def budget_aware_alerts(
    alerts: dict[str, Any],
    budget_state: dict[str, Any] | None,
    book: str,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Apply book enablement and card-budget % stake allocation to an alert payload."""
    if not budget_state:
        return alerts
    return scale_alerts_to_book_pool(alerts, budget_state, book, profile=profile)


def collect_dashboard_risk_warnings(
    alerts: dict[str, Any] | None,
    budget_state: dict[str, Any] | None,
    *,
    bankroll: float | None = None,
) -> list[tuple[str, str]]:
    """
    Unified risk warnings for dashboard tabs.

    Returns list of (severity, message) where severity is critical | warn | info.
    """
    import config as _cfg

    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(severity: str, msg: str) -> None:
        key = msg.strip()
        if not key or key in seen:
            return
        seen.add(key)
        out.append((severity, key))

    br = float(
        (budget_state or {}).get("total_bankroll")
        or (alerts or {}).get("bankroll")
        or bankroll
        or _cfg.INITIAL_BANKROLL
    )

    if budget_state:
        budget_warn = live_card_budget_warning(budget_state)
        if budget_warn:
            _add("critical", budget_warn)
        card_eff, cap_warnings = effective_card_budget_usd(budget_state)
        for w in cap_warnings:
            _add("warn", w)
        raw_card = float(budget_state.get("card_budget") or 0)
        if _cfg.is_live_profile():
            live_cap = _cfg.live_card_budget_cap_usd(br)
            if raw_card > live_cap:
                _add(
                    "critical",
                    f"Card budget ${raw_card:,.2f} exceeds Live hard cap ${live_cap:,.2f}.",
                )
        elif card_eff < raw_card:
            _add("warn", f"Card budget trimmed to safe cap ${card_eff:,.2f} for current profile.")

    for w in _cfg.live_small_bankroll_warnings(br):
        _add("critical", w)

    if alerts:
        live_warn = _cfg.live_card_risk_warning(alerts, bankroll=br)
        if live_warn:
            _add("critical", live_warn)
        stake = _cfg.estimated_card_stake_usd(alerts)
        cap = _cfg.max_card_stake_cap(br)
        if _cfg.is_live_profile() and stake > 0 and stake > cap:
            _add(
                "critical",
                f"Suggested stakes ${stake:,.2f} exceed your ${cap:,.2f} live card cap.",
            )
        for w in alerts.get("warnings") or []:
            _add("warn", str(w))
        skipped_n = int(alerts.get("skipped_count") or len(alerts.get("skipped") or []))
        if skipped_n:
            _add("info", f"Uncertainty gates skipped {skipped_n} fight(s).")

    return out


def format_risk_warnings(
    warnings: list[tuple[str, str]],
    *,
    max_lines: int = 4,
    separator: str = "  |  ",
) -> tuple[str, str]:
    """Return (display_text, color_hex) for a warning label."""
    if not warnings:
        return "", "#9ca3af"
    critical = [m for s, m in warnings if s == "critical"]
    shown = (critical or [m for s, m in warnings])[:max_lines]
    text = separator.join(shown)
    if critical:
        return f"⚠ {text}", "#f87171"
    return f"⚠ {text}", "#fbbf24"


def _format_american_odds(decimal: float | None) -> str:
    if decimal is None or decimal <= 1:
        return "-"
    if decimal >= 2.0:
        return f"+{int(round((decimal - 1.0) * 100))}"
    return str(int(round(-100.0 / (decimal - 1.0))))


def format_pick_over_opponent(fight: str, pick: str) -> str:
    """e.g. 'Michael Chandler over Mauricio Ruffy'."""
    if " vs " not in fight or not pick:
        return pick or fight
    f1, f2 = [x.strip() for x in fight.split(" vs ", 1)]
    if pick == f1:
        return f"{pick} over {f2}"
    if pick == f2:
        return f"{pick} over {f1}"
    return pick


def _find_prediction_row(preds: pd.DataFrame, single: dict[str, Any]) -> pd.Series | None:
    if preds is None or preds.empty:
        return None
    fid = str(single.get("fight_id") or "")
    fight = str(single.get("fight") or "")
    for _, row in preds.iterrows():
        row_fid = str(row.get("fight_id", ""))
        if fid and row_fid and row_fid == fid:
            return row
        f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
        f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
        if fight and f"{f1} vs {f2}" == fight:
            return row
    return None


def decimal_odds_for_pick(row: pd.Series, pick: str) -> float | None:
    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    if pick == f1:
        val = row.get("f1_odds")
    else:
        val = row.get("f2_odds")
    return sanitize_decimal_odds(val)


def sanitize_decimal_odds(val: Any) -> float | None:
    """
    Normalize to decimal odds in (1, 15].

    Accepts decimal prices or American moneylines (±100+). Rejects
    scraper mistakes (e.g. American +133 stored without sign handling).
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        raw = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(raw):
        return None

    # Already decimal
    if 1.0 < raw <= 15.0:
        return raw

    # American moneyline → decimal
    if raw >= 100.0:
        dec = 1.0 + raw / 100.0
    elif raw <= -100.0:
        dec = 1.0 + 100.0 / abs(raw)
    else:
        return None

    if 1.0 < dec <= 15.0:
        return dec
    return None


MAX_ACTIONABLE_EDGE = 0.25  # 25% — filters bogus scraper edges (e.g. MyBookie glitches)
SUSPECT_EDGE_FLAG = 0.25


def _as_edge_fraction(value: Any, *, percent_points: bool = False) -> float | None:
    """Normalize one edge field to a fraction.

    ``edge`` is a fraction, or percent points when |value| > 1.5.
    ``edge_pct`` is always percent points (26.3 means 26.3%).
    """
    v = _coerce_edge_number(value)
    if v is None:
        return None
    if percent_points:
        return v / 100.0
    if abs(v) > 1.5:
        return v / 100.0
    return v


def _coerce_edge_number(value: Any) -> float | None:
    """Parse edge fields that may be floats or strings like '+26.3%'."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("%", "").replace("\uff05", "")
        if s.startswith("+"):
            s = s[1:]
        if not s:
            return None
        value = s
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def ticket_max_edge_fraction(
    ticket: dict[str, Any] | None = None,
    *,
    edge: Any = None,
    edge_pct: Any = None,
) -> float | None:
    """Largest |edge| implied by sizing ``edge`` and displayed ``edge_pct``."""
    blob: dict[str, Any]
    if isinstance(ticket, dict):
        blob = ticket
    elif ticket is not None:
        try:
            blob = dict(ticket)
        except Exception:
            blob = {}
    else:
        blob = {}
    e = edge if edge is not None else blob.get("edge")
    ep = edge_pct if edge_pct is not None else blob.get("edge_pct")
    fracs: list[float] = []
    ef = _as_edge_fraction(e, percent_points=False)
    epf = _as_edge_fraction(ep, percent_points=True)
    if ef is not None:
        fracs.append(ef)
    if epf is not None:
        fracs.append(epf)
    if not fracs:
        return None
    return max(fracs, key=lambda x: abs(x))


def ticket_edge_exceeds_actionable_cap(
    ticket: dict[str, Any] | None = None,
    *,
    edge: Any = None,
    edge_pct: Any = None,
) -> bool:
    """True if sizing ``edge`` or displayed ``edge_pct`` is above the 25% HA cap.

    Tickets can store ``edge=0.08`` (passes the cap) and ``edge_pct=26.3``
    (what Overview/Ollama print). Either field over 25% is a bogus scrape.
    """
    frac = ticket_max_edge_fraction(ticket, edge=edge, edge_pct=edge_pct)
    if frac is not None and abs(frac) > MAX_ACTIONABLE_EDGE:
        return True
    blob: dict[str, Any]
    if isinstance(ticket, dict):
        blob = ticket
    elif ticket is not None:
        try:
            blob = dict(ticket)
        except Exception:
            blob = {}
    else:
        blob = {}
    printed = _coerce_edge_number(edge_pct if edge_pct is not None else blob.get("edge_pct"))
    return printed is not None and abs(printed) > (MAX_ACTIONABLE_EDGE * 100.0)


def edge_is_actionable(
    edge: float,
    *,
    decimal_odds: float | None = None,
    model_prob: float | None = None,
    edge_suspect: bool = False,
) -> bool:
    """Drop absurd edges that usually mean bad odds merges, not real value."""
    if edge_suspect:
        return False
    if edge <= 0 or edge > MAX_ACTIONABLE_EDGE:
        return False
    if decimal_odds is not None and decimal_odds > 1.0 and model_prob is not None:
        implied = 1.0 / decimal_odds
        if edge > 0.10 and abs(float(model_prob) - implied) > 0.30:
            return False
    return True


def aggregate_top_recommended_bets(
    books: dict[str, dict[str, Any]],
    budget_state: dict[str, Any],
    *,
    limit: int = 5,
    per_book_cap: int = 2,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """
    Best singles across enabled books, deduped by fight, stakes scaled to card budget.

    Takes up to ``per_book_cap`` edges from each enabled book before global dedupe so
    Overview is not dominated by a single sportsbook.
    """
    import config as _cfg

    resolved = resolve_budget_for_calculations(budget_state, profile=profile)
    enabled = [
        book
        for book in _cfg.BUDGET_BOOKS
        if resolved.get(_cfg.BUDGET_USE_KEYS[book], True)
    ]
    bankroll = bankroll_from_budget(resolved)
    strategy = strategy_from_profile(bankroll=bankroll)
    pool_candidates: list[dict[str, Any]] = []

    for book in enabled:
        book_data = books.get(book, {})
        alerts = book_data.get("alerts") or {}
        preds = book_data.get("predictions")
        if not isinstance(preds, pd.DataFrame):
            preds = pd.DataFrame()
        book_singles = sorted(
            alerts.get("singles") or [],
            key=lambda x: float(x.get("edge") or 0),
            reverse=True,
        )[: max(1, per_book_cap)]
        for single in book_singles:
            edge = float(single.get("edge") or 0)
            fid = str(single.get("fight_id") or single.get("fight") or "")
            if not fid:
                continue
            row = _find_prediction_row(preds, single)
            pick = str(single.get("pick") or "")
            fight = str(single.get("fight") or "")
            dec = decimal_odds_for_pick(row, pick) if row is not None else None
            prob_val = single.get("prob")
            prob_f = float(prob_val) if prob_val is not None else None
            max_edge = ticket_max_edge_fraction(single, edge=edge)
            if max_edge is None or not edge_is_actionable(
                max_edge,
                decimal_odds=dec,
                model_prob=prob_f,
                edge_suspect=bool(row.get("edge_suspect")) if row is not None else False,
            ):
                continue
            confidence = str(single.get("confidence") or "").strip()
            if not confidence and row is not None:
                confidence = str(row.get("confidence_label") or "").strip()
            sizing = bet_sizing_metrics(
                bankroll,
                prob=prob_f,
                decimal_odds=dec,
                edge=edge,
                config=strategy,
                row=row,
                market_type="moneyline",
            )
            if str(sizing.get("uncertainty_action") or "") == "skip":
                continue
            if float(sizing.get("kelly_stake_usd") or 0) <= 0 and str(
                sizing.get("uncertainty_action") or ""
            ) in ("skip", "tighten"):
                # Tightened below floor → treat as non-actionable for overview
                if str(sizing.get("uncertainty_action")) == "skip":
                    continue
            pool_candidates.append(
                {
                    "fight_id": fid,
                    "fight": fight,
                    "pick": pick,
                    "pick_line": format_pick_over_opponent(fight, pick),
                    "bet_type": "Moneyline Single",
                    "description": str(single.get("brief") or single.get("reasoning") or "").strip(),
                    "prob": prob_f,
                    "edge": edge,
                    "edge_pct": float(single.get("edge_pct") or edge * 100),
                    "confidence": confidence or "-",
                    "kelly_pct": sizing["kelly_pct"],
                    "kelly_stake_usd": sizing["kelly_stake_usd"],
                    "max_safe_bet_usd": sizing["max_safe_bet_usd"],
                    "strategy_rating_mult": sizing.get("strategy_rating_mult", 1.0),
                    "uncertainty_action": sizing.get("uncertainty_action", "allow"),
                    "uncertainty_reason": sizing.get("uncertainty_reason", ""),
                    "uncertainty_kelly_mult": sizing.get("uncertainty_kelly_mult", 1.0),
                    "ensemble_disagreement": (
                        float(row["ensemble_disagreement"])
                        if row is not None and pd.notna(row.get("ensemble_disagreement"))
                        else single.get("ensemble_disagreement")
                    ),
                    "interval_width": (
                        float(row["interval_width"])
                        if row is not None and pd.notna(row.get("interval_width"))
                        else single.get("interval_width")
                    ),
                    "book": book_display_name(book),
                    "book_key": book,
                    "decimal_odds": dec,
                    "odds_display": f"{dec:.2f}" if dec else "-",
                    "american_odds": _format_american_odds(dec),
                    "raw_stake": float(single.get("suggested_stake") or 0),
                    "gym_note": str(row.get("gym_matchup_note") or "") if row is not None else "",
                    "sos_note": str(row.get("sos_competition_note") or "") if row is not None else "",
                    "f1_gym": str(row.get("f1_gym") or "") if row is not None else "",
                    "f2_gym": str(row.get("f2_gym") or "") if row is not None else "",
                    "f1_gym_strengths": str(row.get("f1_gym_strengths") or "") if row is not None else "",
                    "f2_gym_strengths": str(row.get("f2_gym_strengths") or "") if row is not None else "",
                    "event": str(
                        single.get("event")
                        or single.get("event_name")
                        or (row.get("event_name") if row is not None else "")
                        or ""
                    ),
                    "event_name": str(
                        single.get("event_name")
                        or single.get("event")
                        or (row.get("event_name") if row is not None else "")
                        or ""
                    ),
                }
            )

    # Dedupe on (fight, market, selection, book) — not fight-only (avoids #1/#6 twins)
    try:
        from src.bet_slip import dedupe_rank_top_tickets

        ranked = dedupe_rank_top_tickets(pool_candidates, limit=limit, event="overview_ml")
    except Exception:
        best_by_key: dict[tuple, dict[str, Any]] = {}
        for bet in pool_candidates:
            key = (
                str(bet.get("fight_id") or ""),
                str(bet.get("market_type") or "moneyline"),
                str(bet.get("pick") or ""),
                str(bet.get("book") or "").lower(),
            )
            prev = best_by_key.get(key)
            if prev is None or float(bet.get("edge") or 0) > float(prev.get("edge") or 0):
                best_by_key[key] = bet
        ranked = sorted(
            best_by_key.values(), key=lambda x: float(x.get("edge") or 0), reverse=True
        )[:limit]
    pool = available_card_budget_usd(resolved, profile=profile)
    for i, bet in enumerate(ranked, start=1):
        bet["rank"] = i
        bet["card_pool_usd"] = pool
        bet["is_parlay"] = False
        bet.setdefault("market_type", "moneyline")
    return ranked


def _overview_over_15_props(
    books: dict[str, dict[str, Any]],
    *,
    limit: int = 2,
    allowed_fights: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect allowed Over 1.5 prop singles from book payloads for the overview slip."""
    import config as _cfg

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    book_order = list(
        dict.fromkeys(
            (
                *(_cfg.BUDGET_BOOKS or ()),
                "Odds API",
                "MyBookie",
                "Overview",
                "Consensus",
                *tuple(books or {}),
            )
        )
    )
    for book in book_order:
        book_data = books.get(book) or {}
        if not isinstance(book_data, dict):
            continue
        alerts = book_data.get("alerts") or {}
        props = list(alerts.get("prop_singles") or [])
        # Dashboard props live at books[X]["props"]["singles"]
        props_blob = book_data.get("props")
        if isinstance(props_blob, dict):
            props.extend(list(props_blob.get("singles") or []))
        extra = book_data.get("prop_singles") or book_data.get("ranked_props") or []
        if isinstance(extra, list):
            props.extend(extra)
        for p in props:
            if not isinstance(p, dict):
                continue
            key = str(p.get("prop_key") or "").strip().lower()
            label = str(p.get("prop_type") or p.get("label") or p.get("prop_short") or "").lower()
            if key != "over_1_5_rounds" and "over 1.5" not in label:
                continue
            # HA overview slip: only strict + live Over 1.5 (relaxed/synthetic stay fun-only)
            try:
                from src.props import is_live_prop_odds_source

                odds_src = str(p.get("odds_source") or "")
                if not is_live_prop_odds_source(odds_src):
                    continue
            except Exception:
                if str(p.get("odds_source") or "").strip().lower() in {
                    "",
                    "synthetic",
                    "model",
                    "fair",
                }:
                    continue
            if p.get("strict_qualified") is False:
                continue
            if not prop_may_receive_ha_stake({**p, "prop_key": "over_1_5_rounds"}):
                continue
            fid = str(p.get("fight_id") or p.get("fight") or "")
            if not fid or fid in seen:
                continue
            if allowed_fights and fid not in allowed_fights and str(p.get("fight") or "") not in allowed_fights:
                continue
            edge = float(p.get("edge") or 0)
            if p.get("edge_pct") is not None and not edge:
                try:
                    edge = float(p["edge_pct"]) / 100.0
                except (TypeError, ValueError):
                    edge = 0.0
            edge_pct = p.get("edge_pct")
            try:
                edge_pct_f = float(edge_pct) if edge_pct is not None else edge * 100.0
            except (TypeError, ValueError):
                edge_pct_f = edge * 100.0
            row = {
                **dict(p),
                "fight_id": fid,
                "prop_key": "over_1_5_rounds",
                "market_type": "prop",
                "bet_type": "Over 1.5 Rounds",
                "is_parlay": False,
                "display_label": str(
                    p.get("display_label")
                    or p.get("label")
                    or p.get("prop_short")
                    or "Over 1.5 Rounds"
                ),
                "edge": edge,
                "edge_pct": edge_pct_f,
                "book": book_display_name(book),
                "book_key": book,
            }
            out.append(row)
            seen.add(fid)
            if len(out) >= limit:
                return out
    return out


def aggregate_overview_recommendations(
    books: dict[str, dict[str, Any]],
    budget_state: dict[str, Any],
    *,
    limit: int = 5,
    max_parlays: int = 2,
    profile: str | None = None,
    allowed_fights: set[str] | None = None,
) -> dict[str, Any]:
    """
    Compact Overview slip: top singles + Over 1.5 + 2-leg parlays.

    Stakes are conf/odds strength → % of card budget (sum ≤ 100%).
    """
    from src.high_accuracy_strategy import clamp_max_tickets

    singles_cap = max(3, min(5, int(limit)))
    parlays_cap = max(0, min(2, int(max_parlays)))
    ticket_cap = clamp_max_tickets(
        None  # profile default
    )
    singles = aggregate_top_recommended_bets(
        books, budget_state, limit=singles_cap, per_book_cap=2, profile=profile
    )
    # Fill remaining slots from Overview alerts if cross-book dedupe left gaps
    if len(singles) < singles_cap:
        overview_singles = (books.get("Overview", {}).get("alerts") or {}).get("singles") or []
        seen = {s.get("fight_id") for s in singles}
        for s in sorted(overview_singles, key=lambda x: float(x.get("edge") or 0), reverse=True):
            fid = str(s.get("fight_id") or s.get("fight") or "")
            if not fid or fid in seen:
                continue
            edge = float(s.get("edge") or 0)
            prob = s.get("prob")
            prob_f = float(prob) if prob is not None else None
            max_edge = ticket_max_edge_fraction(s, edge=edge)
            if max_edge is None or not edge_is_actionable(max_edge, model_prob=prob_f):
                continue
            brief = str(s.get("brief") or s.get("reasoning") or "").strip()
            singles.append(
                {
                    "fight_id": fid,
                    "fight": s.get("fight"),
                    "pick": s.get("pick"),
                    "pick_line": format_pick_over_opponent(str(s.get("fight") or ""), str(s.get("pick") or "")),
                    "bet_type": "Moneyline Single",
                    "description": brief,
                    "brief": brief,
                    "edge": float(s.get("edge") or 0),
                    "edge_pct": float(s.get("edge_pct") or float(s.get("edge") or 0) * 100),
                    "prob": s.get("prob"),
                    "confidence": s.get("confidence") or s.get("confidence_label"),
                    "book": "Overview",
                    "book_key": "Overview",
                    "suggested_stake": float(s.get("suggested_stake") or 0),
                    "american_odds": "-",
                    "odds_display": "-",
                    "is_parlay": False,
                    "market_type": "moneyline",
                    "event": str(s.get("event") or s.get("event_name") or ""),
                    "event_name": str(s.get("event_name") or s.get("event") or ""),
                }
            )
            seen.add(fid)
            if len(singles) >= singles_cap:
                break
    for single in singles:
        pick = str(single.get("pick") or "").strip()
        single["display_label"] = f"{pick} ML" if pick else str(single.get("pick_line") or "-")
        if not single.get("brief") and single.get("description"):
            single["brief"] = single["description"]
        single["is_parlay"] = False
        single.setdefault("market_type", "moneyline")

    cleaned_singles: list[dict[str, Any]] = []
    for item in singles:
        max_edge = ticket_max_edge_fraction(item)
        if max_edge is None:
            continue
        prob = item.get("prob")
        prob_f = float(prob) if prob is not None else None
        dec = item.get("decimal_odds")
        dec_f = float(dec) if dec is not None else None
        if edge_is_actionable(max_edge, decimal_odds=dec_f, model_prob=prob_f):
            cleaned_singles.append(item)
    singles = cleaned_singles

    if allowed_fights:
        filtered = [
            b
            for b in singles
            if str(b.get("fight_id") or b.get("fight") or "") in allowed_fights
            or str(b.get("fight") or "") in allowed_fights
        ]
        if filtered:
            singles = filtered

    singles = singles[:singles_cap]

    props = _overview_over_15_props(books, limit=2, allowed_fights=allowed_fights)

    parlays = aggregate_top_parlays(books, budget_state, limit=parlays_cap, profile=profile)
    for i, p in enumerate(parlays, start=1):
        p["rank"] = i
        p["is_parlay"] = True
        legs = str(p.get("pick_line") or "").strip()
        bt = str(p.get("bet_type") or "Parlay")
        leg_n = bt.split("-Leg")[0].strip() if "-Leg" in bt else ""
        if legs and leg_n.isdigit():
            p["display_label"] = f"{legs} {leg_n}-leg"
        else:
            p["display_label"] = legs or bt

    resolved = resolve_budget_for_calculations(budget_state, profile=profile)
    pool = available_card_budget_usd(resolved, profile=profile)

    # Prefer singles, then Over 1.5, then parlays within ticket cap
    tickets: list[dict[str, Any]] = []
    for s in singles:
        if len(tickets) >= ticket_cap:
            break
        tickets.append(s)
    for p in props:
        if len(tickets) >= ticket_cap:
            break
        tickets.append(p)
    for p in parlays:
        if len(tickets) >= ticket_cap:
            break
        tickets.append(p)

    import config as _cfg

    is_live = (
        str(profile).strip().lower() == "live"
        if profile is not None
        else bool(_cfg.is_live_profile())
    )
    tickets = sorted(
        tickets,
        key=lambda t: ticket_allocation_weight(t, live=is_live),
        reverse=True,
    )
    allocated = allocate_card_budget_pct(tickets, pool, profile=profile, inplace=True)
    for i, t in enumerate(allocated, start=1):
        t["rank"] = i

    out_singles = [t for t in allocated if not t.get("is_parlay") and str(t.get("prop_key") or "") == ""]
    out_props = [
        t
        for t in allocated
        if str(t.get("prop_key") or "") == "over_1_5_rounds" or str(t.get("market_type") or "") == "prop"
    ]
    out_parlays = [t for t in allocated if t.get("is_parlay")]

    return {
        "singles": out_singles,
        "parlays": out_parlays,
        "prop_singles": out_props,
        "card_pool_usd": pool,
        "stake_allocation": {
            "pool_usd": pool,
            "n_tickets": len(allocated),
            "sum_pct": round(sum(float(t.get("stake_pct") or 0) for t in allocated), 1),
            "mode": "live" if is_live else "paper",
            "sizing": "conf_odds",
        },
        # Flat list kept for callers that still iterate one sequence.
        "items": allocated,
    }


def aggregate_top_parlays(
    books: dict[str, dict[str, Any]],
    budget_state: dict[str, Any],
    *,
    limit: int = 2,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """Top N positive-EV parlays across enabled books (deduped by leg set)."""
    import config as _cfg
    from src.parlay_builder import enrich_parlays_for_display, format_recommended_parlay_header

    resolved = resolve_budget_for_calculations(budget_state, profile=profile)
    enabled = [
        book
        for book in _cfg.BUDGET_BOOKS
        if resolved.get(_cfg.BUDGET_USE_KEYS[book], True)
    ]
    pool = available_card_budget_usd(resolved, profile=profile)
    candidates: list[dict[str, Any]] = []

    for book in enabled:
        book_data = books.get(book, {})
        alerts = book_data.get("alerts") or {}
        preds = book_data.get("predictions")
        if not isinstance(preds, pd.DataFrame):
            preds = pd.DataFrame()
        parlays = alerts.get("parlays") or []
        if not parlays:
            continue
        ranked = sorted(parlays, key=lambda x: float(x.get("expected_value", 0)), reverse=True)
        for top in ranked[: max(1, limit)]:
            ev = float(top.get("expected_value", 0) or 0)
            if ev <= 0:
                continue
            enriched = enrich_parlays_for_display([top], preds)
            p = enriched[0] if enriched else dict(top)
            combined_dec = float(p.get("combined_odds", 0) or 0)
            legs_txt = str(p.get("picks") or "")
            if not legs_txt and p.get("legs"):
                from src.parlay_builder import leg_pick_label

                legs_txt = " + ".join(leg_pick_label(leg) for leg in p["legs"])
            stake = round(min(float(p.get("suggested_stake") or 0), pool * 0.25), 2)
            n_legs = int(p.get("n_legs", 2) or 2)
            candidates.append(
                {
                    "bet_type": f"{n_legs}-Leg Parlay",
                    "n_legs": n_legs,
                    "pick_line": legs_txt or format_recommended_parlay_header({**p, "rank": 1}),
                    "picks": legs_txt,
                    "description": f"{n_legs}-leg +EV slip",
                    "brief": f"{n_legs}-leg +EV slip",
                    "prob": p.get("combined_prob"),
                    "edge": float(p.get("min_leg_edge", 0) or 0),
                    "edge_pct": float(p.get("min_leg_edge", 0) or 0) * 100,
                    "book": book_display_name(book),
                    "book_key": book,
                    "decimal_odds": combined_dec if combined_dec > 1 else None,
                    "combined_odds": combined_dec if combined_dec > 1 else None,
                    "american_odds": _format_american_odds(combined_dec if combined_dec > 1 else None),
                    "odds_display": f"{combined_dec:.2f}" if combined_dec > 1 else "-",
                    "suggested_stake": stake,
                    "expected_value": ev,
                    "card_pool_usd": pool,
                    "is_parlay": True,
                }
            )

    # Dedupe by normalized leg text; keep highest EV
    best_by_key: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = " ".join(str(item.get("pick_line") or "").lower().split())
        prev = best_by_key.get(key)
        if prev is None or float(item.get("expected_value") or 0) > float(prev.get("expected_value") or 0):
            best_by_key[key] = item
    ranked_out = sorted(
        best_by_key.values(),
        key=lambda x: float(x.get("expected_value") or 0),
        reverse=True,
    )
    return ranked_out[: max(0, int(limit))]


def aggregate_best_parlay(
    books: dict[str, dict[str, Any]],
    budget_state: dict[str, Any],
    *,
    profile: str | None = None,
) -> dict[str, Any] | None:
    """Highest-EV parlay across enabled books (for Overview highlight)."""
    top = aggregate_top_parlays(books, budget_state, limit=1, profile=profile)
    return top[0] if top else None


def apply_narrative_tilt_after_model_sizing(
    bets: list[dict[str, Any]] | dict[str, Any],
    narrative_result: dict[str, Any] | None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Final sizing step: narrative Kelly tilt after uncertainty gates + strategy rating.

    Model pick/edge stay locked; only stake fields may move within profile ±bounds.
    """
    from src.grok_analysis import apply_grok_kelly_adjustments

    return apply_grok_kelly_adjustments(bets, narrative_result)

