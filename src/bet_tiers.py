"""Fight-row / fun-bet color tiers — math + final decision, not vibes.

Legend (exact meaning):
- Deep Blue = full HA ticket (clears gates, normal real stake)
- Sky Blue  = Paper wide override only (tiny stake, paper_wide_override)
- Green     = decent fun bet (positive edge / solid model lean, but NOT a real ticket)
- Yellow    = caution (thin edge, borderline prob, or soft uncertainty)
- Red       = don't bet (negative edge, low model prob, or hard skip)

Rules (apply in this order):
1. If Paper + paper_wide_override + stake_pct > 0 → SKY_BLUE
2. Else if final decision is BET / stake% > 0 / clears HA gates → BLUE (deep)
3. Else if edge < 0 → RED
4. Else if model_prob < 0.52 OR status is hard skip (no_odds / low_model_prob) → RED
5. Else if edge >= 0.05 AND model_prob >= 0.60 AND SKIP for uncertainty/wide_interval only → GREEN
6. Else → YELLOW
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

TIER_BLUE = "blue"
TIER_SKY_BLUE = "sky_blue"
TIER_GREEN = "green"
TIER_YELLOW = "yellow"
TIER_RED = "red"

# Sort: Deep Blue → Sky Blue → Green → Yellow → Red
TIER_ORDER = (TIER_BLUE, TIER_SKY_BLUE, TIER_GREEN, TIER_YELLOW, TIER_RED)
TIER_SORT_RANK = {
    TIER_BLUE: 0,
    TIER_SKY_BLUE: 1,
    TIER_GREEN: 2,
    TIER_YELLOW: 3,
    TIER_RED: 4,
}

# Hex must stay visually distinct on dark UI (yellow ≠ red, red ≠ gold).
TIER_COLORS = {
    TIER_BLUE: "#3b82f6",  # deep blue — full HA
    TIER_SKY_BLUE: "#57B9FF",  # sky blue — Paper wide override only
    TIER_GREEN: "#22c55e",
    TIER_YELLOW: "#eab308",  # gold yellow — clearly not red
    TIER_RED: "#f87171",  # soft red — clearly not yellow
}

TIER_LABELS = {
    TIER_BLUE: "Blue",
    TIER_SKY_BLUE: "Sky Blue",
    TIER_GREEN: "Green",
    TIER_YELLOW: "Yellow",
    TIER_RED: "Red",
}

# Plain-English action verbs for Ollama / Top recommended (what the user should do).
TIER_ACTIONS = {
    TIER_BLUE: "BET THIS",
    TIER_SKY_BLUE: "TINY PAPER BET",
    TIER_GREEN: "FUN ONLY",
    TIER_YELLOW: "CAUTION — SKIP SIZED",
    TIER_RED: "DO NOT BET",
}

_ACTIONABLE_TIERS = frozenset({TIER_BLUE, TIER_SKY_BLUE})
_PAPER_WIDE_OVERRIDE_TOKEN = "paper_wide_override"

# Color thresholds (do not change HA sizing gates).
_RED_PROB_FLOOR = 0.52
_GREEN_MIN_EDGE = 0.05
_GREEN_MIN_PROB = 0.60

_HARD_SKIP = frozenset(
    {
        "no_odds",
        "low_model_prob",
        "no_pick",
    }
)
_SOFT_UNCERTAINTY_SKIP = frozenset(
    {
        "wide_interval",
        "high_disagreement",
    }
)


def _is_paper_profile() -> bool:
    try:
        import config

        return bool(config.is_paper_profile())
    except Exception:
        return False


def _has_paper_wide_override_reason(
    *,
    uncertainty_reason: str | None = None,
    status: str | None = None,
    row: pd.Series | dict[str, Any] | None = None,
) -> bool:
    blob = " ".join(
        [
            str(uncertainty_reason or ""),
            str(status or ""),
        ]
    ).lower().replace("-", "_").replace(" ", "_")
    if _PAPER_WIDE_OVERRIDE_TOKEN in blob:
        return True
    if row is None:
        return False
    try:
        from src.uncertainty_gates import PAPER_WIDE_OVERRIDE, evaluate_uncertainty_gate

        gate = evaluate_uncertainty_gate(row)
        if gate.primary_reason == PAPER_WIDE_OVERRIDE:
            return True
        if PAPER_WIDE_OVERRIDE in (gate.reasons or []):
            return True
    except Exception:
        return False
    return False


def _parse_stake_pct_from_status(status: str | None) -> float | None:
    """Parse leading '1.25%' from Kelly/status strings like '1.00% paper_wide_override'."""
    text = str(status or "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def is_sky_blue_ticket(
    *,
    stake_pct: float | None = None,
    stake_usd: float | None = None,
    uncertainty_reason: str | None = None,
    status: str | None = None,
    row: pd.Series | dict[str, Any] | None = None,
) -> bool:
    """Paper-only sky blue: paper_wide_override + positive stake_pct."""
    if not _is_paper_profile():
        return False
    if not _has_paper_wide_override_reason(
        uncertainty_reason=uncertainty_reason, status=status, row=row
    ):
        return False
    pct = _safe_float(stake_pct)
    if pct is None:
        pct = _parse_stake_pct_from_status(status)
    if pct is not None and pct > 0:
        return True
    # Fallback: positive USD stake counts when pct not carried on the ticket
    usd = _safe_float(stake_usd)
    return usd is not None and usd > 0


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _row_pick_edge(row: pd.Series | dict[str, Any]) -> tuple[float | None, str | None]:
    from src.alerts import _pick_edge

    if isinstance(row, dict):
        row = pd.Series(row)
    return _pick_edge(row)


def _row_model_prob(row: pd.Series | dict[str, Any], pick: str | None) -> float | None:
    if isinstance(row, dict):
        row = pd.Series(row)
    f1 = str(row.get("fighter_1") or row.get("fighter1") or "")
    f2 = str(row.get("fighter_2") or row.get("fighter2") or "")
    raw = row.get("predicted_prob")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        if pick and pick == f2:
            raw = row.get("prob_f2_win")
        else:
            raw = row.get("prob_f1_win")
    return _safe_float(raw)


def _fight_keys(row: pd.Series | dict[str, Any]) -> set[str]:
    if isinstance(row, dict):
        row = pd.Series(row)
    f1 = str(row.get("fighter_1") or row.get("fighter1") or "").strip()
    f2 = str(row.get("fighter_2") or row.get("fighter2") or "").strip()
    fid = str(row.get("fight_id") or "").strip()
    keys = {k for k in (fid, f"{f1} vs {f2}", f"{f2} vs {f1}") if k}
    return keys


def _normalize_skip_reason(reason: str | None) -> str:
    text = str(reason or "").strip().lower().replace(" ", "_")
    if text.startswith("skip:"):
        text = text[5:]
    # Truncated Kelly labels: SKIP:wide / SKIP:high_s
    if text.startswith("wide"):
        return "wide_interval"
    if text.startswith("high_s") or text.startswith("high_d"):
        return "high_disagreement"
    if text.startswith("missing"):
        return "missing_uncertainty"
    if text.startswith("low_m") or text.startswith("low_p"):
        return "low_model_prob"
    if text.startswith("no_odd") or text == "noodds":
        return "no_odds"
    aliases = {
        "wide": "wide_interval",
        "wide_ci": "wide_interval",
        "interval": "wide_interval",
        "disagreement": "high_disagreement",
        "high_disagree": "high_disagreement",
        "missing": "missing_uncertainty",
        "missing_unc": "missing_uncertainty",
        "noodds": "no_odds",
        "low_prob": "low_model_prob",
        "lowprob": "low_model_prob",
    }
    return aliases.get(text, text)


def singles_cleared_keys(singles: list[dict[str, Any]] | None) -> set[str]:
    """Keys for HA singles with real money stake only (BLUE candidates)."""
    keys: set[str] = set()
    for s in singles or []:
        if s.get("advisory") or s.get("fun_bet"):
            continue
        stake = _safe_float(s.get("suggested_stake"))
        if stake is None:
            stake = _safe_float(s.get("stake_usd"))
        stake_pct = _safe_float(s.get("stake_pct"))
        if (stake is None or stake <= 0) and (stake_pct is None or stake_pct <= 0):
            continue
        for k in (
            str(s.get("fight_id") or "").strip(),
            str(s.get("fight") or "").strip(),
        ):
            if k:
                keys.add(k)
    return keys


def row_clears_gates(row: pd.Series | dict[str, Any], cleared: set[str]) -> bool:
    if not cleared:
        return False
    return bool(_fight_keys(row) & cleared)


def resolve_row_decision(
    row: pd.Series | dict[str, Any],
    *,
    clears_gates: bool = False,
    stake_pct: float | None = None,
    stake_usd: float | None = None,
    min_model_prob: float | None = None,
) -> dict[str, Any]:
    """Derive pick/edge/prob + final decision label used for color rules."""
    series = pd.Series(row) if isinstance(row, dict) else row

    edge, pick = _row_pick_edge(series)
    prob = _row_model_prob(series, pick)
    stake_pct_f = _safe_float(stake_pct)
    stake_usd_f = _safe_float(stake_usd)
    if stake_pct_f is None:
        stake_pct_f = _safe_float(series.get("stake_pct"))
    if stake_usd_f is None:
        stake_usd_f = _safe_float(series.get("suggested_stake"))
        if stake_usd_f is None:
            stake_usd_f = _safe_float(series.get("stake_usd"))

    # Real money only — never treat advisory / zero-stake as BET.
    actionable = (
        (stake_pct_f is not None and stake_pct_f > 0)
        or (stake_usd_f is not None and stake_usd_f > 0)
        or bool(clears_gates)
    )
    if actionable and (
        (stake_pct_f is not None and stake_pct_f > 0)
        or (stake_usd_f is not None and stake_usd_f > 0)
        or clears_gates
    ):
        # clears_gates already means stake>0 singles; keep BET.
        return {
            "decision": "BET",
            "skip_reason": "",
            "pick": pick,
            "edge": edge,
            "model_prob": prob,
            "stake_pct": stake_pct_f,
            "stake_usd": stake_usd_f,
        }

    if not pick:
        return {
            "decision": "SKIP",
            "skip_reason": "no_pick",
            "pick": pick,
            "edge": edge,
            "model_prob": prob,
            "stake_pct": stake_pct_f,
            "stake_usd": stake_usd_f,
        }
    if edge is None:
        return {
            "decision": "SKIP",
            "skip_reason": "no_odds",
            "pick": pick,
            "edge": edge,
            "model_prob": prob,
            "stake_pct": stake_pct_f,
            "stake_usd": stake_usd_f,
        }

    # Match Kelly column: uncertainty SKIP takes precedence.
    try:
        from src.uncertainty_gates import evaluate_uncertainty_gate

        gate = evaluate_uncertainty_gate(series)
        if gate.skip:
            reason = _normalize_skip_reason(gate.reason_label() or gate.primary_reason)
            return {
                "decision": "SKIP",
                "skip_reason": reason or "wide_interval",
                "pick": pick,
                "edge": edge,
                "model_prob": prob,
                "stake_pct": stake_pct_f,
                "stake_usd": stake_usd_f,
            }
    except Exception:
        pass

    floor = 0.70 if min_model_prob is None else float(min_model_prob)
    if prob is None or prob < floor:
        return {
            "decision": "SKIP",
            "skip_reason": "low_model_prob",
            "pick": pick,
            "edge": edge,
            "model_prob": prob,
            "stake_pct": stake_pct_f,
            "stake_usd": stake_usd_f,
        }

    return {
        "decision": "SKIP",
        "skip_reason": "not_actionable",
        "pick": pick,
        "edge": edge,
        "model_prob": prob,
        "stake_pct": stake_pct_f,
        "stake_usd": stake_usd_f,
    }


def classify_bet_tier(
    row: pd.Series | dict[str, Any] | None = None,
    *,
    clears_gates: bool = False,
    stake_pct: float | None = None,
    stake_usd: float | None = None,
    min_model_prob: float | None = None,
    status: str | None = None,
    edge: float | None = None,
    model_prob: float | None = None,
    pick: str | None = None,
    uncertainty_reason: str | None = None,
    debug: bool = True,
) -> tuple[str, str]:
    """Classify color from final decision/status, edge, model_prob, stake.

    HARD RULES:
    1. SKY_BLUE if Paper + paper_wide_override + stake_pct > 0
    2. BLUE (deep) only if stake>0 / BET|TAKE|PLAY. If status contains SKIP → NEVER blue.
    3. RED if edge < 0
    4. RED if displayed model_prob < 52% OR no usable odds
    5. GREEN if edge >= 0.05 and model_prob >= 0.60 and decision is SKIP
    6. YELLOW otherwise
    """
    # Kelly / status column often carries "1.00% paper_wide_override" while the
    # prediction row has no stake_* fields. Parse before resolve so book tables
    # match Ollama (which paints sky from ticket stake + reason).
    if stake_pct is None and status:
        stake_pct = _parse_stake_pct_from_status(status)

    if row is not None:
        info = resolve_row_decision(
            row,
            clears_gates=clears_gates,
            stake_pct=stake_pct,
            stake_usd=stake_usd,
            min_model_prob=min_model_prob,
        )
        if edge is None:
            edge = info.get("edge")
        if model_prob is None:
            model_prob = info.get("model_prob")
        if pick is None:
            pick = info.get("pick")
        if stake_pct is None:
            stake_pct = info.get("stake_pct")
        if stake_usd is None:
            stake_usd = info.get("stake_usd")
        if uncertainty_reason is None:
            try:
                uncertainty_reason = str(pd.Series(row).get("uncertainty_reason") or "") or None
            except Exception:
                uncertainty_reason = None
        if status is None:
            if info.get("decision") == "BET":
                status = "BET"
            elif info.get("skip_reason"):
                status = f"SKIP:{info['skip_reason']}"
            else:
                status = str(info.get("decision") or "")
    else:
        info = {
            "decision": "BET" if (stake_pct and stake_pct > 0) or (stake_usd and stake_usd > 0) else "SKIP",
            "skip_reason": _normalize_skip_reason(status),
        }

    status_s = str(status or "")
    status_u = status_s.upper()
    decision = str(info.get("decision") or "")
    skip_reason = _normalize_skip_reason(status_s or info.get("skip_reason"))
    edge_f = _safe_float(edge)
    prob_f = _safe_float(model_prob)
    stake_pct_f = _safe_float(stake_pct)
    stake_usd_f = _safe_float(stake_usd)
    if stake_pct_f is None:
        stake_pct_f = _parse_stake_pct_from_status(status_s)

    # --- Rule 1: SKY BLUE — Paper wide override tiny stake only ---
    # SKIP in the Kelly/status text is authoritative. Do not let resolve_row_decision
    # "SKIP" override a positive stake % status (book tables pass Kelly text only).
    has_stake = (stake_pct_f is not None and stake_pct_f > 0) or (
        stake_usd_f is not None and stake_usd_f > 0
    )
    is_bet_word = status_u in {"BET", "TAKE", "PLAY"} or status_u.startswith("BET")
    status_has_skip = "SKIP" in status_u
    is_skip = status_has_skip or (decision == "SKIP" and not has_stake and not is_bet_word)
    actionable = not is_skip and (
        has_stake or is_bet_word or (clears_gates and decision == "BET")
    )
    if actionable and is_sky_blue_ticket(
        stake_pct=stake_pct_f,
        stake_usd=stake_usd_f,
        uncertainty_reason=uncertainty_reason,
        status=status_s,
        row=row,
    ):
        # Require positive stake_pct (or parseable %) for sky; USD-only already gated above
        tier, reason = TIER_SKY_BLUE, "paper_wide_override"
        _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
        return tier, reason

    # --- Rule 2: DEEP BLUE only for real money; SKIP never blue ---
    if actionable:
        tier, reason = TIER_BLUE, "stake_or_bet_decision"
        _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
        return tier, reason

    # --- Rule 3: negative edge ---
    if edge_f is not None and edge_f < 0:
        tier, reason = TIER_RED, "negative_edge"
        _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
        return tier, reason

    # --- Rule 4: low display-prob or no odds ---
    # Display-aligned: round(prob*100) < 52 so shown "52%" can still be yellow.
    if edge_f is None or skip_reason == "no_odds":
        tier, reason = TIER_RED, "no_usable_odds"
        _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
        return tier, reason
    if prob_f is None or round(float(prob_f) * 100) < 52:
        tier, reason = TIER_RED, "model_prob_below_52"
        _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
        return tier, reason

    # --- Rule 5: GREEN strong lean + SKIP (e.g. SKIP:wide / Guilherme) ---
    if (
        is_skip
        and edge_f >= _GREEN_MIN_EDGE
        and prob_f is not None
        and float(prob_f) >= _GREEN_MIN_PROB
    ):
        tier, reason = TIER_GREEN, f"skip_strong_lean_{skip_reason or 'skip'}"
        _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
        return tier, reason

    # --- Rule 6: YELLOW ---
    tier, reason = TIER_YELLOW, "borderline_or_thin"
    _log_color(pick, prob_f, edge_f, status_s, stake_pct_f, stake_usd_f, tier, reason, debug)
    return tier, reason


def _log_color(
    pick: Any,
    prob: float | None,
    edge: float | None,
    status: Any,
    stake_pct: float | None,
    stake_usd: float | None,
    tier: str,
    reason: str,
    debug: bool,
) -> None:
    if not debug:
        return
    stake_s = (
        f"pct={stake_pct}"
        if stake_pct is not None
        else (f"usd={stake_usd}" if stake_usd is not None else "stake=None")
    )
    msg = (
        f"tier_color pick={pick!r} prob={prob if prob is None else f'{prob:.4f}'} "
        f"edge={edge if edge is None else f'{edge:+.4f}'} status={status!r} "
        f"{stake_s} color={tier} reason={reason}"
    )
    logger.info(msg)


def _bet_dict_from_row(
    row: pd.Series,
    *,
    tier: str,
    reason: str,
    book: str = "",
) -> dict[str, Any]:
    edge, pick = _row_pick_edge(row)
    f1 = str(row.get("fighter_1") or row.get("fighter1") or "").strip()
    f2 = str(row.get("fighter_2") or row.get("fighter2") or "").strip()
    fight = f"{f1} vs {f2}"
    fid = str(row.get("fight_id") or f"{f1}|{f2}").strip()
    prob = _row_model_prob(row, pick)
    edge_f = float(edge) if edge is not None else 0.0
    book_name = (
        book
        or str(
            row.get("bookmaker")
            or row.get("odds_book")
            or row.get("odds_source")
            or "n/a"
        ).strip()
        or "n/a"
    )
    if book_name.lower() in {"the_odds_api", "odds_api"}:
        book_name = "Odds API"

    from src.strategy import format_pick_over_opponent

    pick_s = str(pick or "-")
    label = format_pick_over_opponent(fight, pick_s) if pick_s != "-" else fight
    return {
        "fight_id": fid,
        "fight": fight,
        "pick": pick_s,
        "pick_line": label,
        "display_label": f"{pick_s} ML" if pick_s != "-" else label,
        "bet_type": "Moneyline Single",
        "edge": edge_f,
        "edge_pct": edge_f * 100.0,
        "prob": prob,
        "confidence": str(row.get("confidence_label") or row.get("confidence") or "").strip(),
        "book": book_name,
        "book_key": book_name,
        "suggested_stake": 0.0,
        "stake_usd": 0.0,
        "stake_pct": 0.0,
        "american_odds": "-",
        "odds_display": "-",
        "is_parlay": False,
        "market_type": "moneyline",
        "bet_tier": tier,
        "tier": tier,
        "tier_label": TIER_LABELS.get(tier, tier),
        "tier_reason": reason,
        "fun_bet": tier in {TIER_GREEN, TIER_YELLOW},
        "advisory": tier not in _ACTIONABLE_TIERS,
        "brief": f"{TIER_LABELS.get(tier, tier)} — {reason.replace('_', ' ')}",
        "description": f"{TIER_LABELS.get(tier, tier)} — {reason.replace('_', ' ')}",
        "event": str(row.get("event_name") or row.get("event") or "").strip(),
        "event_name": str(row.get("event_name") or row.get("event") or "").strip(),
    }


def rank_card_bet_tiers(
    predictions: pd.DataFrame | None,
    *,
    cleared_singles: list[dict[str, Any]] | None = None,
    limit_per_tier: int = 8,
    min_model_prob: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Classify every fight; return buckets + ranked fun list (green then yellow)."""
    out: dict[str, list[dict[str, Any]]] = {
        TIER_BLUE: [],
        TIER_SKY_BLUE: [],
        TIER_GREEN: [],
        TIER_YELLOW: [],
        TIER_RED: [],
        "best_fun": [],
    }
    if predictions is None or getattr(predictions, "empty", True):
        for s in cleared_singles or []:
            stake = _safe_float(s.get("suggested_stake")) or _safe_float(s.get("stake_usd")) or 0.0
            stake_pct = _safe_float(s.get("stake_pct")) or 0.0
            if stake <= 0 and stake_pct <= 0:
                continue
            item = dict(s)
            unc = str(s.get("uncertainty_reason") or "")
            if is_sky_blue_ticket(
                stake_pct=stake_pct if stake_pct > 0 else None,
                stake_usd=stake if stake > 0 else None,
                uncertainty_reason=unc,
            ):
                tier = TIER_SKY_BLUE
            else:
                tier = TIER_BLUE
            item["bet_tier"] = tier
            item["tier"] = tier
            item["tier_label"] = TIER_LABELS[tier]
            item["fun_bet"] = False
            item["advisory"] = False
            out[tier].append(item)
        return out

    if min_model_prob is None:
        min_model_prob = 0.70
        try:
            from src.strategy import strategy_from_profile
            import config as _cfg

            strat = strategy_from_profile(
                bankroll=float(getattr(_cfg, "INITIAL_BANKROLL", 100) or 100)
            )
            min_model_prob = float(
                getattr(strat, "min_model_prob", min_model_prob) or min_model_prob
            )
        except Exception:
            pass

    cleared = singles_cleared_keys(cleared_singles)
    # Map fight keys → alert single (for stake + uncertainty_reason on override tickets)
    cleared_by_key: dict[str, dict[str, Any]] = {}
    for s in cleared_singles or []:
        for k in (
            str(s.get("fight_id") or "").strip(),
            str(s.get("fight") or "").strip(),
        ):
            if k and k not in cleared_by_key:
                cleared_by_key[k] = s

    seen: set[str] = set()
    for _, row in predictions.iterrows():
        keys = _fight_keys(row)
        dedupe = next(iter(keys), "")
        if dedupe and dedupe in seen:
            continue
        if dedupe:
            seen.add(dedupe)
        clears = bool(keys & cleared)
        alert = None
        for k in keys:
            if k in cleared_by_key:
                alert = cleared_by_key[k]
                break
        stake_pct = _safe_float(alert.get("stake_pct")) if alert else None
        stake_usd = None
        if alert:
            stake_usd = _safe_float(alert.get("suggested_stake"))
            if stake_usd is None:
                stake_usd = _safe_float(alert.get("stake_usd"))
        unc = str(alert.get("uncertainty_reason") or "") if alert else None
        tier, reason = classify_bet_tier(
            row,
            clears_gates=clears,
            stake_pct=stake_pct,
            stake_usd=stake_usd,
            uncertainty_reason=unc,
            min_model_prob=min_model_prob,
        )
        item = _bet_dict_from_row(row, tier=tier, reason=reason)
        if alert and tier in _ACTIONABLE_TIERS:
            if stake_usd is not None and stake_usd > 0:
                item["suggested_stake"] = stake_usd
                item["stake_usd"] = stake_usd
            if stake_pct is not None and stake_pct > 0:
                item["stake_pct"] = stake_pct
            if unc:
                item["uncertainty_reason"] = unc
        out[tier].append(item)

    for tier in TIER_ORDER:
        out[tier].sort(key=lambda b: float(b.get("edge") or 0), reverse=True)
        out[tier] = out[tier][:limit_per_tier]

    fun = list(out[TIER_GREEN]) + list(out[TIER_YELLOW])
    fun.sort(
        key=lambda b: (
            0 if b.get("bet_tier") == TIER_GREEN else 1,
            -float(b.get("edge") or 0),
            -(float(b.get("prob") or 0)),
        )
    )
    out["best_fun"] = fun[:limit_per_tier]
    return out


def format_tier_legend() -> str:
    return (
        "BET THIS (Blue) = sized bankroll ticket that passed HA gates | "
        "TINY PAPER BET (Sky blue) = failed wide CI — Paper override only, not Live HA | "
        "FUN ONLY (Green) = $0 research lean — not sized | "
        "CAUTION (Yellow) = skip sized bankroll | "
        "DO NOT BET (Red)"
    )


def action_label_for_bet(bet: dict[str, Any] | None) -> str:
    """Return plain action for UI/prompt: BET THIS / FUN ONLY / DO NOT BET."""
    b = bet or {}
    tier = str(b.get("bet_tier") or b.get("tier") or "").strip().lower()
    if tier == "advisory":
        tier = TIER_GREEN if b.get("fun_bet") else TIER_YELLOW
    if tier not in TIER_ACTIONS:
        # Fail closed: never invent Blue/Sky from leftover stake fields alone.
        # Only explicit bet_tier (from HA clears) can say BET THIS.
        if b.get("fun_bet") or b.get("advisory"):
            tier = TIER_GREEN
        else:
            tier = TIER_YELLOW

    action = TIER_ACTIONS.get(tier, "CAUTION — SKIP SIZED")
    if tier in _ACTIONABLE_TIERS:
        stake = _safe_float(b.get("stake_usd"))
        if stake is None:
            stake = _safe_float(b.get("suggested_stake"))
        if stake is not None and stake > 0:
            return f"{action} ${stake:.2f}"
        stake_pct = _safe_float(b.get("stake_pct"))
        if stake_pct is not None and stake_pct > 0:
            return f"{action} ({stake_pct:.1f}% card)"
        return action
    if tier == TIER_GREEN:
        return f"{action} ($0 — not a real ticket)"
    if tier == TIER_YELLOW:
        return f"{action} ($0)"
    return action


def _header_pick_names(bets: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    names: list[str] = []
    for b in bets[:limit]:
        side = str(b.get("pick") or b.get("side") or "—")
        names.append(f"{side} ({action_label_for_bet(b)})")
    return names


def format_what_to_do_header(
    tiers: dict[str, list[dict[str, Any]]] | None = None,
    *,
    slip: list[dict[str, Any]] | None = None,
) -> str:
    """Lead line: HA-passed Blue vs Paper override (failed wide CI) vs fun/skip."""
    blue: list[dict[str, Any]] = []
    sky: list[dict[str, Any]] = []
    green: list[dict[str, Any]] = []
    if isinstance(tiers, dict):
        blue = list(tiers.get(TIER_BLUE) or [])
        sky = list(tiers.get(TIER_SKY_BLUE) or [])
        green = list(tiers.get(TIER_GREEN) or [])
    elif slip:
        for b in slip:
            t = str(b.get("bet_tier") or "").strip().lower()
            if t == TIER_BLUE:
                blue.append(b)
            elif t == TIER_SKY_BLUE:
                sky.append(b)
            elif t == TIER_GREEN or b.get("fun_bet"):
                green.append(b)

    ha_line = "WHAT TO BET (HA — passed gates): NONE"
    if blue:
        ha_line = "WHAT TO BET (HA — passed gates): " + " · ".join(_header_pick_names(blue))

    if sky:
        override = (
            "PAPER OVERRIDE (failed wide CI — not Live HA): "
            + " · ".join(_header_pick_names(sky))
        )
        return f"{ha_line}. {override}"

    if blue:
        return ha_line

    if green:
        names = [str(b.get("pick") or b.get("side") or "—") for b in green[:3]]
        return (
            "WHAT TO BET (HA — passed gates): NONE — bankroll stays flat. "
            f"FUN ONLY leans (not sized): {', '.join(names)}."
        )
    return "WHAT TO BET (HA — passed gates): NONE — NO BET this card."


def prop_status_for_tier(prop: dict[str, Any]) -> str:
    """Map prop payload fields onto classify_bet_tier status (HA stake → BET).

    Display/card-budget stakes must NEVER turn synthetic or relaxed props into BET/Blue.
    Synthetic and non-strict lines are always SKIP (Green/Yellow leans only).
    """
    odds_source = str(prop.get("odds_source") or "").strip().lower()
    if odds_source in {"", "synthetic", "model", "fair"}:
        return "SKIP:relaxed"
    if prop.get("strict_qualified") is False:
        return "SKIP:relaxed"

    try:
        from src.props import is_live_prop_odds_source

        live = is_live_prop_odds_source(odds_source)
    except Exception:
        live = odds_source in {"live", "the_odds_api"}

    key = str(prop.get("prop_key") or "").strip().lower()
    # HA actionable props are Over 1.5 only
    if key and key not in {"over_1_5_rounds", "under_1_5_rounds", "round_1_finish"}:
        return "SKIP:prop_gate"

    stake = _safe_float(prop.get("suggested_stake"))
    if stake is None:
        stake = _safe_float(prop.get("stake_usd"))
    stake_pct = _safe_float(prop.get("stake_pct"))
    has_stake = (stake is not None and stake > 0) or (stake_pct is not None and stake_pct > 0)

    # Live + strict Over 1.5 with real HA size → BET (Blue candidate)
    if live and has_stake and key in {"", "over_1_5_rounds"}:
        return "BET"

    edge = _safe_float(prop.get("edge"))
    if edge is None and prop.get("edge_pct") is not None:
        try:
            edge = float(prop["edge_pct"]) / 100.0
        except (TypeError, ValueError):
            edge = None
    if edge is None:
        return "SKIP:no_odds"
    return "SKIP:prop_gate"


def classify_prop_bet_tier(prop: dict[str, Any], *, debug: bool = False) -> tuple[str, str]:
    """Color-tier a prop single with the same Blue/Green/Yellow/Red math as ML."""
    edge = _safe_float(prop.get("edge"))
    if edge is None and prop.get("edge_pct") is not None:
        try:
            edge = float(prop["edge_pct"]) / 100.0
        except (TypeError, ValueError):
            edge = None
    stake = _safe_float(prop.get("suggested_stake"))
    if stake is None:
        stake = _safe_float(prop.get("stake_usd"))
    return classify_bet_tier(
        None,
        status=prop_status_for_tier(prop),
        edge=edge,
        model_prob=_safe_float(prop.get("prob")),
        stake_pct=_safe_float(prop.get("stake_pct")),
        stake_usd=stake,
        uncertainty_reason=str(prop.get("uncertainty_reason") or "") or None,
        pick=str(prop.get("label") or prop.get("prop_short") or prop.get("pick") or "Over 1.5"),
        debug=debug,
    )


def prop_bet_dict_from_row(prop: dict[str, Any], *, tier: str, reason: str) -> dict[str, Any]:
    """Normalize a prop single into a fun-tier / slip-compatible dict."""
    edge = _safe_float(prop.get("edge")) or 0.0
    if abs(edge) > 1.5:
        edge = edge / 100.0
    edge_pct = _safe_float(prop.get("edge_pct"))
    if edge_pct is None:
        edge_pct = edge * 100.0
    fight = str(prop.get("fight") or "").strip()
    from src.props import PROP_MARKET_LABELS, event_from_record

    event = event_from_record(prop)
    label = str(
        prop.get("display_label")
        or prop.get("label")
        or prop.get("prop_short")
        or PROP_MARKET_LABELS.get(str(prop.get("prop_key") or ""), "")
        or "Prop"
    ).strip()
    stake = _safe_float(prop.get("suggested_stake"))
    if stake is None:
        stake = _safe_float(prop.get("stake_usd")) or 0.0
    stake_pct = _safe_float(prop.get("stake_pct")) or 0.0
    actionable = stake > 0 or stake_pct > 0
    return {
        **dict(prop),
        "fight_id": str(prop.get("fight_id") or fight),
        "fight": fight,
        "event": event,
        "event_name": event,
        "pick": label,
        "pick_line": f"{fight} — {label}" if fight and fight not in label else label,
        "display_label": label,
        "bet_type": label,
        "market_type": "prop",
        "prop_key": str(prop.get("prop_key") or "over_1_5_rounds"),
        "edge": edge,
        "edge_pct": edge_pct,
        "prob": _safe_float(prop.get("prob")),
        "book": str(prop.get("book") or prop.get("book_key") or "n/a"),
        "book_key": str(prop.get("book_key") or prop.get("book") or ""),
        "suggested_stake": stake if actionable else 0.0,
        "stake_usd": stake if actionable else 0.0,
        "stake_pct": stake_pct if actionable else 0.0,
        "odds": prop.get("odds") or prop.get("decimal_odds"),
        "decimal_odds": prop.get("decimal_odds") or prop.get("odds"),
        "odds_display": str(prop.get("odds_display") or prop.get("american_odds") or "-"),
        "is_parlay": False,
        "bet_tier": tier,
        "tier": tier,
        "tier_label": TIER_LABELS.get(tier, tier),
        "tier_reason": reason,
        "fun_bet": (not actionable) and tier in {TIER_GREEN, TIER_YELLOW},
        "advisory": not actionable,
        "brief": f"{TIER_LABELS.get(tier, tier)} — {reason.replace('_', ' ')}",
        "description": f"{TIER_LABELS.get(tier, tier)} — {reason.replace('_', ' ')}",
    }


def collect_props_from_books(
    books: dict[str, dict[str, Any]] | None,
    *,
    limit: int = 6,
    allowed_fights: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Deduped Over 1.5 (and ranked) prop singles from book props.singles payloads."""
    import config as _cfg

    if not books:
        return []
    preferred = (
        *(_cfg.BUDGET_BOOKS or ()),
        "Odds API",
        "MyBookie",
        "Overview",
        "Consensus",
    )
    book_order = list(dict.fromkeys((*preferred, *tuple(books))))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for book in book_order:
        book_data = books.get(book) or {}
        if not isinstance(book_data, dict):
            continue
        alerts = book_data.get("alerts") or {}
        props = list(alerts.get("prop_singles") or [])
        props_blob = book_data.get("props")
        if isinstance(props_blob, dict):
            props.extend(list(props_blob.get("singles") or []))
        for extra_key in ("prop_singles", "ranked_props"):
            extra = book_data.get(extra_key) or []
            if isinstance(extra, list):
                props.extend(extra)
        for p in props:
            if not isinstance(p, dict):
                continue
            key = str(p.get("prop_key") or "").strip().lower()
            label = str(p.get("prop_type") or p.get("label") or p.get("prop_short") or "").lower()
            if key and key != "over_1_5_rounds" and "over 1.5" not in label:
                # HA actionable props are Over 1.5 only; skip other markets here.
                continue
            if not key and "over 1.5" not in label:
                continue
            fid = str(p.get("fight_id") or p.get("fight") or "").strip()
            if not fid:
                continue
            dedupe = f"{fid}|{key or 'over_1_5_rounds'}".lower()
            if dedupe in seen:
                continue
            if allowed_fights and fid not in allowed_fights and str(p.get("fight") or "") not in allowed_fights:
                continue
            row = dict(p)
            row.setdefault("book", book)
            row.setdefault("book_key", book)
            row.setdefault("prop_key", "over_1_5_rounds")
            row.setdefault("market_type", "prop")
            out.append(row)
            seen.add(dedupe)
            if len(out) >= max(1, int(limit) * 3):
                # gather extra then rank/trim in rank_prop_bet_tiers
                break
        if len(out) >= max(1, int(limit) * 3):
            break
    return out


def rank_prop_bet_tiers(
    props: list[dict[str, Any]] | None,
    *,
    limit_per_tier: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    """Classify prop singles into Blue/Green/Yellow/Red buckets."""
    out: dict[str, list[dict[str, Any]]] = {
        TIER_BLUE: [],
        TIER_SKY_BLUE: [],
        TIER_GREEN: [],
        TIER_YELLOW: [],
        TIER_RED: [],
        "best_fun": [],
    }
    for p in props or []:
        # Classify on a copy so card-budget display stakes cannot invent Blue
        # for synthetic / relaxed MyBookie lines.
        gate = dict(p)
        src = str(gate.get("odds_source") or "").strip().lower()
        if src in {"", "synthetic", "model", "fair"} or gate.get("strict_qualified") is False:
            gate["suggested_stake"] = 0.0
            gate["stake_usd"] = 0.0
            gate["stake_pct"] = 0.0
        tier, reason = classify_prop_bet_tier(gate, debug=False)
        item = prop_bet_dict_from_row(gate if tier not in _ACTIONABLE_TIERS else p, tier=tier, reason=reason)
        if tier not in _ACTIONABLE_TIERS:
            item["suggested_stake"] = 0.0
            item["stake_usd"] = 0.0
            item["stake_pct"] = 0.0
            item["fun_bet"] = True
            item["advisory"] = True
        out[tier].append(item)

    for tier in TIER_ORDER:
        out[tier].sort(key=lambda b: float(b.get("edge") or 0), reverse=True)
        out[tier] = out[tier][:limit_per_tier]

    fun = list(out[TIER_GREEN]) + list(out[TIER_YELLOW])
    fun.sort(
        key=lambda b: (
            0 if b.get("bet_tier") == TIER_GREEN else 1,
            -float(b.get("edge") or 0),
            -(float(b.get("prob") or 0)),
        )
    )
    out["best_fun"] = fun[:limit_per_tier]
    return out


def merge_bet_tier_dicts(
    primary: dict[str, list[dict[str, Any]]] | None,
    secondary: dict[str, list[dict[str, Any]]] | None,
    *,
    limit_per_tier: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Merge ML + prop tier buckets (dedupe by fight_id|prop_key|pick)."""
    base = {
        TIER_BLUE: [],
        TIER_SKY_BLUE: [],
        TIER_GREEN: [],
        TIER_YELLOW: [],
        TIER_RED: [],
        "best_fun": [],
    }
    seen: set[str] = set()

    def _key(b: dict[str, Any]) -> str:
        return "|".join(
            [
                str(b.get("fight_id") or b.get("fight") or "").strip().lower(),
                str(b.get("prop_key") or b.get("market_type") or "ml").strip().lower(),
                str(b.get("pick") or b.get("side") or "").strip().lower(),
            ]
        )

    for src in (primary or {}, secondary or {}):
        for tier in TIER_ORDER:
            for b in src.get(tier) or []:
                k = _key(b)
                if not k or k in seen:
                    continue
                seen.add(k)
                base[tier].append(dict(b))

    for tier in TIER_ORDER:
        base[tier].sort(key=lambda b: float(b.get("edge") or 0), reverse=True)
        base[tier] = base[tier][:limit_per_tier]

    fun = list(base[TIER_GREEN]) + list(base[TIER_YELLOW])
    fun.sort(
        key=lambda b: (
            0 if b.get("bet_tier") == TIER_GREEN else 1,
            -float(b.get("edge") or 0),
            -(float(b.get("prob") or 0)),
        )
    )
    base["best_fun"] = fun[:limit_per_tier]
    return base


def format_tiered_best_bets(
    tiers: dict[str, list[dict[str, Any]]],
    *,
    event: str = "",
    compact: bool = False,
) -> str:
    """Text briefing for Ollama chat / Stats — gates stay strict for Blue."""
    blue = list(tiers.get(TIER_BLUE) or [])
    sky = list(tiers.get(TIER_SKY_BLUE) or [])
    green = list(tiers.get(TIER_GREEN) or [])
    yellow = list(tiers.get(TIER_YELLOW) or [])

    if compact:
        lines: list[str] = []
        if event:
            lines.append(str(event).strip())
        lines.append(format_what_to_do_header(tiers=tiers))
        if blue:
            lines.append("")
            lines.append("HA passed gates:")
            for i, b in enumerate(blue[:5], start=1):
                lines.append(_line_compact(b, i))
        if sky:
            lines.append("")
            lines.append("Paper override (failed wide CI):")
            for i, b in enumerate(sky[:5], start=1):
                lines.append(_line_compact(b, i))
        if green:
            lines.append("")
            lines.append("Fun only ($0):")
            for i, b in enumerate(green[:4], start=1):
                lines.append(_line_compact(b, i, fun=True))
        if yellow:
            lines.append("")
            lines.append("Skip sized:")
            for i, b in enumerate(yellow[:3], start=1):
                lines.append(_line_compact(b, i, fun=True))
        return "\n".join(lines).strip()

    lines: list[str] = []
    title = "Best bets — read the action verbs"
    if event:
        title += f" — {event}"
    lines.append(title)
    lines.append(format_what_to_do_header(tiers))
    lines.append(format_tier_legend())
    lines.append("Includes moneyline + Over 1.5 props when available.")

    if blue:
        lines.append(f"BET THIS (Blue / full HA) — {len(blue)}:")
        for i, b in enumerate(blue[:5], start=1):
            lines.append(_line(b, i))
    else:
        lines.append("BET THIS (Blue) — none. Sized bankroll = $0 this card.")

    if sky:
        lines.append(
            f"TINY PAPER BET (Sky blue) — failed wide-CI HA gate, Paper only — {len(sky)}:"
        )
        for i, b in enumerate(sky[:5], start=1):
            lines.append(_line(b, i))

    if green:
        lines.append(f"FUN ONLY (Green, $0 research) — {len(green)}:")
        for i, b in enumerate(green[:5], start=1):
            lines.append(_line(b, i, fun=True))
    else:
        lines.append("FUN ONLY — no decent fun edges on this card.")

    if yellow:
        lines.append(f"CAUTION — SKIP SIZED (Yellow) — top {min(3, len(yellow))}:")
        for i, b in enumerate(yellow[:3], start=1):
            lines.append(_line(b, i, fun=True))

    return "\n".join(lines)


def _prop_market_short(b: dict[str, Any]) -> str:
    key = str(b.get("prop_key") or "")
    label = str(b.get("prop_short") or b.get("prop_type") or b.get("market") or "").strip()
    if label and label.lower() not in {"prop", "over 1.5"}:
        if "round" in label.lower() or label.lower().startswith("over") or label.lower().startswith("under"):
            return label.replace(" Rounds", "").replace(" rounds", "")
    try:
        from src.props import PROP_MARKET_LABELS

        mapped = PROP_MARKET_LABELS.get(key, "")
        if mapped:
            return mapped.replace(" Rounds", "")
    except Exception:
        pass
    return "Prop" if str(b.get("market_type") or "").lower() == "prop" else "ML"


def _event_bit(b: dict[str, Any]) -> str:
    try:
        from src.props import event_from_record

        ev = event_from_record(b)
    except Exception:
        ev = str(b.get("event") or b.get("event_name") or "").strip()
    return f"{ev} · " if ev else ""


def _line(b: dict[str, Any], rank: int, *, fun: bool = False) -> str:
    is_prop = (
        str(b.get("market_type") or "").lower() == "prop"
        or str(b.get("prop_key") or "").endswith("_rounds")
    )
    side = str(b.get("pick") or b.get("side") or "—")
    fight = str(b.get("fight") or "").strip()
    if is_prop and fight and fight not in side:
        side = f"{fight} — {side}"
    market = _prop_market_short(b) if is_prop else "ML"
    edge = b.get("edge_pct")
    if edge is None and b.get("edge") is not None:
        edge = float(b["edge"]) * 100.0
    prob = b.get("prob")
    reason = str(b.get("tier_reason") or b.get("brief") or "").replace("_", " ")
    edge_s = f"{float(edge):+.1f}%" if edge is not None else "n/a"
    prob_s = f"{float(prob):.0%}" if prob is not None else "n/a"
    action = action_label_for_bet({**b, "fun_bet": fun or b.get("fun_bet")})
    fight_s = f" | {fight}" if fight and fight not in side else ""
    return (
        f"{rank}. {action} · {_event_bit(b)}[{market}] {side}{fight_s} · edge {edge_s} · "
        f"prob {prob_s} · {reason}"
    )


def _line_compact(b: dict[str, Any], rank: int, *, fun: bool = False) -> str:
    """Short single-line pick for chat / compact briefing."""
    is_prop = (
        str(b.get("market_type") or "").lower() == "prop"
        or str(b.get("prop_key") or "").endswith("_rounds")
    )
    side = str(b.get("pick") or b.get("side") or "—")
    fight = str(b.get("fight") or "").strip()
    if is_prop and fight:
        label = f"{fight} — {side}"
    elif fight and fight not in side:
        label = f"{side} ({fight})"
    else:
        label = side
    market = _prop_market_short(b) if is_prop else "ML"
    edge = b.get("edge_pct")
    if edge is None and b.get("edge") is not None:
        edge = float(b["edge"]) * 100.0
    edge_s = f"{float(edge):+.1f}%" if edge is not None else "n/a"
    action = action_label_for_bet({**b, "fun_bet": fun or b.get("fun_bet")})
    return f"  {rank}. {action} | {_event_bit(b)}{label} | {market} | edge {edge_s}"
