"""Optional Ollama narrative analysis for UFC card edges (non-blocking, Kelly adjust)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ProgressFn = Callable[[str, float | None], None]

import config
import pandas as pd

logger = logging.getLogger(__name__)

_GROK_CACHE_DIR = config.CACHE_DIR / "grok_analysis"

# Stake fields narrative may scale — never pick/edge/prob.
_NARRATIVE_STAKE_FIELDS = (
    "kelly_stake_usd",
    "kelly_pct",
    "max_safe_bet_usd",
    "suggested_stake",
    "raw_stake",
)
# Model-owned fields — narrative must never change these.
_MODEL_LOCKED_FIELDS = (
    "pick",
    "predicted_winner",
    "edge",
    "edge_pct",
    "prob",
    "predicted_prob",
    "prob_f1_win",
    "prob_f2_win",
    "best_edge",
    "edge_f1",
    "edge_f2",
)
_EDGE_INFLATE_KEYS = (
    "edge_adjustment",
    "edge_delta",
    "revised_edge",
    "model_edge",
    "new_edge",
    "inflated_edge",
    "edge_pct_adjustment",
)


def grok_available() -> bool:
    """True when local Ollama analysis is enabled and the daemon is reachable."""
    if not bool(getattr(config, "OLLAMA_ENABLED", True)):
        return False
    try:
        from src.ollama_client import ollama_available

        return ollama_available()
    except Exception:
        return False


def llm_available() -> bool:
    """Alias for dashboard / callers — Ollama is the analysis engine."""
    return grok_available()


def clamp_kelly_factor(value: Any, *, force_one_on_invalid: bool = True) -> float:
    """Clamp narrative Kelly multiplier to profile bounds (Paper ±10%, Live tighter).

    Invalid / non-finite values → 1.0 (fail-closed) when ``force_one_on_invalid``.
    """
    try:
        lo, hi = config.narrative_kelly_bounds()
    except Exception:
        lo, hi = 0.90, 1.10
    try:
        factor = float(value)
    except (TypeError, ValueError):
        return 1.0 if force_one_on_invalid else lo
    if not math.isfinite(factor):
        return 1.0 if force_one_on_invalid else lo
    return round(max(lo, min(hi, factor)), 3)


@dataclass
class NarrativeTiltDecision:
    factor: float = 1.0
    status: str = "applied"  # applied | rejected | fail_closed
    reason: str = ""
    conviction: str = ""
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "status": self.status,
            "reason": self.reason,
            "conviction": self.conviction,
            "narrative": self.narrative,
        }


def log_narrative_tilt(
    *,
    fight: str = "",
    pick: str = "",
    factor: float = 1.0,
    status: str = "",
    reason: str = "",
    conviction: str = "",
    event: str = "",
    context: str = "sizing",
) -> None:
    """Append one tilt applied/rejected line to narrative_tilts.jsonl + logger."""
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "profile": getattr(config, "UFC_PROFILE", "paper"),
        "event": event,
        "fight": fight,
        "pick": pick,
        "factor": float(factor),
        "status": status,
        "reason": reason,
        "conviction": conviction,
        "context": context,
    }
    msg = (
        f"narrative_tilt {status} factor={factor:.3f} reason={reason} "
        f"pick={pick!r} fight={fight!r}"
    )
    if status == "applied" and abs(float(factor) - 1.0) > 1e-6:
        logger.info(msg)
    else:
        logger.info(msg)
    try:
        path = Path(getattr(config, "NARRATIVE_TILT_LOG", config.DATA_DIR / "logs" / "narrative_tilts.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:
        logger.debug("narrative tilt log write failed: %s", exc)
    try:
        from src.bet_journal import log_journal_row

        log_journal_row(
            "narrative_tilt",
            event=event,
            fight=fight,
            pick=pick,
            notes=f"status={status} factor={factor:.3f} reason={reason} conviction={conviction}",
        )
    except Exception:
        pass


def _normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _llm_asserts_pick_flip(bet: dict[str, Any], item: dict[str, Any]) -> bool:
    """True when Ollama explicitly names a different fighter/side than the model pick."""
    model_pick = _normalize_name(bet.get("pick") or bet.get("predicted_winner"))
    if not model_pick:
        return False
    for key in ("pick", "winner", "side", "recommended_pick", "bet_side"):
        raw = item.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        llm_pick = _normalize_name(raw)
        if not llm_pick:
            continue
        # Side codes
        if llm_pick in ("f1", "fighter1", "fighter_1") and model_pick == _normalize_name(
            bet.get("fighter_1") or bet.get("fighter1")
        ):
            continue
        if llm_pick in ("f2", "fighter2", "fighter_2") and model_pick == _normalize_name(
            bet.get("fighter_2") or bet.get("fighter2")
        ):
            continue
        if llm_pick in ("f1", "fighter1", "fighter_1", "f2", "fighter2", "fighter_2"):
            # Explicit opposite corner
            return True
        if llm_pick != model_pick and model_pick not in llm_pick and llm_pick not in model_pick:
            return True
    return False


def _llm_tries_inflate_edge(item: dict[str, Any]) -> bool:
    for key in _EDGE_INFLATE_KEYS:
        if key not in item:
            continue
        val = item.get(key)
        if val is None or val == "" or val == 0 or val == 1.0:
            continue
        try:
            if abs(float(val)) > 1e-9:
                return True
        except (TypeError, ValueError):
            if str(val).strip():
                return True
    return False


def resolve_narrative_tilt(
    bet: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    grok_ok: bool = True,
    context: str = "sizing",
) -> NarrativeTiltDecision:
    """
    Resolve Kelly tilt factor with model-first constraints.

    Fail-closed to 1.0 when Ollama is down, tilt disabled, low conviction,
    pick flip, edge inflation attempt, or uncertainty skip.
    Applied only to stake sizing — never mutates pick/edge.
    """
    fight = str(bet.get("fight") or bet.get("pick_line") or "")
    pick = str(bet.get("pick") or bet.get("predicted_winner") or "")
    event = str(bet.get("event") or bet.get("event_name") or "")

    def _done(factor: float, status: str, reason: str, conv: str = "", narrative: str = "") -> NarrativeTiltDecision:
        dec = NarrativeTiltDecision(
            factor=float(factor),
            status=status,
            reason=reason,
            conviction=conv,
            narrative=narrative,
        )
        log_narrative_tilt(
            fight=fight,
            pick=pick,
            factor=dec.factor,
            status=status,
            reason=reason,
            conviction=conv,
            event=event,
            context=context,
        )
        return dec

    if not bool(getattr(config, "NARRATIVE_TILT_ENABLED", True)):
        return _done(1.0, "fail_closed", "tilt_disabled")

    if not grok_ok:
        return _done(1.0, "fail_closed", "ollama_unavailable")

    # Must run after uncertainty gates — never size up skipped fights
    unc = str(bet.get("uncertainty_action") or "").strip().lower()
    if unc == "skip" or str(bet.get("skip_reason") or "").strip():
        return _done(1.0, "rejected", "after_uncertainty_skip")

    if not item:
        return _done(1.0, "fail_closed", "no_narrative_pick")

    narrative = str(
        item.get("reason") or item.get("narrative_edge") or item.get("narrative") or ""
    ).strip()
    conviction = str(item.get("conviction") or item.get("confidence") or "medium").strip().lower()

    if _llm_asserts_pick_flip(bet, item):
        return _done(1.0, "rejected", "pick_flip_rejected", conviction, narrative)

    if _llm_tries_inflate_edge(item):
        return _done(1.0, "rejected", "edge_inflate_rejected", conviction, narrative)

    if bool(getattr(config, "NARRATIVE_LOW_CONVICTION_FORCE_ONE", True)) and conviction in (
        "low",
        "weak",
        "poor",
    ):
        return _done(1.0, "rejected", "low_conviction", conviction, narrative)

    raw = item.get("kelly_adjustment", item.get("kelly_factor", 1.0))
    try:
        raw_f = float(raw)
    except (TypeError, ValueError):
        return _done(1.0, "fail_closed", "invalid_kelly_adjustment", conviction, narrative)

    if not math.isfinite(raw_f):
        return _done(1.0, "fail_closed", "invalid_kelly_adjustment", conviction, narrative)

    factor = clamp_kelly_factor(raw_f)
    if abs(factor - 1.0) < 1e-9:
        return _done(1.0, "applied", "neutral_or_clamped_to_one", conviction, narrative)
    return _done(factor, "applied", "kelly_tilt", conviction, narrative)


def _repair_json_text(text: str) -> str:
    """Fix common local-LLM JSON flaws: smart quotes, trailing commas, light truncation."""
    s = (text or "").strip()
    if not s:
        return s
    # Normalize curly/smart quotes that break json.loads
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u00a0", " ")
    )
    # Drop BOM / leading junk before first {
    start = s.find("{")
    if start > 0:
        s = s[start:]
    # Trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # If truncated mid-object, close open quotes/brackets enough to parse.
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in s:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    if in_string:
        s += '"'
    # Trim a dangling incomplete key/value after last comma if needed
    s = re.sub(r",\s*(\"[^\"]*\"\s*:)?\s*$", "", s)
    while stack:
        s += stack.pop()
    return s


def _salvage_pick_objects(text: str) -> list[dict[str, Any]]:
    """Pull individually complete pick objects even when the root JSON is truncated."""
    picks: list[dict[str, Any]] = []
    # Match objects that look like analysis picks (have narrative_edge or kelly_adjustment).
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        blob = match.group(0)
        if "kelly_adjustment" not in blob and "narrative_edge" not in blob:
            continue
        try:
            obj = json.loads(_repair_json_text(blob))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("id") or obj.get("narrative_edge")):
            picks.append(obj)
    return picks


def _extract_json_blob(text: str) -> dict[str, Any]:
    """Parse Ollama JSON; repair truncated / messy local-LLM output when needed."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Ollama returned empty content (no JSON).")

    candidates: list[str] = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    last_err: Exception | None = None
    seen: set[str] = set()
    for raw in candidates:
        for variant in (raw, _repair_json_text(raw)):
            key = variant[:2000]
            if not variant or key in seen:
                continue
            seen.add(key)
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"picks": parsed}
            except json.JSONDecodeError as exc:
                last_err = exc
                continue

    salvaged = _salvage_pick_objects(_repair_json_text(text))
    if salvaged:
        logger.warning(
            "Ollama JSON truncated/malformed; salvaged %s pick object(s)",
            len(salvaged),
        )
        return {
            "summary": "Partial analysis (model output was truncated; salvaged complete picks).",
            "picks": salvaged,
            "partial": True,
        }

    detail = f"{type(last_err).__name__}: {last_err}" if last_err else "unknown parse error"
    # Helpful hint for the classic "Expecting ',' delimiter: line N" message.
    raise ValueError(
        f"Ollama returned invalid JSON ({detail}). "
        "Usually truncated output or unescaped quotes in a narrative. "
        "Click Run Ollama Analysis again — parser will salvage partial picks when possible."
    )


def _pick_id(item: dict[str, Any]) -> str:
    for key in ("id", "fight_id", "label", "pick_line", "pick"):
        val = str(item.get(key) or "").strip()
        if val:
            return val
    return ""


def build_grok_prompt(inputs: dict[str, Any]) -> str:
    """Compact Top-N narration prompt (keep short — CPU Ollama timeouts are common)."""
    event = inputs.get("event") or "Upcoming UFC card"
    profile = str(inputs.get("profile") or "paper").upper()
    bankroll = inputs.get("bankroll")
    card_budget = inputs.get("card_budget")
    tickets = list(inputs.get("tickets") or [])[:8]
    parlays = list(inputs.get("recommended_parlays") or [])[:2]
    warning = str(inputs.get("top5_warning") or TOP5_WARNING)

    ticket_lines: list[str] = []
    try:
        from src.bet_tiers import action_label_for_bet
    except Exception:
        action_label_for_bet = None  # type: ignore[assignment]
    for t in tickets:
        is_fun = bool(t.get("advisory") or t.get("fun_bet"))
        if action_label_for_bet is not None:
            action = action_label_for_bet(t)
        else:
            action = "FUN ONLY ($0)" if is_fun else "BET THIS"
        ev = str(t.get("event") or t.get("event_name") or "").strip()
        ev_bit = f"event={ev} | " if ev else ""
        ticket_lines.append(
            f"- id={t.get('id')} | {ev_bit}ACTION={action} | {t.get('side')} | {t.get('market')} | "
            f"book={t.get('book') or 'n/a'} | stake={t.get('stake_pct')}%/${t.get('stake_usd')} | "
            f"prob={t.get('prob')} | edge={t.get('edge_pct')}% | conf={t.get('confidence')}"
            + (f" | gym={t.get('gym_note')}" if t.get("gym_note") else "")
            + (f" | photo={t.get('photo_note')}" if t.get("photo_note") else "")
            + (" | PHOTO_FADE_OVER_15" if t.get("photo_over_15_caution") else "")
        )
    tickets_block = "\n".join(ticket_lines) if ticket_lines else "- (none — NO BET / sized $0)"

    photo_notes = str(inputs.get("photo_notes") or "").strip()
    if not photo_notes:
        try:
            from src.photo_analysis import card_photo_notes

            photo_notes = card_photo_notes(tickets)
        except Exception:
            photo_notes = ""
    photo_block = f"\nPhoto desk (stills only, not HA):\n{photo_notes}\n" if photo_notes else ""

    parlay_lines: list[str] = []
    for p in parlays:
        ev = str(p.get("event") or p.get("event_name") or "").strip()
        ev_bit = f"event={ev} | " if ev else ""
        parlay_lines.append(
            f"- id={p.get('id')} | {ev_bit}FUN ONLY research $0 | {p.get('n_legs')}-leg | {p.get('picks')} | "
            f"combined_prob={float(p.get('combined_prob') or 0):.0%} | "
            f"ha_qualified={bool(p.get('ha_qualified'))}"
        )
    parlays_block = "\n".join(parlay_lines) if parlay_lines else "- (none)"

    br_txt = f"${float(bankroll):.2f}" if bankroll is not None else "n/a"
    card_txt = f"${float(card_budget):.2f}" if card_budget is not None else "n/a"
    total_pct = inputs.get("total_stake_pct")
    total_usd = inputs.get("total_stake_usd")
    n_act = inputs.get("n_actionable")
    n_adv = inputs.get("n_advisory")
    what_to_do = ""
    try:
        from src.bet_tiers import format_what_to_do_header

        what_to_do = format_what_to_do_header(slip=tickets) + "\n"
    except Exception:
        what_to_do = ""

    return f"""UFC desk. JSON only. Profile={profile} Bankroll={br_txt} Card={card_txt}
{what_to_do}{warning}
Rules: use ONLY listed tickets/parlays; copy stake_pct/stake_usd exactly; never invent odds/edge/prob;
BET THIS (Blue) = passed HA gates (live odds, min prob, min edge, uncertainty). Never call TINY PAPER BET a passed HA bet.
TINY PAPER BET (Sky) = FAILED wide CI — Paper override only, not Live HA.
In summary + each pick reason, lead with ACTION verbs: BET THIS ($), TINY PAPER BET (failed wide CI), FUN ONLY ($0), CAUTION — SKIP SIZED, or DO NOT BET.
Name the event on every pick and parlay (Next Two often has two cards).
FUN ONLY / advisory stakes stay 0 — never tell the user to size those as bankroll bets.
Parlays are research ($0) unless already BET THIS; one-line reason (<=90 chars); empty ACT list => say sized NO BET.

Tickets ({n_act} BET THIS / {n_adv} FUN ONLY):
{tickets_block}
{photo_block}BET THIS totals: {total_pct}% / ${total_usd}

Auto parlays (narrate as FUN ONLY research — do not invent legs):
{parlays_block}

Return JSON:
{{"event":"{event}","summary":"start with WHAT TO BET line and name the event(s)","picks":[{{"id":"...","event":"...","side":"...","market":"...","book":"...","stake_pct":0.0,"stake_usd":0.0,"reason":"BET THIS $x or FUN ONLY $0 — Event — ...","conviction":"high|medium|low"}}],"parlays":[{{"id":"...","event":"...","n_legs":2,"reason":"FUN ONLY $0 — Event — ...","conviction":"high|medium|low"}}]}}"""


def _ticket_slip_id(ticket: dict[str, Any]) -> str:
    return str(
        ticket.get("fight_id")
        or ticket.get("pick_line")
        or ticket.get("picks")
        or ticket.get("display_label")
        or ticket.get("fight")
        or ticket.get("id")
        or ""
    ).strip()


def _ticket_to_slip_row(
    ticket: dict[str, Any],
    *,
    rank: int,
    tier: str = "actionable",
) -> dict[str, Any]:
    """Normalize an HA-sized ticket into Ollama bet-slip row fields."""
    from src.props import PROP_MARKET_LABELS, event_from_record

    is_parlay = bool(ticket.get("is_parlay")) or int(ticket.get("n_legs") or 1) >= 2
    is_prop = (
        str(ticket.get("market_type") or "").lower() == "prop"
        or str(ticket.get("prop_key") or "").endswith("_rounds")
    )
    fight = str(ticket.get("fight") or "").strip()
    event = event_from_record(ticket)
    if is_parlay:
        n_legs = int(ticket.get("n_legs") or 2)
        market = f"{n_legs}-leg parlay"
        side = str(ticket.get("pick_line") or ticket.get("picks") or ticket.get("display_label") or "")
    elif is_prop:
        market = str(
            ticket.get("prop_short")
            or ticket.get("prop_type")
            or PROP_MARKET_LABELS.get(str(ticket.get("prop_key") or ""), "")
            or "Prop"
        )
        label = str(
            ticket.get("display_label")
            or ticket.get("label")
            or ticket.get("prop_short")
            or ticket.get("pick_line")
            or market
        )
        side = f"{fight} — {label}" if fight and fight not in label else label
    else:
        market = "moneyline"
        side = str(
            ticket.get("pick_line")
            or ticket.get("display_label")
            or ticket.get("pick")
            or ""
        )
    edge = ticket.get("edge")
    if edge is None and ticket.get("edge_pct") is not None:
        try:
            edge = float(ticket["edge_pct"]) / 100.0
        except (TypeError, ValueError):
            edge = 0.0
    try:
        edge_f = float(edge or 0.0)
    except (TypeError, ValueError):
        edge_f = 0.0
    if abs(edge_f) > 1.5:
        edge_f = edge_f / 100.0
    displayed_pct = None
    if ticket.get("edge_pct") is not None:
        try:
            displayed_pct = float(ticket["edge_pct"])
        except (TypeError, ValueError):
            displayed_pct = None
    if displayed_pct is None:
        displayed_pct = round(edge_f * 100.0, 1)
    stake_pct = float(ticket.get("stake_pct") or 0.0)
    stake_usd = float(ticket.get("suggested_stake") or ticket.get("stake_usd") or 0.0)
    conf = str(ticket.get("confidence") or ticket.get("confidence_label") or "-")
    conf_l = conf.lower()
    # Fail-closed display: zero stake if missing odds / skip uncertainty
    odds = ticket.get("decimal_odds") or ticket.get("combined_odds") or ticket.get("odds")
    unc = str(ticket.get("uncertainty_action") or "allow").strip().lower()
    if odds is None or unc in {"skip", "block", "missing", "missing_uncertainty"}:
        stake_pct = 0.0
        stake_usd = 0.0
    tier_l = str(tier or "actionable").strip().lower()
    if tier_l == "advisory":
        stake_pct = 0.0
        stake_usd = 0.0
    bet_tier = str(ticket.get("bet_tier") or "").strip().lower()
    return {
        "id": _ticket_slip_id(ticket) or f"ticket-{rank}",
        "rank": rank,
        "side": side,
        "market": market,
        "book": str(ticket.get("book") or ticket.get("book_key") or "n/a"),
        "odds_display": str(ticket.get("odds_display") or ticket.get("american_odds") or "-"),
        "stake_pct": round(stake_pct, 1),
        "stake_usd": round(stake_usd, 2),
        "prob": ticket.get("prob") or ticket.get("combined_prob"),
        "edge": edge_f,
        "edge_pct": displayed_pct,
        "confidence": conf,
        "strength_score": ticket.get("strength_score"),
        "uncertainty_action": ticket.get("uncertainty_action") or "allow",
        "event": event,
        "event_name": event,
        "fight": ticket.get("fight"),
        "pick": ticket.get("pick") or (side if is_prop else ticket.get("pick")),
        "prop_key": ticket.get("prop_key") if is_prop else "",
        "market_type": "prop" if is_prop else ("parlay" if is_parlay else "moneyline"),
        "is_parlay": is_parlay,
        "reason": "",
        "conviction": "high" if conf_l == "high" else ("low" if conf_l == "low" else "medium"),
        "tier": tier_l,
        "advisory": tier_l == "advisory",
        "bet_tier": bet_tier or None,
        "fun_bet": bool(ticket.get("fun_bet")),
        "tier_reason": ticket.get("tier_reason") or "",
        "gym_note": str(ticket.get("gym_note") or "")[:220],
        "photo_note": str(ticket.get("photo_note") or "")[:220],
        "photo_over_15_caution": bool(ticket.get("photo_over_15_caution")),
    }


TOP5_WARNING = (
    "Clarity: BET THIS = HA-sized bankroll ($). "
    "FUN ONLY = research lean ($0, not in card budget). "
    "Never treat FUN ONLY / Yellow as sized bets. "
    "If no BET THIS tickets, say sized NO BET up front."
)


def _candidate_dedupe_key(ticket: dict[str, Any]) -> str:
    """Align with bet_slip.ticket_dedupe_key: fight|market|selection|book."""
    try:
        from src.bet_slip import ticket_dedupe_key

        fight, market, selection, book = ticket_dedupe_key(ticket)
        return f"{fight}|{market}|{selection}|{book}"
    except Exception:
        base = str(
            ticket.get("fight_id")
            or ticket.get("fight")
            or ticket.get("pick_line")
            or ticket.get("display_label")
            or ""
        ).strip().lower()
        market = str(ticket.get("market_type") or ticket.get("prop_key") or "moneyline").lower()
        selection = str(ticket.get("pick") or ticket.get("side") or "").strip().lower()
        book = str(ticket.get("book") or "").strip().lower()
        return f"{base}|{market}|{selection}|{book}"


def _uncertainty_blocks_ticket(ticket: dict[str, Any]) -> bool:
    unc = str(ticket.get("uncertainty_action") or "").strip().lower()
    return unc in {"skip", "block", "missing", "missing_uncertainty"}


def _ticket_in_cleared_keys(ticket: dict[str, Any], cleared: set[str]) -> bool:
    if not cleared:
        return False
    for k in (
        str(ticket.get("fight_id") or "").strip(),
        str(ticket.get("fight") or "").strip(),
    ):
        if k and k in cleared:
            return True
    return False


def collect_card_analysis_inputs(
    books: dict[str, dict[str, Any]],
    budget_state: dict[str, Any] | None,
    *,
    event_label: str = "",
    max_fights: int | None = None,
    max_props: int | None = None,
    profile: str | None = None,
    allowed_fights: set[str] | None = None,
) -> dict[str, Any]:
    """
    Gather HA-gated + fun-tier tickets for Ollama bet-slip narration.

    Builds one merged Top N list (default 5):
    - CLEARS GATES (HA-sized) first — Blue/Sky only from real alert clears
    - then DECENT FUN / caution fillers
    - props compete in the same pool — never concatenated past Top N

    Dedupes on (fight_id, market_type, selection, book) via dedupe_rank_top_tickets.
    """
    from src.bet_slip import dedupe_rank_top_tickets
    from src.bet_tiers import (
        TIER_BLUE,
        TIER_GREEN,
        TIER_SKY_BLUE,
        TIER_YELLOW,
        demote_suspect_edge_ticket,
        demote_photo_caution_ticket,
        is_sky_blue_ticket,
        singles_cleared_keys,
    )
    from src.strategy import (
        aggregate_overview_recommendations,
        aggregate_top_recommended_bets,
        ticket_edge_exceeds_actionable_cap,
    )

    fight_cap = max_fights if max_fights is not None else config.GROK_MAX_FIGHTS
    prop_cap = max_props if max_props is not None else int(getattr(config, "GROK_MAX_PROPS", 6) or 6)
    bs = budget_state or {}
    prof = profile or config.UFC_PROFILE
    top_n = max(3, min(5, int(fight_cap)))

    slip = aggregate_overview_recommendations(
        books,
        bs,
        limit=top_n,
        max_parlays=2,
        profile=prof,
        allowed_fights=allowed_fights,
    )
    items = list(slip.get("items") or [])
    if not items:
        items = (
            list(slip.get("singles") or [])
            + list(slip.get("prop_singles") or [])
            + list(slip.get("parlays") or [])
        )

    # HA-cleared keys from book alerts (same source Overview uses for Blue)
    cleared_singles: list[dict[str, Any]] = []
    for book_data in (books or {}).values():
        if not isinstance(book_data, dict):
            continue
        cleared_singles.extend(list((book_data.get("alerts") or {}).get("singles") or []))
    cleared_keys = singles_cleared_keys(cleared_singles)

    raw_candidates: list[dict[str, Any]] = []

    # Overview tickets: Blue only when HA-cleared + sized + not uncertainty-blocked
    for t in items:
        stake_usd = float(t.get("suggested_stake") or t.get("stake_usd") or 0)
        stake_pct = float(t.get("stake_pct") or 0)
        stake_ok = stake_usd > 0 and stake_pct > 0
        ha_ok = (
            stake_ok
            and _ticket_in_cleared_keys(t, cleared_keys)
            and not _uncertainty_blocks_ticket(t)
            and not bool(t.get("advisory") or t.get("fun_bet"))
            and not bool(t.get("is_parlay"))
            and int(t.get("n_legs") or 1) < 2
            and not ticket_edge_exceeds_actionable_cap(t)
        )
        # Live Over 1.5 props: only HA-Blue if classify agrees (never invent from stake alone)
        is_prop = str(t.get("market_type") or "").lower() == "prop" or bool(t.get("prop_key"))
        if is_prop and stake_ok and not _uncertainty_blocks_ticket(t):
            from src.bet_tiers import classify_prop_bet_tier
            from src.props import is_live_prop_odds_source

            prop_tier, _ = classify_prop_bet_tier(t, debug=False)
            if (
                prop_tier in {TIER_BLUE, TIER_SKY_BLUE}
                and is_live_prop_odds_source(str(t.get("odds_source") or ""))
                and t.get("strict_qualified") is not False
                and str(t.get("prop_key") or "") == "over_1_5_rounds"
            ):
                from src.strategy import prop_may_receive_ha_stake

                ha_ok = prop_may_receive_ha_stake(t)
            else:
                ha_ok = False
        if ha_ok:
            row = _ticket_to_slip_row(t, rank=0, tier="actionable")
            if not row.get("event") and event_label:
                row["event"] = event_label
                row["event_name"] = event_label
            if is_sky_blue_ticket(
                stake_pct=stake_pct if stake_pct > 0 else None,
                stake_usd=stake_usd if stake_usd > 0 else None,
                uncertainty_reason=str(t.get("uncertainty_reason") or ""),
            ):
                row["bet_tier"] = TIER_SKY_BLUE
            else:
                row["bet_tier"] = TIER_BLUE
            row["fun_bet"] = False
            row["advisory"] = False
            raw_candidates.append(row)
        else:
            adv = _ticket_to_slip_row(t, rank=0, tier="advisory")
            if not adv.get("event") and event_label:
                adv["event"] = event_label
                adv["event_name"] = event_label
            adv["advisory"] = True
            adv["fun_bet"] = True
            adv["stake_usd"] = 0.0
            adv["stake_pct"] = 0.0
            if not adv.get("bet_tier") or adv.get("bet_tier") in {"blue", "sky_blue"}:
                adv["bet_tier"] = TIER_YELLOW
            raw_candidates.append(adv)

    # Extra ranked singles (may overlap — dedupe later); never invent Blue here
    try:
        extra = aggregate_top_recommended_bets(
            books, bs, limit=top_n, per_book_cap=2, profile=prof
        )
    except Exception:
        extra = []
    for t in list(slip.get("prop_singles") or []) + list(slip.get("parlays") or []) + list(extra):
        extra_row = _ticket_to_slip_row(t, rank=0, tier="advisory")
        if not extra_row.get("event") and event_label:
            extra_row["event"] = event_label
            extra_row["event_name"] = event_label
        extra_row["advisory"] = True
        extra_row["fun_bet"] = True
        extra_row["stake_usd"] = 0.0
        extra_row["stake_pct"] = 0.0
        if not extra_row.get("bet_tier") or extra_row.get("bet_tier") in {"blue", "sky_blue"}:
            extra_row["bet_tier"] = TIER_YELLOW
        raw_candidates.append(extra_row)

    # Skips (context only — not in top list)
    skipped_payload: list[dict[str, Any]] = []
    for book_data in (books or {}).values():
        alerts = (book_data or {}).get("alerts") or {}
        for s in alerts.get("skipped") or []:
            skipped_payload.append(
                {
                    "fight": s.get("fight"),
                    "pick": s.get("pick"),
                    "skip_reason": s.get("skip_reason"),
                    "disagreement": s.get("disagreement"),
                    "interval_width": s.get("interval_width"),
                }
            )
    seen_skip: set[str] = set()
    unique_skipped: list[dict[str, Any]] = []
    for s in skipped_payload:
        key = str(s.get("fight") or "")
        if key and key not in seen_skip:
            seen_skip.add(key)
            unique_skipped.append(s)

    pool = float(slip.get("card_pool_usd") or 0.0)
    try:
        from src.strategy import resolve_display_card_budget

        display_card, _over = resolve_display_card_budget(bs, profile=prof)
        if display_card > 0:
            pool = float(display_card)
    except Exception:
        pass

    fun_tiers: dict[str, Any] = {}
    prop_tiers: dict[str, Any] = {}
    try:
        from src.bet_tiers import (
            classify_prop_bet_tier,
            collect_props_from_books,
            merge_bet_tier_dicts,
            rank_card_bet_tiers,
            rank_prop_bet_tiers,
        )

        preds = None
        for book_data in (books or {}).values():
            if not isinstance(book_data, dict):
                continue
            if preds is None:
                p = book_data.get("predictions")
                if isinstance(p, pd.DataFrame) and not p.empty:
                    preds = p
        overview = (books or {}).get("Overview") or {}
        if preds is None and isinstance(overview.get("predictions"), pd.DataFrame):
            preds = overview.get("predictions")
        fun_tiers = rank_card_bet_tiers(preds, cleared_singles=cleared_singles, limit_per_tier=6)

        raw_props = collect_props_from_books(
            books, limit=prop_cap, allowed_fights=allowed_fights
        )
        sized_props = {
            _candidate_dedupe_key(t): t for t in list(slip.get("prop_singles") or [])
        }
        merged_props: list[dict[str, Any]] = []
        seen_props: set[str] = set()
        for t in list(slip.get("prop_singles") or []) + raw_props:
            key = _candidate_dedupe_key(t)
            if not key or key in seen_props:
                continue
            seen_props.add(key)
            merged_props.append(sized_props.get(key) or t)
        prop_tiers = rank_prop_bet_tiers(merged_props, limit_per_tier=prop_cap)
        fun_tiers = merge_bet_tier_dicts(fun_tiers, prop_tiers, limit_per_tier=8)

        # Inject HA Blue/Sky from tier ranker (authoritative — matches Overview colors)
        for tier_name in (TIER_BLUE, TIER_SKY_BLUE):
            for b in fun_tiers.get(tier_name) or []:
                row = _ticket_to_slip_row(b, rank=0, tier="actionable")
                if not row.get("event") and event_label:
                    row["event"] = event_label
                    row["event_name"] = event_label
                row["bet_tier"] = tier_name
                row["fun_bet"] = False
                row["advisory"] = False
                # Keep stakes only when the tier source actually sized them
                if float(row.get("stake_usd") or 0) <= 0 and float(b.get("suggested_stake") or b.get("stake_usd") or 0) > 0:
                    row["stake_usd"] = float(b.get("suggested_stake") or b.get("stake_usd") or 0)
                    row["stake_pct"] = float(b.get("stake_pct") or 0)
                raw_candidates.append(row)

        # Fun ML + prop candidates compete in the SAME pool (merged below)
        for fun in list(fun_tiers.get(TIER_GREEN) or []) + list(fun_tiers.get(TIER_YELLOW) or []):
            row = _ticket_to_slip_row(fun, rank=0, tier="advisory")
            if not row.get("event") and event_label:
                row["event"] = event_label
                row["event_name"] = event_label
            row["bet_tier"] = fun.get("bet_tier") or TIER_GREEN
            row["fun_bet"] = True
            row["advisory"] = True
            row["stake_usd"] = 0.0
            row["stake_pct"] = 0.0
            row["reason"] = fun.get("brief") or row.get("reason") or "Fun tier (not HA-sized)"
            raw_candidates.append(row)

        for tier_name in (TIER_BLUE, TIER_SKY_BLUE, TIER_GREEN, TIER_YELLOW):
            for p in prop_tiers.get(tier_name) or []:
                key = _candidate_dedupe_key(p)
                src = sized_props.get(key) or p
                stake_ok = float(src.get("suggested_stake") or src.get("stake_usd") or 0) > 0 and float(
                    src.get("stake_pct") or 0
                ) > 0
                # Props: Blue only when tier ranker said blue/sky AND live HA-eligible stake
                from src.strategy import prop_may_receive_ha_stake

                prop_ha = (
                    tier_name in {TIER_BLUE, TIER_SKY_BLUE}
                    and stake_ok
                    and prop_may_receive_ha_stake(src)
                )
                row = _ticket_to_slip_row(
                    src,
                    rank=0,
                    tier="actionable" if prop_ha else "advisory",
                )
                row["bet_tier"] = p.get("bet_tier") or tier_name
                row["tier_reason"] = p.get("tier_reason") or row.get("tier_reason") or ""
                if prop_ha:
                    row["fun_bet"] = False
                    row["advisory"] = False
                    if is_sky_blue_ticket(
                        stake_pct=float(src.get("stake_pct") or 0) or None,
                        stake_usd=float(src.get("suggested_stake") or src.get("stake_usd") or 0)
                        or None,
                        uncertainty_reason=str(src.get("uncertainty_reason") or ""),
                    ):
                        row["bet_tier"] = TIER_SKY_BLUE
                    else:
                        row["bet_tier"] = TIER_BLUE
                else:
                    row["stake_usd"] = 0.0
                    row["stake_pct"] = 0.0
                    row["advisory"] = True
                    row["fun_bet"] = True
                    if row.get("bet_tier") in {TIER_BLUE, TIER_SKY_BLUE}:
                        # Downgrade — no size means not Blue
                        try:
                            tier, reason = classify_prop_bet_tier(src, debug=False)
                            row["bet_tier"] = tier if tier not in {TIER_BLUE, TIER_SKY_BLUE} else TIER_YELLOW
                            row["tier_reason"] = reason
                        except Exception:
                            row["bet_tier"] = TIER_YELLOW
                if not row.get("event") and event_label:
                    row["event"] = event_label
                    row["event_name"] = event_label
                raw_candidates.append(row)
    except Exception as exc:
        logger.debug("fun/prop tier build skipped: %s", exc)

    # Annotate missing bet_tier — never invent Blue from stake
    for t in raw_candidates:
        if t.get("bet_tier"):
            continue
        if str(t.get("market_type") or "") == "prop" or "Over 1.5" in str(t.get("market") or ""):
            try:
                from src.bet_tiers import classify_prop_bet_tier

                tier, reason = classify_prop_bet_tier(t, debug=False)
                if tier in {TIER_BLUE, TIER_SKY_BLUE} and float(t.get("stake_usd") or 0) <= 0:
                    tier = TIER_YELLOW
                t["bet_tier"] = tier
                t["tier_reason"] = reason
            except Exception:
                t["bet_tier"] = TIER_YELLOW
        else:
            t["bet_tier"] = TIER_YELLOW
        if t.get("bet_tier") not in {TIER_BLUE, TIER_SKY_BLUE}:
            t["advisory"] = True
            t["fun_bet"] = True
            t["stake_usd"] = 0.0
            t["stake_pct"] = 0.0

    raw_candidates = [demote_suspect_edge_ticket(t) or t for t in raw_candidates]
    raw_candidates = [demote_photo_caution_ticket(t) or t for t in raw_candidates]

    tickets = dedupe_rank_top_tickets(
        raw_candidates,
        limit=top_n,
        event=event_label,
        log=logger,
    )

    actionable_only = [
        t
        for t in tickets
        if str(t.get("bet_tier") or "") in {TIER_BLUE, TIER_SKY_BLUE}
        and not t.get("advisory")
        and not t.get("fun_bet")
        and float(t.get("stake_usd") or 0) > 0
    ]
    total_pct = round(sum(float(t.get("stake_pct") or 0) for t in actionable_only), 1)
    total_usd = round(sum(float(t.get("stake_usd") or 0) for t in actionable_only), 2)
    br = float(bs.get("total_bankroll") or config.DEFAULT_TOTAL_BANKROLL)

    fights = [t for t in tickets if t.get("market") == "moneyline"]
    props = [
        t
        for t in tickets
        if "Over 1.5" in str(t.get("market") or "")
        or str(t.get("market_type") or "") == "prop"
    ]
    n_actionable = len(actionable_only)
    n_advisory = sum(1 for t in tickets if t.get("advisory") or t.get("fun_bet"))
    photo_notes = ""
    try:
        from src.photo_analysis import card_photo_notes

        photo_notes = card_photo_notes(tickets)
    except Exception:
        photo_notes = ""

    recommended_parlays: list[dict[str, Any]] = []
    try:
        from src.strategy import build_auto_parlay_recommendations

        preds_df = _predictions_frame_from_books(books)
        if allowed_fights and preds_df is not None and not preds_df.empty:
            # Soft-filter to loaded card fights when keys are available
            try:
                mask = pd.Series(False, index=preds_df.index)
                if "fight_id" in preds_df.columns:
                    mask = mask | preds_df["fight_id"].astype(str).isin(allowed_fights)
                if "fighter_1" in preds_df.columns and "fighter_2" in preds_df.columns:
                    labels = (
                        preds_df["fighter_1"].astype(str)
                        + " vs "
                        + preds_df["fighter_2"].astype(str)
                    )
                    mask = mask | labels.isin(allowed_fights)
                if bool(mask.any()):
                    preds_df = preds_df.loc[mask].copy()
            except Exception:
                pass
        recommended_parlays = build_auto_parlay_recommendations(
            preds_df,
            ha_singles=_ha_singles_from_books(books),
        )
        recommended_parlays = merge_ollama_reasons_into_parlays(recommended_parlays, [])
    except Exception as exc:
        logger.debug("auto parlay recommendations skipped: %s", exc)
        recommended_parlays = []

    return {
        "event": event_label,
        "profile": config.normalize_profile(prof),
        "bankroll": br,
        "card_budget": pool,
        "tickets": tickets,
        "fights": fights,
        "props": props,
        "skipped": unique_skipped[:12],
        "total_stake_pct": total_pct,
        "total_stake_usd": total_usd,
        "no_bet": n_actionable == 0,
        "stake_allocation": slip.get("stake_allocation") or {},
        "top5_warning": TOP5_WARNING,
        "n_actionable": n_actionable,
        "n_advisory": n_advisory,
        "fun_tiers": fun_tiers,
        "prop_tiers": prop_tiers,
        "top_n": top_n,
        "top_n_shown": len(tickets),
        "recommended_parlays": recommended_parlays,
        "photo_notes": photo_notes,
    }


def books_have_usable_odds(books: dict[str, dict[str, Any]] | None) -> bool:
    """True when any book has matched / usable decimal odds (> 1)."""
    if not books:
        return False
    for book_data in books.values():
        if not isinstance(book_data, dict):
            continue
        try:
            if int(book_data.get("odds_matched") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        preds = book_data.get("predictions")
        try:
            import pandas as pd

            if isinstance(preds, pd.DataFrame) and not preds.empty:
                if "odds_matched" in preds.columns and bool(
                    preds["odds_matched"].fillna(False).astype(bool).any()
                ):
                    return True
                for col in ("f1_odds", "f2_odds", "decimal_odds", "best_odds"):
                    if col in preds.columns:
                        series = pd.to_numeric(preds[col], errors="coerce")
                        if (series.fillna(0) > 1.0).any():
                            return True
        except Exception:
            pass
        alerts = book_data.get("alerts") or {}
        for group in ("singles", "parlays", "props"):
            for row in alerts.get(group) or []:
                if not isinstance(row, dict):
                    continue
                for key in ("decimal_odds", "odds", "combined_odds"):
                    try:
                        if float(row.get(key) or 0) > 1.0:
                            return True
                    except (TypeError, ValueError):
                        continue
    return False


def merge_ollama_reasons_into_slip(
    tickets: list[dict[str, Any]],
    ollama_picks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach Ollama one-line reasons onto HA-sized tickets (stakes unchanged)."""
    by_id = {str(p.get("id") or "").strip(): p for p in ollama_picks if p.get("id")}
    out: list[dict[str, Any]] = []
    for t in tickets:
        row = dict(t)
        match = by_id.get(str(t.get("id") or "").strip())
        if match:
            reason = str(match.get("reason") or match.get("narrative_edge") or "").strip()
            reason = " ".join(reason.replace("\n", " ").split())
            if len(reason) > 120:
                reason = reason[:119].rstrip() + "…"
            if reason:
                row["reason"] = reason
            conv = str(match.get("conviction") or "").lower()
            if conv in {"high", "medium", "low"}:
                row["conviction"] = conv
            ev = str(match.get("event") or match.get("event_name") or "").strip()
            if ev and not str(row.get("event") or row.get("event_name") or "").strip():
                row["event"] = ev
                row["event_name"] = ev
        if not row.get("reason"):
            # Deterministic fallback reason from model fields
            conf = str(row.get("confidence") or "-")
            edge_pct = row.get("edge_pct")
            prefix = "ADVISORY: " if row.get("advisory") else ""
            row["reason"] = f"{prefix}{conf} conf, edge {edge_pct}%"
        elif row.get("advisory") and not str(row.get("reason") or "").upper().startswith("ADVISORY"):
            row["reason"] = f"ADVISORY: {row['reason']}"
        # Never let Ollama inflate advisory stakes
        if row.get("advisory"):
            row["stake_pct"] = 0.0
            row["stake_usd"] = 0.0
            row["tier"] = "advisory"
        out.append(row)
    return out


def merge_ollama_reasons_into_parlays(
    parlays: list[dict[str, Any]],
    ollama_parlays: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach Ollama one-line reasons onto auto 2/3-leg parlay recs ($0 research)."""
    by_id = {
        str(p.get("id") or "").strip(): p
        for p in (ollama_parlays or [])
        if isinstance(p, dict) and p.get("id")
    }
    by_legs: dict[int, dict[str, Any]] = {}
    for p in ollama_parlays or []:
        if not isinstance(p, dict):
            continue
        n = int(p.get("n_legs") or 0)
        if n >= 2 and n not in by_legs:
            by_legs[n] = p
    out: list[dict[str, Any]] = []
    for p in parlays or []:
        row = dict(p)
        match = by_id.get(str(row.get("id") or "").strip())
        if match is None:
            match = by_legs.get(int(row.get("n_legs") or 0))
        if match:
            reason = str(match.get("reason") or match.get("narrative_edge") or "").strip()
            reason = " ".join(reason.replace("\n", " ").split())
            if len(reason) > 120:
                reason = reason[:119].rstrip() + "…"
            if reason:
                row["reason"] = reason
            conv = str(match.get("conviction") or "").lower()
            if conv in {"high", "medium", "low"}:
                row["conviction"] = conv
            ev = str(match.get("event") or match.get("event_name") or "").strip()
            if ev and not str(row.get("event") or row.get("event_name") or "").strip():
                row["event"] = ev
                row["event_name"] = ev
        if not row.get("reason"):
            comb = float(row.get("combined_prob") or 0)
            tag = "HA legs" if row.get("ha_qualified") else "research"
            row["reason"] = (
                f"ADVISORY: auto {row.get('n_legs')}-leg · {comb:.0%} combined · {tag}"
            )
        elif not str(row.get("reason") or "").upper().startswith("ADVISORY"):
            row["reason"] = f"ADVISORY: {row['reason']}"
        row["advisory"] = True
        row["suggested_stake"] = 0.0
        row["stake_usd"] = 0.0
        row["stake_pct"] = 0.0
        out.append(row)
    return out


def _predictions_frame_from_books(books: dict[str, dict[str, Any]] | None):
    """Best available card predictions DataFrame across books."""
    import pandas as pd

    if not books:
        return None
    preferred = ("Overview", "Odds API", "MyBookie", "Consensus")
    order = list(dict.fromkeys((*preferred, *tuple(books))))
    for book in order:
        data = books.get(book) or {}
        if not isinstance(data, dict):
            continue
        preds = data.get("predictions")
        if isinstance(preds, pd.DataFrame) and not preds.empty:
            return preds
    return None


def _ha_singles_from_books(books: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for data in (books or {}).values():
        if not isinstance(data, dict):
            continue
        alerts = data.get("alerts") or {}
        out.extend(list(alerts.get("singles") or []))
    return out


def normalize_grok_result(raw: dict[str, Any], *, event_label: str = "") -> dict[str, Any]:
    """Normalize Grok/Ollama JSON into dashboard-friendly structure."""
    picks_in = raw.get("picks") or raw.get("analyses") or []
    if not isinstance(picks_in, list):
        picks_in = []

    picks: list[dict[str, Any]] = []
    for row in picks_in:
        if not isinstance(row, dict):
            continue
        pid = _pick_id(row)
        risks = row.get("invalidation_risks") or row.get("risks") or []
        if isinstance(risks, str):
            risks = [r.strip() for r in risks.split(";") if r.strip()]
        narrative = str(
            row.get("reason")
            or row.get("narrative_edge")
            or row.get("narrative")
            or ""
        ).strip()
        narrative = " ".join(narrative.replace("\n", " ").split())
        if len(narrative) > 120:
            narrative = narrative[:119].rstrip() + "…"
        picks.append(
            {
                "id": pid,
                "pick_type": str(row.get("pick_type") or row.get("market") or row.get("type") or "moneyline"),
                "side": str(row.get("side") or row.get("pick") or "").strip(),
                "market": str(row.get("market") or row.get("pick_type") or "moneyline").strip(),
                "book": str(row.get("book") or "").strip(),
                "stake_pct": row.get("stake_pct"),
                "stake_usd": row.get("stake_usd"),
                "reason": narrative,
                "narrative_edge": narrative,
                "event": str(row.get("event") or row.get("event_name") or event_label or "").strip(),
                "crowd_positioning": str(
                    row.get("crowd_positioning") or row.get("crowd") or ""
                ).strip()[:80],
                "invalidation_risks": [str(r).strip() for r in risks if str(r).strip()][:2],
                "kelly_adjustment": clamp_kelly_factor(
                    row.get("kelly_adjustment") or row.get("kelly_factor") or 1.0
                ),
                "conviction": str(row.get("conviction") or row.get("confidence") or "medium").lower(),
                "pick": str(row.get("pick") or row.get("winner") or row.get("recommended_pick") or "").strip(),
            }
        )
        for bad in _EDGE_INFLATE_KEYS:
            picks[-1].pop(bad, None)

    summary = str(raw.get("summary") or raw.get("card_summary") or "").strip()
    summary = " ".join(summary.replace("\n", " ").split())
    if len(summary) > 160:
        summary = summary[:159].rstrip() + "…"

    parlays_in = raw.get("parlays") or []
    if not isinstance(parlays_in, list):
        parlays_in = []
    ollama_parlays: list[dict[str, Any]] = []
    for row in parlays_in:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or row.get("narrative_edge") or "").strip()
        reason = " ".join(reason.replace("\n", " ").split())
        if len(reason) > 120:
            reason = reason[:119].rstrip() + "…"
        n_raw = row.get("n_legs")
        try:
            n_legs = int(n_raw) if n_raw is not None else None
        except (TypeError, ValueError):
            n_legs = None
        ollama_parlays.append(
            {
                "id": str(row.get("id") or "").strip(),
                "n_legs": n_legs,
                "reason": reason,
                "conviction": str(row.get("conviction") or "medium").lower(),
                "event": str(row.get("event") or row.get("event_name") or event_label or "").strip(),
            }
        )

    return {
        "event": str(raw.get("event") or event_label or ""),
        "summary": summary,
        "picks": picks,
        "ollama_parlays": ollama_parlays,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "ollama",
    }


def _cache_key(inputs: dict[str, Any]) -> str:
    blob = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _load_cache(key: str) -> dict[str, Any] | None:
    path = _GROK_CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = data.get("cached_at")
        if cached_at:
            ts = datetime.fromisoformat(str(cached_at).replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
            if age_h > config.GROK_CACHE_TTL_HOURS:
                return None
        return data.get("result")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _save_cache(key: str, result: dict[str, Any]) -> None:
    try:
        _GROK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        (_GROK_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("Grok cache write failed: %s", exc)


def query_ollama(
    prompt: str,
    *,
    timeout_sec: int | None = None,
) -> tuple[str, str]:
    """Call local Ollama. Returns (model_used, response_text). Never calls xAI."""
    from src.ollama_client import ollama_available, ollama_complete, ollama_status_message

    if not bool(getattr(config, "OLLAMA_ENABLED", True)):
        raise RuntimeError("Ollama analysis disabled (OLLAMA_ENABLED=false).")
    if not ollama_available():
        raise RuntimeError(ollama_status_message())

    system = (
        "You are a concise UFC betting analyst. "
        "Respond with valid JSON only — no commentary outside the JSON object."
    )
    # Card narrate is short JSON — don't burn the full 600s budget on CPU.
    configured = int(getattr(config, "OLLAMA_TIMEOUT_SEC", 600) or 600)
    effective = int(timeout_sec) if timeout_sec is not None else min(configured, 180)
    effective = max(45, effective)
    model_used, text = ollama_complete(
        prompt,
        system=system,
        timeout_sec=effective,
        json_mode=True,
        temperature=0.2,
    )
    logger.info("Ollama analysis complete model=%s chars=%s", model_used, len(text or ""))
    if not str(text or "").strip():
        raise RuntimeError("Ollama returned empty content.")
    return model_used, str(text)


def query_grok(prompt: str) -> str:
    """
    Legacy name — routes to local Ollama (xAI disabled).

    Returns response text only for older callers.
    """
    _model, text = query_ollama(prompt)
    return text


def analyze_card_with_grok(
    books: dict[str, dict[str, Any]],
    budget_state: dict[str, Any] | None,
    *,
    event_label: str = "",
    use_cache: bool = True,
    profile: str | None = None,
    allowed_fights: set[str] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Run local Ollama narration on HA-sized tickets (primary bet slip).

    Stakes come from conf/odds sizing — Ollama only supplies one-line reasons.
    Fail-closed: never invents bets; when Ollama is offline/timeout/model-missing,
    still returns the HA-sized slip with a clear health banner.
    """
    import time

    from src.ollama_client import (
        check_ollama_health,
        classify_ollama_error,
        ollama_status_message,
    )

    t0 = time.perf_counter()

    def _latency_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    def _prog(msg: str, pct: float | None = None) -> None:
        if progress:
            progress(msg, pct)

    _prog("Collecting HA tickets...", 0.08)
    inputs = collect_card_analysis_inputs(
        books,
        budget_state,
        event_label=event_label,
        profile=profile,
        allowed_fights=allowed_fights,
    )
    tickets = list(inputs.get("tickets") or [])
    skipped = list(inputs.get("skipped") or [])
    no_odds = not books_have_usable_odds(books)
    no_bet_reason = ""
    if no_odds:
        no_bet_reason = "NO BET — no usable odds (fail-closed)"
    elif not tickets:
        no_bet_reason = "NO BET — nothing cleared HA gates for this card."
    elif inputs.get("no_bet"):
        # Top 5 may still include advisory research rows with $0
        no_bet_reason = ""

    def _base_result(**extra: Any) -> dict[str, Any]:
        slip = merge_ollama_reasons_into_slip(tickets, []) if tickets else []
        parlays = merge_ollama_reasons_into_parlays(
            list(inputs.get("recommended_parlays") or []),
            [],
        )
        out: dict[str, Any] = {
            "event": event_label or inputs.get("event") or "",
            "picks": [],
            "bet_slip": slip,
            "recommended_parlays": parlays,
            "skipped": skipped,
            "total_stake_pct": inputs.get("total_stake_pct") if slip else 0.0,
            "total_stake_usd": inputs.get("total_stake_usd") if slip else 0.0,
            "card_budget": inputs.get("card_budget"),
            "bankroll": inputs.get("bankroll"),
            "profile": inputs.get("profile"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source": "ha_slip",
            "from_cache": False,
            "latency_ms": _latency_ms(),
            "no_bet": not bool([t for t in slip if not t.get("advisory")]),
            "no_usable_odds": no_odds,
            "top5_warning": inputs.get("top5_warning") or TOP5_WARNING,
            "n_actionable": inputs.get("n_actionable"),
            "n_advisory": inputs.get("n_advisory"),
            "fun_tiers": inputs.get("fun_tiers") or {},
            "prop_tiers": inputs.get("prop_tiers") or {},
            "props": inputs.get("props") or [],
        }
        out.update(extra)
        return out

    _prog("Checking Ollama...", 0.18)
    if not bool(getattr(config, "OLLAMA_ENABLED", True)):
        _prog("Ollama disabled", 1.0)
        logger.warning(
            "Ollama health error_class=disabled latency_ms=%s — showing model tickets only",
            _latency_ms(),
        )
        return _base_result(
            ok=True,
            warning="Ollama disabled — set OLLAMA_ENABLED=true in .env",
            error_class="ok",
            health_banner="Ollama disabled — showing HA tickets",
            summary=no_bet_reason
            or "HA bet slip ready — Ollama narrative skipped.",
            model=getattr(config, "OLLAMA_MODEL", "ollama"),
            narrative_degraded=True,
            ollama_error_class="disabled",
        )

    health = check_ollama_health(force=True)
    health_class = str(health.get("error_class") or "other")
    if health_class in {"offline", "model_missing"} or not health.get("reachable"):
        _prog("Ollama offline — HA tickets only", 1.0)
        banner = "Ollama offline — showing HA tickets"
        if health_class == "model_missing":
            banner = "Ollama model missing — showing HA tickets"
        logger.warning(
            "Ollama health error_class=%s latency_ms=%s — %s",
            health_class,
            health.get("latency_ms"),
            banner,
        )
        return _base_result(
            ok=True,
            warning=str(health.get("error") or ollama_status_message())[:240],
            error_class="ok",
            health_banner=banner,
            ollama_latency_ms=health.get("latency_ms"),
            summary=no_bet_reason
            or "HA bet slip ready — Ollama narrative skipped.",
            model=health.get("resolved_model")
            or getattr(config, "OLLAMA_MODEL", "ollama"),
            narrative_degraded=True,
            ollama_error_class=health_class if health_class != "ok" else "offline",
        )

    # Explicit NO BET when no odds or nothing at all to show
    if no_odds or not tickets:
        _prog("No sized tickets for this card", 1.0)
        return {
            "ok": True,
            "no_bet": True,
            "no_usable_odds": no_odds,
            "event": event_label or inputs.get("event") or "",
            "summary": no_bet_reason or "NO BET — nothing cleared HA gates for this card.",
            "picks": [],
            "bet_slip": [],
            "recommended_parlays": merge_ollama_reasons_into_parlays(
                list(inputs.get("recommended_parlays") or []),
                [],
            ),
            "skipped": skipped,
            "total_stake_pct": 0.0,
            "total_stake_usd": 0.0,
            "card_budget": inputs.get("card_budget"),
            "bankroll": inputs.get("bankroll"),
            "profile": inputs.get("profile"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source": "ollama",
            "model": getattr(config, "OLLAMA_MODEL", "ollama"),
            "from_cache": False,
            "latency_ms": _latency_ms(),
            "error_class": "ok",
            "health_banner": health.get("banner"),
            "top5_warning": inputs.get("top5_warning") or TOP5_WARNING,
            "n_actionable": 0,
            "n_advisory": 0,
        }

    cache_key = _cache_key(inputs)
    if use_cache:
        _prog("Checking narrative cache...", 0.28)
        cached = _load_cache(cache_key)
        if cached:
            _prog("Loaded cached narrative", 1.0)
            out = dict(cached)
            out["from_cache"] = True
            out["ok"] = True
            out["bet_slip"] = merge_ollama_reasons_into_slip(
                tickets, out.get("picks") or []
            )
            out["recommended_parlays"] = merge_ollama_reasons_into_parlays(
                list(inputs.get("recommended_parlays") or []),
                out.get("ollama_parlays") or out.get("parlays") or [],
            )
            out["skipped"] = skipped
            out["total_stake_pct"] = inputs.get("total_stake_pct")
            out["total_stake_usd"] = inputs.get("total_stake_usd")
            out["card_budget"] = inputs.get("card_budget")
            out["bankroll"] = inputs.get("bankroll")
            out["profile"] = inputs.get("profile")
            out["no_bet"] = bool(inputs.get("no_bet"))
            out["no_usable_odds"] = no_odds
            out["latency_ms"] = _latency_ms()
            out["error_class"] = "ok"
            out["health_banner"] = health.get("banner")
            out["top5_warning"] = inputs.get("top5_warning") or TOP5_WARNING
            out["n_actionable"] = inputs.get("n_actionable")
            out["n_advisory"] = inputs.get("n_advisory")
            return out

    _prog("Building prompt...", 0.35)
    prompt = build_grok_prompt(inputs)
    try:
        _prog("Ollama is thinking...", 0.45)
        model_used, raw_text = query_ollama(prompt)
        _prog("Parsing response...", 0.88)
        try:
            parsed = _extract_json_blob(raw_text)
        except ValueError as parse_exc:
            # Do NOT spend another full Ollama call on repair — salvage or degrade.
            logger.warning("Ollama JSON parse failed (%s); using salvage/HA reasons", parse_exc)
            salvaged = _salvage_pick_objects(_repair_json_text(raw_text))
            if salvaged:
                parsed = {
                    "summary": "Partial narrative (salvaged picks).",
                    "picks": salvaged,
                    "partial": True,
                }
            else:
                raise
        result = normalize_grok_result(parsed, event_label=event_label or inputs.get("event", ""))
        result["ok"] = True
        result["from_cache"] = False
        result["model"] = model_used
        result["source"] = "ollama"
        result["no_bet"] = bool(inputs.get("no_bet"))
        result["no_usable_odds"] = no_odds
        result["bet_slip"] = merge_ollama_reasons_into_slip(tickets, result.get("picks") or [])
        result["recommended_parlays"] = merge_ollama_reasons_into_parlays(
            list(inputs.get("recommended_parlays") or []),
            result.get("ollama_parlays") or [],
        )
        result["skipped"] = skipped
        result["total_stake_pct"] = inputs.get("total_stake_pct")
        result["total_stake_usd"] = inputs.get("total_stake_usd")
        result["card_budget"] = inputs.get("card_budget")
        result["bankroll"] = inputs.get("bankroll")
        result["profile"] = inputs.get("profile")
        result["top5_warning"] = inputs.get("top5_warning") or TOP5_WARNING
        result["n_actionable"] = inputs.get("n_actionable")
        result["n_advisory"] = inputs.get("n_advisory")
        result["fun_tiers"] = inputs.get("fun_tiers") or {}
        result["prop_tiers"] = inputs.get("prop_tiers") or {}
        result["props"] = inputs.get("props") or []
        result["latency_ms"] = _latency_ms()
        result["error_class"] = "ok"
        result["health_banner"] = health.get("banner")
        result["ollama_latency_ms"] = _latency_ms()
        try:
            n = len(result["bet_slip"])
            if not result.get("summary"):
                n_act = int(inputs.get("n_actionable") or 0)
                n_adv = int(inputs.get("n_advisory") or 0)
                result["summary"] = (
                    f"Top {n}: {n_act} actionable (HA sized), {n_adv} advisory ($0)."
                )
            if parsed.get("partial"):
                result["partial"] = True
            _save_cache(cache_key, result)
            logger.info(
                "Ollama analysis ok error_class=ok latency_ms=%s model=%s tickets=%s "
                "actionable=%s advisory=%s",
                result["latency_ms"],
                model_used,
                n,
                inputs.get("n_actionable"),
                inputs.get("n_advisory"),
            )
        except Exception:
            pass
        _prog("Analysis complete", 1.0)
        return result
    except Exception as exc:
        _prog("Narrative unavailable — HA tickets only", 1.0)
        err_class = classify_ollama_error(exc)
        latency = _latency_ms()
        logger.warning(
            "Ollama narrative unavailable error_class=%s latency_ms=%s: %s — "
            "returning HA slip with deterministic reasons",
            err_class,
            latency,
            exc,
        )
        # Never hard-fail the tab when HA tickets exist — narrative is optional.
        slip = merge_ollama_reasons_into_slip(tickets, [])
        n_act = int(inputs.get("n_actionable") or 0)
        n_adv = int(inputs.get("n_advisory") or 0)
        summary = no_bet_reason or (
            f"Top {len(slip)}: {n_act} actionable, {n_adv} advisory "
            f"(HA slip — Ollama narrative skipped: {err_class})."
        )
        banner = {
            "timeout": "Ollama slow — showing HA tickets (model reasons)",
            "offline": "Ollama offline — showing HA tickets",
            "model_missing": "Ollama model missing — showing HA tickets",
            "disabled": "Ollama disabled — showing HA tickets",
        }.get(err_class, "Ollama narrative skipped — showing HA tickets")
        return _base_result(
            ok=True,
            error=None,
            warning=str(exc)[:240],
            error_class="ok",
            health_banner=banner,
            summary=summary,
            model=getattr(config, "OLLAMA_MODEL", "ollama"),
            source="ha_slip",
            narrative_degraded=True,
            ollama_error_class=err_class,
            bet_slip=slip,
        )


analyze_card_with_ollama = analyze_card_with_grok


def build_best_bets_briefing(
    result: dict[str, Any] | None,
    *,
    predictions: Any = None,
    cleared_singles: list[dict[str, Any]] | None = None,
    compact: bool = False,
) -> str:
    """Best-bet briefing with Blue/Green/Yellow/Red — HA gates unchanged for Blue."""
    from src.bet_tiers import (
        format_tiered_best_bets,
        rank_card_bet_tiers,
    )

    event = ""
    slip: list[dict[str, Any]] = []
    if isinstance(result, dict):
        event = str(result.get("event") or "").strip()
        slip = list(result.get("bet_slip") or [])
        if result.get("fun_tiers"):
            return format_tiered_best_bets(
                result["fun_tiers"], event=event, compact=compact
            )

    singles = list(cleared_singles or [])
    if not singles and slip:
        # Only HA-tagged Blue/Sky rows — never promote bare stake leftovers to Blue
        from src.bet_tiers import TIER_BLUE, TIER_SKY_BLUE

        singles = [
            b
            for b in slip
            if str(b.get("bet_tier") or "") in {TIER_BLUE, TIER_SKY_BLUE}
            and not b.get("advisory")
            and not b.get("fun_bet")
            and float(b.get("stake_usd") or b.get("suggested_stake") or 0) > 0
        ]

    preds = predictions
    if preds is None and isinstance(result, dict):
        preds = result.get("predictions")

    try:
        import pandas as pd

        if preds is not None and not isinstance(preds, pd.DataFrame):
            preds = None
    except Exception:
        preds = None

    tiers = rank_card_bet_tiers(preds, cleared_singles=singles, limit_per_tier=6)
    # Do NOT invent Blue from slip leftovers — if HA cleared none, Blue stays empty.

    if not isinstance(result, dict) and preds is None and not singles:
        return (
            "No card analysis loaded yet. Run Refresh Next Two, then Run Ollama Analysis "
            "(or ask again after the slip appears)."
        )

    return format_tiered_best_bets(tiers, event=event or "Current card", compact=compact)


def _analysis_context_for_chat(result: dict[str, Any] | None) -> str:
    """Compact grounded context for follow-up Ollama Q&A."""
    briefing = build_best_bets_briefing(result)
    if not isinstance(result, dict):
        return briefing
    slip = list(result.get("bet_slip") or [])
    extra: list[str] = []
    for b in slip[:8]:
        reason = str(b.get("reason") or b.get("ollama_reason") or "").strip()
        if reason:
            side = str(b.get("side") or b.get("pick") or "—")
            extra.append(f"- {side}: {reason[:180]}")
    if extra:
        return briefing + "\n\nTicket reasons:\n" + "\n".join(extra)
    return briefing


_STOP_CHAT_TOKENS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "vs",
        "versus",
        "fight",
        "fights",
        "fighter",
        "stats",
        "stat",
        "tell",
        "me",
        "about",
        "on",
        "for",
        "what",
        "are",
        "is",
        "how",
        "much",
        "edge",
        "odds",
        "prob",
        "probability",
        "model",
        "card",
        "bet",
        "bets",
        "please",
        "show",
        "give",
        "info",
        "information",
        "breakdown",
        "analysis",
        "ollama",
        "this",
        "that",
        "with",
        "from",
        "into",
        "over",
        "under",
        "round",
        "rounds",
    }
)


def _chat_query_tokens(question: str) -> list[str]:
    raw = re.sub(r"[^a-z0-9\s]", " ", str(question or "").lower())
    return [t for t in raw.split() if len(t) >= 3 and t not in _STOP_CHAT_TOKENS]


def _token_hits_name(token: str, name: str) -> bool:
    from src.predictor import _fighter_name_key, _names_match

    if not token or not name:
        return False
    if _names_match(token, name):
        return True
    key = _fighter_name_key(name)
    parts = [p for p in key.split() if p and p not in {"jr", "sr", "ii", "iii", "iv"}]
    if token in parts:
        return True
    # Last-name / partial (e.g. "mcconico" in "eric mcconico")
    return any(token in p or p in token for p in parts if len(token) >= 4 and len(p) >= 4)


def _fight_label_from_row(row: dict[str, Any] | pd.Series) -> str:
    fight = str(row.get("fight") or row.get("pick_line") or "").strip()
    if fight:
        return fight
    f1 = str(row.get("fighter_1") or row.get("fighter1") or "").strip()
    f2 = str(row.get("fighter_2") or row.get("fighter2") or "").strip()
    if f1 and f2:
        return f"{f1} vs {f2}"
    return f1 or f2 or "Unknown fight"


def _iter_card_fight_rows(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Collect unique fight rows from predictions, slip, tiers, and skips."""
    if not isinstance(result, dict):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(raw: dict[str, Any]) -> None:
        label = _fight_label_from_row(raw).lower()
        if not label or label in seen:
            return
        seen.add(label)
        rows.append(dict(raw))

    preds = result.get("predictions")
    if isinstance(preds, pd.DataFrame) and not preds.empty:
        for _, r in preds.iterrows():
            _add(r.to_dict())

    for b in list(result.get("bet_slip") or []):
        if isinstance(b, dict):
            _add(b)

    fun_tiers = result.get("fun_tiers") or {}
    if isinstance(fun_tiers, dict):
        for bucket in fun_tiers.values():
            for b in list(bucket or []):
                if isinstance(b, dict):
                    _add(b)

    for s in list(result.get("skipped") or []):
        if isinstance(s, dict):
            _add(s)

    for p in list(result.get("props") or []):
        if isinstance(p, dict):
            _add(p)

    return rows


def _match_fights_for_question(
    question: str,
    result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    tokens = _chat_query_tokens(question)
    if not tokens:
        return []
    hits: list[tuple[int, dict[str, Any]]] = []
    for row in _iter_card_fight_rows(result):
        f1 = str(row.get("fighter_1") or row.get("fighter1") or "").strip()
        f2 = str(row.get("fighter_2") or row.get("fighter2") or "").strip()
        fight = _fight_label_from_row(row)
        pick = str(row.get("pick") or row.get("side") or row.get("predicted_winner") or "")
        blob = " ".join([f1, f2, fight, pick])
        score = sum(1 for t in tokens if _token_hits_name(t, blob) or t in fight.lower())
        if score:
            hits.append((score, row))
    hits.sort(key=lambda x: (-x[0], _fight_label_from_row(x[1])))
    return [r for _, r in hits[:3]]


def _fmt_pct(value: Any, *, already_pct: bool = False) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        v = float(value)
        if not already_pct and abs(v) <= 1.5:
            v *= 100.0
        return f"{v:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_prob(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        v = float(value)
        if v > 1.5:
            v /= 100.0
        return f"{v:.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_odds(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _lookup_tier_for_fight(
    result: dict[str, Any] | None,
    fight: str,
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    from src.predictor import _names_match

    fun_tiers = result.get("fun_tiers") or {}
    if isinstance(fun_tiers, dict):
        for tier_name, bucket in fun_tiers.items():
            for b in list(bucket or []):
                if not isinstance(b, dict):
                    continue
                label = _fight_label_from_row(b)
                if _names_match(fight, label) or fight.lower() in label.lower():
                    out = dict(b)
                    out.setdefault("bet_tier", tier_name)
                    return out
    for b in list(result.get("bet_slip") or []):
        if not isinstance(b, dict):
            continue
        label = _fight_label_from_row(b)
        if _names_match(fight, label) or fight.lower() in label.lower():
            return dict(b)
    for s in list(result.get("skipped") or []):
        if not isinstance(s, dict):
            continue
        label = _fight_label_from_row(s)
        if _names_match(fight, label) or fight.lower() in label.lower():
            out = dict(s)
            out.setdefault("bet_tier", "red")
            return out
    return None


def build_fight_stats_answer(
    question: str,
    result: dict[str, Any] | None,
    *,
    event_label: str = "",
) -> str | None:
    """Instant grounded answer for fight/stats questions — no Ollama call."""
    matches = _match_fights_for_question(question, result)
    if not matches:
        return None

    from src.bet_tiers import action_label_for_bet

    event = event_label or (
        str((result or {}).get("event") or "").strip() if isinstance(result, dict) else ""
    )
    blocks: list[str] = []
    if event:
        blocks.append(event)

    for row in matches:
        fight = _fight_label_from_row(row)
        f1 = str(row.get("fighter_1") or row.get("fighter1") or "").strip()
        f2 = str(row.get("fighter_2") or row.get("fighter2") or "").strip()
        if (not f1 or not f2) and " vs " in fight.lower():
            parts = re.split(r"\s+vs\.?\s+", fight, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                f1, f2 = parts[0].strip(), parts[1].strip()

        p1 = row.get("p_f1", row.get("prob_f1_win", row.get("prob_f1")))
        p2 = row.get("p_f2", row.get("prob_f2_win", row.get("prob_f2")))
        edge_pct = row.get("edge_pct")
        if edge_pct is None and row.get("best_edge") is not None:
            edge_pct = float(row["best_edge"]) * 100.0
        e1 = row.get("edge_f1")
        e2 = row.get("edge_f2")
        o1 = row.get("f1_odds")
        o2 = row.get("f2_odds")
        matched = row.get("odds_matched")
        source = str(row.get("odds_source") or row.get("book") or "").strip() or "n/a"
        width = row.get("interval_width")
        disagree = row.get("disagreement")
        pick = str(
            row.get("predicted_winner")
            or row.get("pick")
            or row.get("side")
            or ""
        ).strip()

        tier_row = _lookup_tier_for_fight(result, fight) or row
        action = action_label_for_bet(tier_row)
        skip_reason = str(
            tier_row.get("skip_reason")
            or tier_row.get("tier_reason")
            or row.get("skip_reason")
            or ""
        ).replace("_", " ").strip()

        lines = [fight]
        if f1 and f2:
            lines.append(
                f"  Model: {f1} {_fmt_prob(p1)} / {f2} {_fmt_prob(p2)}"
                + (f"  -> pick {pick}" if pick else "")
            )
            lines.append(
                f"  Odds: {f1} {_fmt_odds(o1)} / {f2} {_fmt_odds(o2)}"
                f"  · matched={bool(matched) if matched is not None else 'n/a'}"
                f"  · source={source}"
            )
            if e1 is not None or e2 is not None:
                lines.append(
                    f"  Edge: {f1} {_fmt_pct(e1)} / {f2} {_fmt_pct(e2)}"
                    + (f"  · best {_fmt_pct(edge_pct, already_pct=True)}" if edge_pct is not None else "")
                )
            elif edge_pct is not None:
                lines.append(f"  Best edge: {_fmt_pct(edge_pct, already_pct=True)}")
        else:
            lines.append(
                f"  Pick {pick or 'n/a'} · edge {_fmt_pct(edge_pct, already_pct=True)}"
                f" · prob {_fmt_prob(tier_row.get('prob') or p1)}"
            )
        if width is not None or disagree is not None:
            lines.append(
                f"  Uncertainty: width={_fmt_pct(width)} · disagreement={_fmt_pct(disagree)}"
            )
        lines.append(f"  Action: {action}" + (f" ({skip_reason})" if skip_reason else ""))
        blocks.append("\n".join(lines))

    blocks.append("(Live card stats — no Ollama wait)")
    return "\n\n".join(blocks)


def answer_ollama_chat(
    question: str,
    *,
    analysis_result: dict[str, Any] | None = None,
    event_label: str = "",
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Answer a user question about the current card using grounded HA slip context.

    Best-bet and fight-stats questions are answered instantly from card data.
    Open-ended chat may call Ollama, but never invents tickets.
    """
    from src.ollama_client import check_ollama_health, ollama_complete, ollama_status_message

    def _prog(msg: str, pct: float | None = None) -> None:
        if progress:
            progress(msg, pct)

    q = " ".join(str(question or "").split()).strip()
    if not q:
        return {"ok": False, "error": "Empty question.", "answer": "", "briefing": ""}

    _prog("Preparing context...", 0.15)
    briefing = build_best_bets_briefing(analysis_result)
    q_low = q.lower()
    best_intent = any(
        k in q_low
        for k in (
            "best bet",
            "best bets",
            "what should i bet",
            "what to bet",
            "recommend",
            "top ticket",
            "top 5",
            "which bet",
            "which fight",
            "value bet",
            "actionable",
        )
    )

    # Best-bet / stats questions: instant HA briefing (never invent tickets, never wait on LLM).
    if best_intent:
        _prog("Stats briefing ready", 1.0)
        return {
            "ok": True,
            "answer": build_best_bets_briefing(analysis_result, compact=True),
            "briefing": briefing,
            "source": "ha_briefing",
            "model": "stats",
        }

    # Fight-specific stats ("mcconico fight", "edge on Alvarez") — card data only.
    fight_answer = build_fight_stats_answer(
        q, analysis_result, event_label=event_label
    )
    if fight_answer:
        _prog("Fight stats ready", 1.0)
        return {
            "ok": True,
            "answer": fight_answer,
            "briefing": briefing,
            "source": "ha_fight_stats",
            "model": "stats",
        }

    stats_intent = any(
        k in q_low
        for k in (
            "stats",
            "stat",
            "edge",
            "odds",
            "probability",
            "prob",
            "interval",
            "uncertainty",
            "breakdown",
        )
    )
    if stats_intent:
        _prog("No matching fight — card briefing", 1.0)
        return {
            "ok": True,
            "answer": (
                "No matching fight found on the loaded card for that name.\n\n"
                + build_best_bets_briefing(analysis_result, compact=True)
            ),
            "briefing": briefing,
            "source": "ha_briefing",
            "model": "stats",
        }

    _prog("Checking Ollama...", 0.25)
    health = check_ollama_health(force=False)
    if not health.get("reachable") or not bool(getattr(config, "OLLAMA_ENABLED", True)):
        _prog("Ollama offline — stats only", 1.0)
        return {
            "ok": True,
            "answer": briefing
            + "\n\n(Ollama offline — showing HA/stats briefing only. "
            + (health.get("banner") or ollama_status_message())
            + ")",
            "briefing": briefing,
            "source": "ha_briefing",
            "model": "stats",
            "health_banner": health.get("banner"),
        }

    context = _analysis_context_for_chat(analysis_result)
    event = event_label or (
        str((analysis_result or {}).get("event") or "") if analysis_result else ""
    )
    system = (
        "You are a concise UFC betting assistant for this dashboard. "
        "Use ONLY the provided ticket/stats context. "
        "Never invent fights, odds, edges, or stakes. "
        "Always separate BET THIS (passed HA gates, sized $) from "
        "TINY PAPER BET (failed wide CI, Paper only) from FUN ONLY ($0 research) and DO NOT BET. "
        "If no BET THIS tickets, say HA NO BET first even if Sky Blue overrides exist. "
        "Never imply FUN ONLY or Sky Blue picks passed the full HA test. "
        "Keep answers under 180 words with short bullets when listing bets."
    )
    prompt = (
        f"Event: {event or 'current card'}\n\n"
        f"GROUNDED CONTEXT (source of truth):\n{context}\n\n"
        f"USER QUESTION: {q}\n\n"
        "Answer using the context. If they ask what to bet, start with "
        "'WHAT TO BET (HA — passed gates):' listing only BET THIS tickets + dollar stakes; "
        "then 'PAPER OVERRIDE (failed wide CI):' for Sky Blue if any; "
        "then 'FUN ONLY ($0):' if any; then 'SKIP:' for caution/red."
    )
    try:
        _prog("Ollama is answering...", 0.45)
        # Chat replies should stay short — soft-cap timeout so CPU hosts don't hang.
        chat_timeout = min(45, int(getattr(config, "OLLAMA_TIMEOUT_SEC", 60) or 60))
        model_used, text = ollama_complete(
            prompt,
            system=system,
            timeout_sec=max(20, chat_timeout),
            json_mode=False,
            temperature=0.2,
        )
        _prog("Formatting reply...", 0.9)
        answer = " ".join(str(text or "").split())
        if not answer:
            answer = briefing
        _prog("Done", 1.0)
        return {
            "ok": True,
            "answer": answer,
            "briefing": briefing,
            "source": "ollama_chat",
            "model": model_used,
        }
    except Exception as exc:
        _prog("Chat failed — stats only", 1.0)
        logger.warning("Ollama chat failed: %s", exc)
        fallback = build_fight_stats_answer(
            q, analysis_result, event_label=event_label
        ) or build_best_bets_briefing(analysis_result, compact=True)
        return {
            "ok": True,
            "answer": f"{fallback}\n\n(Ollama chat timed out / failed — card stats above)",
            "briefing": briefing,
            "source": "ha_briefing",
            "model": "stats",
            "error": str(exc),
        }


def _lookup_pick(grok_picks: list[dict[str, Any]], bet: dict[str, Any]) -> dict[str, Any] | None:
    keys = [
        str(bet.get("fight_id") or "").strip(),
        str(bet.get("pick_line") or "").strip(),
        str(bet.get("fight") or "").strip(),
        str(bet.get("pick") or "").strip(),
        str(bet.get("label") or "").strip(),
    ]
    by_id = {str(p.get("id") or "").strip(): p for p in grok_picks if p.get("id")}
    for key in keys:
        if key and key in by_id:
            return by_id[key]
    return None


def apply_grok_kelly_adjustments(
    bets: list[dict[str, Any]] | dict[str, Any],
    grok_result: dict[str, Any] | None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Apply narrative Kelly tilt to stake fields only (model-first).

    Must run AFTER uncertainty gates + strategy rating have set model stakes.
    Never flips pick, never changes edge/prob. Fail-closed → factor 1.0.
    """
    if isinstance(bets, dict) and ("singles" in bets or "parlays" in bets):
        singles = apply_grok_kelly_adjustments(list(bets.get("singles") or []), grok_result)
        parlays = apply_grok_kelly_adjustments(list(bets.get("parlays") or []), grok_result)
        assert isinstance(singles, list) and isinstance(parlays, list)
        out = dict(bets)
        out["singles"] = singles
        out["parlays"] = parlays
        out["items"] = [*singles, *parlays]
        return out

    # No analysis run yet — leave model stakes untouched (no log spam).
    if grok_result is None:
        return bets

    grok_ok = bool(grok_result.get("ok"))
    grok_picks = list(grok_result.get("picks") or []) if grok_ok else []

    adjusted: list[dict[str, Any]] = []
    for bet in bets:  # type: ignore[union-attr]
        row = dict(bet)
        # Snapshot model-owned fields so narrative cannot mutate them
        locked = {k: row.get(k) for k in _MODEL_LOCKED_FIELDS if k in row}
        row["model_edge_pct"] = row.get("edge_pct", row.get("edge"))
        row["model_pick"] = row.get("pick") or row.get("predicted_winner")

        item = _lookup_pick(grok_picks, row) if grok_ok else None
        decision = resolve_narrative_tilt(row, item, grok_ok=grok_ok, context="apply_kelly")
        factor = float(decision.factor)

        row["grok_kelly_factor"] = factor
        row["narrative_tilt_status"] = decision.status
        row["narrative_tilt_reason"] = decision.reason
        row["grok_conviction"] = decision.conviction or (item or {}).get("conviction", "")
        row["grok_narrative"] = decision.narrative
        if decision.narrative:
            # Explanation only — keep existing model brief when present
            row["reason"] = decision.narrative
            if not row.get("brief"):
                row["brief"] = decision.narrative
        if item:
            row["grok_crowd"] = item.get("crowd_positioning", "")
            row["grok_risks"] = item.get("invalidation_risks") or []

        if abs(factor - 1.0) > 1e-9:
            for field in _NARRATIVE_STAKE_FIELDS:
                if row.get(field) is not None:
                    try:
                        row[field] = round(float(row[field]) * factor, 2)
                    except (TypeError, ValueError):
                        pass

        # Restore locked model fields (never flip pick / inflate edge)
        for k, v in locked.items():
            row[k] = v
        adjusted.append(row)
    return adjusted


def attach_narrative_tilts_to_alerts(
    alerts: dict[str, Any] | None,
    grok_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply narrative Kelly tilts to alert singles/parlays after model sizing."""
    if not alerts:
        return alerts or {}
    out = dict(alerts)
    applied = apply_grok_kelly_adjustments(
        {"singles": list(out.get("singles") or []), "parlays": list(out.get("parlays") or [])},
        grok_result,
    )
    assert isinstance(applied, dict)
    out["singles"] = applied.get("singles") or []
    out["parlays"] = applied.get("parlays") or []
    out["narrative_tilt_applied"] = bool(grok_result and grok_result.get("ok"))
    return out
