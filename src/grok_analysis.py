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
from typing import Any

import config

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
    """Prompt for concise Top-5 bet-slip narration (actionable + advisory)."""
    event = inputs.get("event") or "Upcoming UFC card"
    profile = str(inputs.get("profile") or "paper").upper()
    bankroll = inputs.get("bankroll")
    card_budget = inputs.get("card_budget")
    tickets = inputs.get("tickets") or []
    skipped = inputs.get("skipped") or []
    warning = str(inputs.get("top5_warning") or TOP5_WARNING)

    ticket_lines: list[str] = []
    for t in tickets:
        tier = "ADVISORY" if t.get("advisory") else "ACTIONABLE"
        ticket_lines.append(
            f"- id={t.get('id')} | tier={tier} | side={t.get('side')} | market={t.get('market')} | "
            f"book={t.get('book') or 'n/a'} | odds={t.get('odds_display') or '-'} | "
            f"stake_pct={t.get('stake_pct')}% | stake_usd=${t.get('stake_usd')} | "
            f"model_prob={t.get('prob')} | edge={t.get('edge_pct')}% | "
            f"confidence={t.get('confidence')} | strength={t.get('strength_score')} | "
            f"uncertainty={t.get('uncertainty_action') or 'allow'}"
        )
    tickets_block = "\n".join(ticket_lines) if ticket_lines else "- (none — NO BET)"

    skip_lines = [
        f"- SKIP {s.get('pick') or '-'} | {s.get('fight')} | reason={s.get('skip_reason')} | "
        f"disagree={s.get('disagreement')} | width={s.get('interval_width')}"
        for s in skipped[:12]
    ]
    skip_section = (
        "\nExplicit skips (already fail-closed by HA gates — do not recommend):\n"
        + "\n".join(skip_lines)
        + "\n"
        if skip_lines
        else ""
    )

    lessons_block = ""
    try:
        from src.prediction_bank import lessons_prompt_block

        lessons_block = lessons_prompt_block()
    except Exception:
        lessons_block = ""
    lessons_section = f"\n{lessons_block}\n" if lessons_block else ""

    br_txt = f"${float(bankroll):.2f}" if bankroll is not None else "n/a"
    card_txt = f"${float(card_budget):.2f}" if card_budget is not None else "n/a"
    total_pct = inputs.get("total_stake_pct")
    total_usd = inputs.get("total_stake_usd")
    n_act = inputs.get("n_actionable")
    n_adv = inputs.get("n_advisory")

    return f"""You are a UFC betting desk. Output ONLY what to bet and why — short and actionable.
Profile={profile} | Bankroll={br_txt} | Card budget={card_txt}
WARNING: {warning}
Stakes on ACTIONABLE rows are FINAL from conf/odds HA sizing (sum ≤ 100% of card). Do NOT change % or $.
ADVISORY rows must keep stake_pct=0 and stake_usd=0 — research only.
{lessons_section}{skip_section}
MODEL-FIRST CONSTRAINTS (mandatory):
- Use ONLY the tickets listed (up to Top 5). Do NOT invent, flip, add, or drop bets.
- Do NOT change stake_pct or stake_usd. Copy them exactly into your JSON.
- NEVER invent odds, edge, or probability.
- If tickets list is empty → summary must say NO BET and picks=[].
- One-line reason each: confidence + edge (max ~100 chars). No essays.
- Summary should mention how many are ACTIONABLE vs ADVISORY when both exist.

Top 5 tickets to narrate ({n_act} actionable / {n_adv} advisory):
{tickets_block}

Card totals (ACTIONABLE only): {total_pct}% / ${total_usd}

Reply with ONLY valid JSON (no markdown). Keep strings short; use apostrophes inside strings, never raw " quotes.
{{
  "event": "{event}",
  "summary": "one short line: top actionable bets / note advisory / or NO BET",
  "picks": [
    {{
      "id": "same id from input",
      "side": "same side",
      "market": "same market",
      "book": "same book",
      "stake_pct": 0.0,
      "stake_usd": 0.0,
      "reason": "conf + edge one-liner",
      "conviction": "high|medium|low"
    }}
  ]
}}"""


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
    is_parlay = bool(ticket.get("is_parlay")) or int(ticket.get("n_legs") or 1) >= 2
    is_prop = (
        str(ticket.get("market_type") or "").lower() == "prop"
        or str(ticket.get("prop_key") or "") == "over_1_5_rounds"
    )
    if is_parlay:
        market = "2-leg parlay"
        side = str(ticket.get("pick_line") or ticket.get("picks") or ticket.get("display_label") or "")
    elif is_prop:
        market = "Over 1.5 Rounds"
        side = str(ticket.get("display_label") or ticket.get("label") or ticket.get("pick_line") or "")
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
        "edge_pct": round(edge_f * 100.0, 1),
        "confidence": conf,
        "strength_score": ticket.get("strength_score"),
        "uncertainty_action": ticket.get("uncertainty_action") or "allow",
        "fight": ticket.get("fight"),
        "pick": ticket.get("pick"),
        "is_parlay": is_parlay,
        "reason": "",
        "conviction": "high" if conf_l == "high" else ("low" if conf_l == "low" else "medium"),
        "tier": tier_l,
        "advisory": tier_l == "advisory",
    }


TOP5_WARNING = (
    "Top 5 structure: ACTIONABLE = HA-gated + card-sized ($). "
    "ADVISORY = ranked for research only ($0, not in card budget). "
    "Do not treat advisory rows as sized bets."
)


def _candidate_dedupe_key(ticket: dict[str, Any]) -> str:
    return str(
        ticket.get("fight_id")
        or ticket.get("fight")
        or ticket.get("pick_line")
        or ticket.get("picks")
        or ticket.get("display_label")
        or ticket.get("id")
        or ""
    ).strip().lower()


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
    Gather HA-gated + conf/odds-sized tickets for Ollama bet-slip narration.

    Always builds a Top 5 list:
    - ACTIONABLE: cleared HA ticket cap + sized stakes
    - ADVISORY: next-best ranked picks with $0 (research only), when fewer than 5
      actionable tickets exist

    Does not invent bets — uses aggregate_overview_recommendations + top singles pool.
    """
    from src.strategy import (
        aggregate_overview_recommendations,
        aggregate_top_recommended_bets,
    )

    fight_cap = max_fights if max_fights is not None else config.GROK_MAX_FIGHTS
    _ = max_props  # props come through overview Over 1.5 path
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

    # Actionable: positive stake after fail-closed checks
    actionable_raw: list[dict[str, Any]] = []
    for i, t in enumerate(items, start=1):
        row = _ticket_to_slip_row(t, rank=i, tier="actionable")
        if float(row.get("stake_usd") or 0) > 0 and float(row.get("stake_pct") or 0) > 0:
            actionable_raw.append(t)

    actionable_by_key = {_candidate_dedupe_key(t): t for t in actionable_raw}

    # Ranked pool to fill Top 5 (actionable first, then other strong singles)
    ranked_pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in actionable_raw:
        key = _candidate_dedupe_key(t)
        if key and key not in seen:
            seen.add(key)
            ranked_pool.append(t)
    try:
        extra = aggregate_top_recommended_bets(
            books, bs, limit=top_n, per_book_cap=2, profile=prof
        )
    except Exception:
        extra = []
    for t in list(slip.get("prop_singles") or []) + list(slip.get("parlays") or []) + list(extra):
        key = _candidate_dedupe_key(t)
        if not key or key in seen:
            continue
        seen.add(key)
        ranked_pool.append(t)
        if len(ranked_pool) >= top_n:
            break

    tickets: list[dict[str, Any]] = []
    for i, t in enumerate(ranked_pool[:top_n], start=1):
        key = _candidate_dedupe_key(t)
        if key in actionable_by_key:
            row = _ticket_to_slip_row(actionable_by_key[key], rank=i, tier="actionable")
            # Re-apply rank after merge order
            if float(row.get("stake_usd") or 0) > 0:
                tickets.append(row)
                continue
        tickets.append(_ticket_to_slip_row(t, rank=i, tier="advisory"))

    # If pool was empty but we somehow have nothing, keep empty (NO BET)
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

    actionable_only = [t for t in tickets if not t.get("advisory")]
    total_pct = round(sum(float(t.get("stake_pct") or 0) for t in actionable_only), 1)
    total_usd = round(sum(float(t.get("stake_usd") or 0) for t in actionable_only), 2)
    br = float(bs.get("total_bankroll") or config.DEFAULT_TOTAL_BANKROLL)

    fights = [t for t in tickets if t.get("market") == "moneyline"]
    props = [t for t in tickets if "Over 1.5" in str(t.get("market") or "")]
    n_actionable = len(actionable_only)
    n_advisory = sum(1 for t in tickets if t.get("advisory"))

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

    return {
        "event": str(raw.get("event") or event_label or ""),
        "summary": summary,
        "picks": picks,
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


def query_ollama(prompt: str) -> tuple[str, str]:
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
    model_used, text = ollama_complete(
        prompt,
        system=system,
        timeout_sec=int(getattr(config, "OLLAMA_TIMEOUT_SEC", 600)),
        json_mode=True,
        temperature=0.25,
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
        out: dict[str, Any] = {
            "event": event_label or inputs.get("event") or "",
            "picks": [],
            "bet_slip": slip,
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
        }
        out.update(extra)
        return out

    if not bool(getattr(config, "OLLAMA_ENABLED", True)):
        logger.warning(
            "Ollama health error_class=disabled latency_ms=%s — showing model tickets only",
            _latency_ms(),
        )
        return _base_result(
            ok=False,
            error="Ollama disabled — set OLLAMA_ENABLED=true in .env",
            error_class="disabled",
            health_banner="Ollama offline — showing model tickets only",
            summary=no_bet_reason
            or "HA bet slip ready — Ollama reasons unavailable.",
            model=getattr(config, "OLLAMA_MODEL", "ollama"),
        )

    health = check_ollama_health(force=True)
    health_class = str(health.get("error_class") or "other")
    if health_class in {"offline", "model_missing"} or not health.get("reachable"):
        banner = str(
            health.get("banner") or "Ollama offline — showing model tickets only"
        )
        logger.warning(
            "Ollama health error_class=%s latency_ms=%s — %s",
            health_class,
            health.get("latency_ms"),
            banner,
        )
        return _base_result(
            ok=False,
            error=health.get("error") or ollama_status_message(),
            error_class=health_class if health_class != "ok" else "offline",
            health_banner=banner,
            ollama_latency_ms=health.get("latency_ms"),
            summary=no_bet_reason
            or "HA bet slip ready — Ollama reasons unavailable.",
            model=health.get("resolved_model")
            or getattr(config, "OLLAMA_MODEL", "ollama"),
        )

    # Explicit NO BET when no odds or nothing at all to show
    if no_odds or not tickets:
        return {
            "ok": True,
            "no_bet": True,
            "no_usable_odds": no_odds,
            "event": event_label or inputs.get("event") or "",
            "summary": no_bet_reason or "NO BET — nothing cleared HA gates for this card.",
            "picks": [],
            "bet_slip": [],
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
        cached = _load_cache(cache_key)
        if cached:
            out = dict(cached)
            out["from_cache"] = True
            out["ok"] = True
            out["bet_slip"] = merge_ollama_reasons_into_slip(
                tickets, out.get("picks") or []
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

    prompt = build_grok_prompt(inputs)
    try:
        model_used, raw_text = query_ollama(prompt)
        try:
            parsed = _extract_json_blob(raw_text)
        except ValueError:
            logger.warning("Primary JSON parse failed; requesting Ollama repair pass")
            fix_prompt = (
                "Fix the following into ONE valid JSON object with keys "
                "event, summary, picks. Keep all complete picks. "
                "Output JSON only.\n\n"
                f"{raw_text[:12000]}"
            )
            model_used, raw_text = query_ollama(fix_prompt)
            parsed = _extract_json_blob(raw_text)
        result = normalize_grok_result(parsed, event_label=event_label or inputs.get("event", ""))
        result["ok"] = True
        result["from_cache"] = False
        result["model"] = model_used
        result["source"] = "ollama"
        result["no_bet"] = bool(inputs.get("no_bet"))
        result["no_usable_odds"] = no_odds
        result["bet_slip"] = merge_ollama_reasons_into_slip(tickets, result.get("picks") or [])
        result["skipped"] = skipped
        result["total_stake_pct"] = inputs.get("total_stake_pct")
        result["total_stake_usd"] = inputs.get("total_stake_usd")
        result["card_budget"] = inputs.get("card_budget")
        result["bankroll"] = inputs.get("bankroll")
        result["profile"] = inputs.get("profile")
        result["top5_warning"] = inputs.get("top5_warning") or TOP5_WARNING
        result["n_actionable"] = inputs.get("n_actionable")
        result["n_advisory"] = inputs.get("n_advisory")
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
        return result
    except Exception as exc:
        err_class = classify_ollama_error(exc)
        latency = _latency_ms()
        logger.warning(
            "Ollama analysis failed error_class=%s latency_ms=%s: %s",
            err_class,
            latency,
            exc,
        )
        banner = (
            "Ollama offline — showing model tickets only"
            if err_class in {"offline", "model_missing", "timeout"}
            else "Ollama error — showing model tickets only"
        )
        if err_class == "timeout":
            banner = "Ollama timeout — showing model tickets only"
        return _base_result(
            ok=False,
            error=str(exc),
            error_class=err_class,
            health_banner=banner,
            summary=no_bet_reason
            or "HA bet slip ready — Ollama reasons unavailable.",
            model=getattr(config, "OLLAMA_MODEL", "ollama"),
            source="ha_slip",
        )


analyze_card_with_ollama = analyze_card_with_grok


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
