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


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def ticket_market_type(ticket: dict[str, Any]) -> str:
    """Normalize market_type for identity: moneyline | prop | parlay."""
    if ticket.get("is_parlay") or int(ticket.get("n_legs") or 1) >= 2:
        return "parlay"
    mt = str(ticket.get("market_type") or "").strip().lower()
    market = str(ticket.get("market") or "").strip().lower()
    prop_key = str(ticket.get("prop_key") or "").strip().lower()
    if mt == "prop" or prop_key == "over_1_5_rounds" or "over 1.5" in market:
        return "prop"
    if mt == "parlay" or "parlay" in market:
        return "parlay"
    return "moneyline"


def ticket_selection(ticket: dict[str, Any]) -> str:
    """Normalize selection / side for identity matching."""
    market = ticket_market_type(ticket)
    if market == "prop":
        # Collapse Over 1.5 label variants onto one selection token
        raw = " ".join(
            str(
                ticket.get("prop_key")
                or ticket.get("label")
                or ticket.get("display_label")
                or ticket.get("prop_short")
                or ticket.get("side")
                or ticket.get("pick")
                or "over_1_5_rounds"
            )
            .strip()
            .lower()
            .replace("—", " ")
            .replace("-", " ")
            .split()
        )
        if "over" in raw and "1.5" in raw.replace(" ", ""):
            return "over_1_5_rounds"
        if raw in {"over_1_5_rounds", "over 1.5", "over 1.5 rounds"}:
            return "over_1_5_rounds"
        return raw or "over_1_5_rounds"
    if market == "parlay":
        return " ".join(
            str(
                ticket.get("pick_line")
                or ticket.get("picks")
                or ticket.get("display_label")
                or ticket.get("side")
                or ""
            )
            .strip()
            .lower()
            .split()
        )
    pick = str(
        ticket.get("pick")
        or ticket.get("side")
        or ticket.get("display_label")
        or ticket.get("pick_line")
        or ""
    ).strip().lower()
    # Strip trailing " ml"
    if pick.endswith(" ml"):
        pick = pick[:-3].strip()
    # If side looks like "Name over Opponent", keep first fighter
    if " over " in pick:
        pick = pick.split(" over ", 1)[0].strip()
    if " — " in pick:
        pick = pick.split(" — ", 1)[0].strip()
    return " ".join(pick.split())


def _norm_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _last_token(name: str) -> str:
    parts = _norm_name(name).split()
    return parts[-1] if parts else ""


def _parse_fighter_pair(text: str) -> tuple[str, str] | None:
    """Extract (fighter_a, fighter_b) from 'A vs B', 'A|B', or similar."""
    raw = str(text or "").strip()
    if not raw:
        return None
    # Drop prop / market suffixes after em-dash before parsing
    for side_sep in (" — ", " – ", " - Over", " - over"):
        if side_sep in raw:
            raw = raw.split(side_sep, 1)[0].strip()
            break
    lower = raw.lower()
    if " vs " in lower:
        idx = lower.index(" vs ")
        a = raw[:idx]
        b = raw[idx + 4 :]
    elif "|" in raw:
        a, b = raw.split("|", 1)
    else:
        return None
    for side_sep in (" — ", " – "):
        if side_sep in b:
            b = b.split(side_sep, 1)[0]
        if side_sep in a:
            a = a.split(side_sep, 1)[0]
    a_n, b_n = _norm_name(a), _norm_name(b)
    if not a_n or not b_n:
        return None
    return a_n, b_n


def _pair_keys(a: str, b: str) -> set[str]:
    """Identity keys for a fight (full-name + last-name, order-invariant)."""
    full = tuple(sorted((_norm_name(a), _norm_name(b))))
    last = tuple(sorted((_last_token(a), _last_token(b))))
    keys = {f"{full[0]}|{full[1]}"}
    if last[0] and last[1]:
        keys.add(f"{last[0]}|{last[1]}")
    return keys


def ticket_fight_aliases(ticket: dict[str, Any]) -> set[str]:
    """All fight-identity aliases so f1|f2 and 'F1 vs F2' collapse."""
    aliases: set[str] = set()
    for field in (
        ticket.get("fight_id"),
        ticket.get("fight"),
        ticket.get("side"),
        ticket.get("pick_line"),
        ticket.get("display_label"),
        ticket.get("id"),
    ):
        pair = _parse_fighter_pair(str(field or ""))
        if pair:
            aliases |= _pair_keys(*pair)
    # Bare fight_id already normalized
    fid = _norm_name(str(ticket.get("fight_id") or ""))
    if fid and "|" in fid:
        pair = _parse_fighter_pair(fid)
        if pair:
            aliases |= _pair_keys(*pair)
        else:
            aliases.add(fid)
    return {a for a in aliases if a}


def ticket_fight_id(ticket: dict[str, Any]) -> str:
    """Canonical fight id: sorted full-name pair when parseable."""
    aliases = ticket_fight_aliases(ticket)
    if not aliases:
        fight = _norm_name(str(ticket.get("fight") or ticket.get("fight_id") or ""))
        return fight
    # Prefer full-name keys (contain a space) over last-name-only
    full = sorted((a for a in aliases if " " in a.replace("|", " ")), key=len, reverse=True)
    if full:
        return full[0]
    return sorted(aliases, key=len, reverse=True)[0]


def ticket_book(ticket: dict[str, Any]) -> str:
    book = str(ticket.get("book") or ticket.get("book_key") or "").strip().lower()
    if book in {"", "-", "n/a", "none", "nan"}:
        return ""
    aliases = {
        "the_odds_api": "odds api",
        "odds_api": "odds api",
        "overview": "overview",
    }
    return aliases.get(book, book)


def ticket_dedupe_key(ticket: dict[str, Any]) -> tuple[str, str, str, str]:
    """Identity: (fight_id, market_type, selection, book)."""
    return (
        ticket_fight_id(ticket),
        ticket_market_type(ticket),
        ticket_selection(ticket),
        ticket_book(ticket),
    )


def ticket_clears_gates(ticket: dict[str, Any]) -> bool:
    """True when ticket is a real HA-sized / Blue or Sky Blue clears-gates pick."""
    tier = str(ticket.get("bet_tier") or ticket.get("tier") or "").strip().lower()
    if ticket.get("fun_bet") or ticket.get("advisory"):
        # Explicit fun/advisory never clears, even if stale stake fields linger
        if _safe_float(ticket.get("stake_usd")) <= 0 and _safe_float(
            ticket.get("suggested_stake")
        ) <= 0 and _safe_float(ticket.get("stake_pct")) <= 0:
            return False
    stake = max(
        _safe_float(ticket.get("stake_usd")),
        _safe_float(ticket.get("suggested_stake")),
    )
    stake_pct = _safe_float(ticket.get("stake_pct"))
    if stake > 0 or stake_pct > 0:
        return True
    if tier in {"blue", "sky_blue"} and not ticket.get("fun_bet") and not ticket.get("advisory"):
        return True
    return False


def _ticket_edge(ticket: dict[str, Any]) -> float:
    edge = ticket.get("edge")
    if edge is not None:
        e = _safe_float(edge)
        if abs(e) > 1.5:
            e = e / 100.0
        return e
    if ticket.get("edge_pct") is not None:
        return _safe_float(ticket.get("edge_pct")) / 100.0
    return 0.0


def _ticket_stake(ticket: dict[str, Any]) -> float:
    return max(
        _safe_float(ticket.get("stake_usd")),
        _safe_float(ticket.get("suggested_stake")),
        _safe_float(ticket.get("stake_pct")),
    )


def ticket_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Prefer CLEARS GATES over FUN; then higher edge; then higher stake."""
    c_clear = ticket_clears_gates(candidate)
    i_clear = ticket_clears_gates(incumbent)
    if c_clear != i_clear:
        return c_clear
    c_edge = _ticket_edge(candidate)
    i_edge = _ticket_edge(incumbent)
    if abs(c_edge - i_edge) > 1e-9:
        return c_edge > i_edge
    return _ticket_stake(candidate) > _ticket_stake(incumbent)


def _rank_bucket(ticket: dict[str, Any]) -> int:
    """Strict color order: blue(0) → sky_blue(1) → green(2) → yellow(3) → red(4)."""
    tier = str(ticket.get("bet_tier") or ticket.get("tier") or "").strip().lower()
    # Normalize aliases that must never outrank their true color
    if tier in {"don't bet", "dont_bet", "dont-bet", "no_bet"}:
        tier = "red"
    if tier in {"caution", "warn", "warning", "amber", "orange"}:
        tier = "yellow"
    if tier in {"sky", "skyblue", "light_blue", "lightblue"}:
        tier = "sky_blue"
    order = {"blue": 0, "sky_blue": 1, "green": 2, "yellow": 3, "red": 4}
    if tier in order:
        return order[tier]
    # Untiered clears-gates with override reason → sky (after deep blue)
    unc = str(ticket.get("uncertainty_reason") or "").lower()
    if "paper_wide_override" in unc and ticket_clears_gates(ticket):
        return 1
    if ticket_clears_gates(ticket):
        return 0
    # Untiered: never promote unknown/red-ish rows above yellow
    if ticket.get("fun_bet") and not ticket.get("advisory"):
        return 2
    if ticket.get("fun_bet") or ticket.get("advisory"):
        return 3
    return 4


def _bet_cluster_key(ticket: dict[str, Any]) -> tuple[str, str]:
    """Market+selection cluster (fight matched via aliases separately)."""
    return (ticket_market_type(ticket), ticket_selection(ticket))


def _same_bet(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two tickets are the same bet (fight aliases + market + selection)."""
    if _bet_cluster_key(a) != _bet_cluster_key(b):
        return False
    aliases_a = ticket_fight_aliases(a)
    aliases_b = ticket_fight_aliases(b)
    if aliases_a and aliases_b:
        return bool(aliases_a & aliases_b)
    # Fallback: canonical fight_id string equality
    fa, fb = ticket_fight_id(a), ticket_fight_id(b)
    return bool(fa) and fa == fb


def dedupe_rank_top_tickets(
    tickets: list[dict[str, Any]] | None,
    *,
    limit: int = 5,
    event: str = "",
    log: Any = None,
) -> list[dict[str, Any]]:
    """
    Merge HA + fun candidates: dedupe → rank → Top N.

    Same bet = overlapping fight aliases + market_type + selection
    (book ignored so Odds API / MyBookie / Overview do not double-count).
    Keep better: clears gates > fun; higher edge; higher stake.
    Rank: clears gates first (stake/edge), then decent fun, then others.
    """
    import logging

    logger = log or logging.getLogger(__name__)
    raw = [dict(t) for t in (tickets or []) if isinstance(t, dict)]
    raw_n = len(raw)

    # Greedy cluster merge — O(n^2) fine for card-sized lists
    clusters: list[dict[str, Any]] = []
    for t in raw:
        merged = False
        for i, kept in enumerate(clusters):
            if _same_bet(t, kept):
                if ticket_is_better(t, kept):
                    clusters[i] = t
                merged = True
                break
        if not merged:
            clusters.append(t)

    deduped = list(clusters)
    deduped_n = len(deduped)

    deduped.sort(
        key=lambda t: (
            _rank_bucket(t),
            -_ticket_stake(t),
            -_ticket_edge(t),
            ticket_selection(t),
            ticket_fight_id(t),
        )
    )

    # Top recommended is for plays / fun leans — drop "don't bet" (red) from the list.
    # Buckets: blue=0, sky_blue=1, green=2, yellow=3, red=4
    playable = [t for t in deduped if _rank_bucket(t) < 4]
    if not playable:
        playable = deduped  # only reds exist — still show something ranked last
    shown = playable[: max(0, int(limit))]
    for i, t in enumerate(shown, start=1):
        t["rank"] = i

    logger.info(
        "top_recommended event=%r raw=%s deduped=%s shown=%s limit=%s",
        event or "",
        raw_n,
        deduped_n,
        len(shown),
        limit,
    )
    return shown


def top_recommended_label(count: int, *, limit: int = 5) -> str:
    """UI label that matches the number of tickets shown."""
    n = max(0, int(count))
    if n <= 0:
        return f"Top {int(limit)} recommended"
    return f"Top {n} recommended"
