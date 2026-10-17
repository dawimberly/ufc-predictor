"""Live quote identity, freshness, and book provenance.

ODDS_FETCH_ONCE may skip another download. It does not make a cached line current.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.predictor import _fighter_name_key, _names_match

ACTIONABLE_STAKE_FIELDS = (
    "suggested_stake",
    "stake_usd",
    "stake_pct",
    "raw_stake",
    "kelly_stake_usd",
    "kelly_pct",
    "max_safe_bet_usd",
)

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_PRICE_CONSENSUS = {"consensus", "mean", "median", "average", "advisory"}
# A provider name is not an executable book. A real book fetched through that
# provider keeps its own bookmaker and is not rejected for the transport name.
_NOT_A_BOOK = _PRICE_CONSENSUS | {"odds api", "the_odds_api", "the odds api"}
_ADVISORY_MATCHES = {"advisory_fuzzy", "ambiguous", "surname", "substring"}
_FUTURE_SKEW = timedelta(minutes=2)


def _core_name(name: str) -> str:
    key = _fighter_name_key(name)
    parts = [part for part in key.split() if part and part not in _SUFFIXES]
    return " ".join(parts)


def _alias_cores(name: str) -> set[str]:
    """Exact normalized name plus explicit verified aliases. No surname expansion."""
    core = _core_name(name)
    if not core:
        return set()
    found = {core}
    try:
        from src.fighter_aliases import FIGHTER_NAME_ALIASES
    except Exception:
        return found
    for key, variants in FIGHTER_NAME_ALIASES.items():
        group = {_core_name(key)}
        group.update(_core_name(variant) for variant in variants)
        group.discard("")
        if core in group:
            found |= group
    return found


def names_match_actionable(left: str, right: str) -> bool:
    """Full-name equality or an explicit alias. Surname-only and substrings do not match."""
    cores = _alias_cores(left)
    other = _alias_cores(right)
    if not cores or not other:
        return False
    return bool(cores & other)


def pair_alignment(card_f1: str, card_f2: str, quote_f1: str, quote_f2: str) -> str | None:
    """Return aligned or reversed when both fighters match. Opponent mismatches return None."""
    if names_match_actionable(card_f1, quote_f1) and names_match_actionable(card_f2, quote_f2):
        return "aligned"
    if names_match_actionable(card_f1, quote_f2) and names_match_actionable(card_f2, quote_f1):
        return "reversed"
    return None


def _fuzzy_pair(card_f1: str, card_f2: str, quote_f1: str, quote_f2: str) -> str | None:
    if _names_match(card_f1, quote_f1) and _names_match(card_f2, quote_f2):
        return "aligned"
    if _names_match(card_f1, quote_f2) and _names_match(card_f2, quote_f1):
        return "reversed"
    return None


def parse_quote_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        stamp = pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return None
    if stamp is None or pd.isna(stamp):
        return None
    parsed = stamp.to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(value: Any) -> datetime | None:
    return parse_quote_time(value)


def quote_max_age() -> timedelta:
    import config

    minutes = float(getattr(config, "ODDS_CACHE_TTL_MINUTES", 20) or 20)
    return timedelta(minutes=max(0.0, minutes))


def _id_namespace(value: str) -> str:
    """Group ids that can be compared. Different groups are not a conflict."""
    text = str(value or "").strip()
    prefixed = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]{0,40}):(.+)", text)
    if prefixed and prefixed.group(2).strip():
        return prefixed.group(1).lower()
    low = text.lower()
    if re.fullmatch(r"[0-9a-f]{16,}", low):
        return "hex"
    if text.isdigit():
        return "numeric"
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", low):
        return "slug"
    return "opaque"


def _id_relation(card_id: str, quote_id: str) -> str:
    """match, conflict, incomparable, or missing."""
    if not card_id or not quote_id:
        return "missing"
    if card_id == quote_id:
        return "match"
    if _id_namespace(card_id) != _id_namespace(quote_id):
        return "incomparable"
    return "conflict"


def _utc_calendar_date(value: Any):
    """Calendar date in UTC. Naive values are UTC. Unparseable values are None."""
    parsed = parse_quote_time(value)
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc).date()


def event_identity_failure(
    card_event_id: Any,
    quote_event_id: Any,
    card_date: Any,
    commence: Any,
) -> str:
    """Empty when the quote belongs to this card.

    A same-namespace id conflict is rejected even if the dates match.
    Missing ids, or ids from different namespaces, fall back to the UTC date.
    """
    card_id = str(card_event_id or "").strip()
    quote_id = str(quote_event_id or "").strip()
    relation = _id_relation(card_id, quote_id)
    if relation == "conflict":
        return "wrong_event"
    if relation == "match":
        return ""
    card_day = _utc_calendar_date(card_date)
    quote_day = _utc_calendar_date(commence)
    if card_day and quote_day and card_day != quote_day:
        return "wrong_event"
    if card_day and quote_day:
        return ""
    return "missing_event_identity"


def _event_compatible(card_event_id: Any, card_date: Any, row: pd.Series | dict[str, Any]) -> bool:
    quote_id = str(row.get("event_id") or row.get("quote_event_id") or "").strip()
    card_id = str(card_event_id or "").strip()
    card_day = _utc_calendar_date(card_date)
    # A card that states neither id nor date does not filter the file.
    # Multiple name hits are still rejected as ambiguous.
    if not card_id and card_day is None:
        return True
    commence = row.get("commence_time") or row.get("quote_event_date")
    return event_identity_failure(card_id, quote_id, card_date, commence) == ""


def _align_row(row: pd.Series, *, reversed_order: bool) -> pd.Series:
    out = row.copy()
    if reversed_order:
        for left, right in (
            ("f1_odds", "f2_odds"),
            ("implied_prob_f1", "implied_prob_f2"),
            ("fighter_1", "fighter_2"),
        ):
            if left in out.index or right in out.index:
                out[left], out[right] = out.get(right), out.get(left)
    return out


def select_moneyline_quote(
    fighter_1: str,
    fighter_2: str,
    odds: pd.DataFrame | None,
    *,
    event_id: Any = None,
    event_date: Any = None,
) -> dict[str, Any]:
    """Pick at most one quote for this fight. Ambiguous or wrong-event rows are not used."""
    rejected = {
        "status": "rejected",
        "reason": "no_quote",
        "quote_match": "none",
        "row": None,
    }
    if odds is None or getattr(odds, "empty", True):
        return rejected

    actionable: list[tuple[pd.Series, str]] = []
    fuzzy: list[tuple[pd.Series, str]] = []
    for _, row in odds.iterrows():
        quote_f1 = str(row.get("fighter_1") or row.get("fighter1") or "")
        quote_f2 = str(row.get("fighter_2") or row.get("fighter2") or "")
        if not _event_compatible(event_id, event_date, row):
            continue
        alignment = pair_alignment(fighter_1, fighter_2, quote_f1, quote_f2)
        if alignment:
            actionable.append((row, alignment))
            continue
        fuzzy_alignment = _fuzzy_pair(fighter_1, fighter_2, quote_f1, quote_f2)
        if fuzzy_alignment and not pair_alignment(fighter_1, fighter_2, quote_f1, quote_f2):
            fuzzy.append((row, fuzzy_alignment))

    if len(actionable) > 1:
        return {
            "status": "rejected",
            "reason": "ambiguous_match",
            "quote_match": "ambiguous",
            "row": None,
        }
    if len(actionable) == 1:
        row, alignment = actionable[0]
        aligned = _align_row(row, reversed_order=alignment == "reversed")
        same_core = _core_name(fighter_1) == _core_name(str(aligned.get("fighter_1") or ""))
        match = "reversed" if alignment == "reversed" else ("exact" if same_core else "alias")
        aligned["quote_match"] = match
        aligned["quote_actionable"] = True
        aligned["quote_block_reason"] = ""
        return {
            "status": "actionable",
            "reason": "",
            "quote_match": match,
            "row": aligned,
        }
    if len(fuzzy) == 1:
        row, alignment = fuzzy[0]
        aligned = _align_row(row, reversed_order=alignment == "reversed")
        aligned["quote_match"] = "advisory_fuzzy"
        aligned["quote_actionable"] = False
        aligned["quote_block_reason"] = "advisory_fuzzy"
        aligned["price_kind"] = "advisory"
        return {
            "status": "advisory",
            "reason": "advisory_fuzzy",
            "quote_match": "advisory_fuzzy",
            "row": aligned,
        }
    if len(fuzzy) > 1:
        return {
            "status": "rejected",
            "reason": "ambiguous_match",
            "quote_match": "ambiguous",
            "row": None,
        }
    return rejected


def _fighters_from_ticket(ticket: dict[str, Any]) -> tuple[str, str]:
    first = str(ticket.get("fighter_1") or "").strip()
    second = str(ticket.get("fighter_2") or "").strip()
    if first and second:
        return first, second
    fight = str(ticket.get("fight") or "")
    parts = [part.strip() for part in fight.split(" vs ", 1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return first, second


def normalized_price_kind(ticket: dict[str, Any]) -> str:
    """Live sizes only an explicit ``book`` price; blank or unknown kinds do not qualify."""
    return str(ticket.get("price_kind") or "").strip().lower()


def _is_consensus(ticket: dict[str, Any]) -> bool:
    kind = normalized_price_kind(ticket)
    if kind in _PRICE_CONSENSUS:
        return True
    try:
        count = ticket.get("bookmaker_count")
        if count is not None and str(count).strip() != "" and float(count) > 1:
            return True
    except (TypeError, ValueError):
        return True
    book = str(ticket.get("bookmaker") or "").strip().lower()
    source = str(ticket.get("odds_source") or "").strip().lower()
    if book in _NOT_A_BOOK or source in _PRICE_CONSENSUS:
        return True
    return False


def live_quote_block_reason(
    ticket: dict[str, Any],
    *,
    now: datetime | None = None,
) -> str:
    """Empty string when this ticket may be sized in Live. Otherwise a fail-closed reason."""
    match = str(ticket.get("quote_match") or "").strip().lower()
    if match in _ADVISORY_MATCHES:
        return match
    if _is_consensus(ticket):
        return "consensus_not_executable"
    if normalized_price_kind(ticket) != "book":
        return "price_kind_not_book"
    book = str(ticket.get("bookmaker") or "").strip()
    if not book:
        return "missing_bookmaker"
    try:
        count = ticket.get("bookmaker_count")
        if count is None or str(count).strip() == "" or float(count) != 1:
            return "not_single_book_quote"
    except (TypeError, ValueError):
        return "not_single_book_quote"

    card_f1, card_f2 = _fighters_from_ticket(ticket)
    # Quote names must be stored on the quote. Card names are not a substitute.
    quote_f1 = str(ticket.get("quote_fighter_1") or "").strip()
    quote_f2 = str(ticket.get("quote_fighter_2") or "").strip()
    if not card_f1 or not card_f2 or not quote_f1 or not quote_f2:
        return "missing_fighters"
    if pair_alignment(card_f1, card_f2, quote_f1, quote_f2) is None:
        return "name_mismatch"
    price_reason = _selection_price_reason(ticket, quote_f1, quote_f2)
    if price_reason:
        return price_reason

    card_id = str(ticket.get("card_event_id") or "").strip()
    quote_id = str(ticket.get("quote_event_id") or "").strip()
    card_date = ticket.get("card_event_date") or ticket.get("event_date")
    identity = event_identity_failure(
        card_id,
        quote_id,
        card_date,
        ticket.get("commence_time"),
    )
    if identity:
        return identity

    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    quoted = parse_quote_time(ticket.get("quote_timestamp"))
    if quoted is None:
        raw = ticket.get("quote_timestamp")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return "missing_timestamp"
        return "invalid_timestamp"
    if quoted - moment > _FUTURE_SKEW:
        return "future_timestamp"
    if moment - quoted > quote_max_age():
        return "expired_quote"
    commence = parse_quote_time(ticket.get("commence_time"))
    if commence is None:
        raw_commence = ticket.get("commence_time")
        if raw_commence is None or (isinstance(raw_commence, str) and not str(raw_commence).strip()):
            return "missing_commence_time"
        return "invalid_timestamp"
    if commence <= moment:
        return "already_started"

    try:
        price = float(ticket.get("decimal_odds") or ticket.get("odds") or 0)
    except (TypeError, ValueError):
        price = 0.0
    if not math.isfinite(price) or price <= 1.0:
        return "missing_price"
    return ""


def _finite_price(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not str(value).strip()):
            return None
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 1.0:
        return None
    return price


def _prices_equal(left: Any, right: Any) -> bool:
    first = _finite_price(left)
    second = _finite_price(right)
    if first is None or second is None:
        return False
    return abs(first - second) <= 0.001


def _selection_text(ticket: dict[str, Any]) -> str:
    return str(ticket.get("pick") or ticket.get("selection") or "").strip()


def _selection_price_reason(ticket: dict[str, Any], quote_f1: str, quote_f2: str) -> str:
    """The sized selection and decimal price must be the quote's, not just the pair."""
    chosen = _selection_text(ticket)
    ticket_price = ticket.get("decimal_odds")
    if _finite_price(ticket_price) is None:
        ticket_price = ticket.get("odds")
    quote_sel = str(ticket.get("quote_selection") or "").strip()
    side_price = None
    if chosen and names_match_actionable(chosen, quote_f1):
        side_price = _finite_price(ticket.get("quote_f1_odds"))
    elif chosen and names_match_actionable(chosen, quote_f2):
        side_price = _finite_price(ticket.get("quote_f2_odds"))
    if side_price is not None:
        if quote_sel and quote_sel.casefold() != chosen.casefold() and not names_match_actionable(
            quote_sel, chosen
        ):
            return "selection_mismatch"
        if not _prices_equal(ticket_price, side_price):
            return "price_mismatch"
        return ""
    quoted = _finite_price(ticket.get("quote_decimal_odds"))
    if quote_sel and quoted is not None:
        if not chosen or (
            chosen.casefold() != quote_sel.casefold() and not names_match_actionable(chosen, quote_sel)
        ):
            return "selection_mismatch"
        if not _prices_equal(ticket_price, quoted):
            return "price_mismatch"
        return ""
    return "price_mismatch"


def clear_actionable_stakes(ticket: dict[str, Any], *, reason: str) -> None:
    """Remove every stake a later allocator or narrative pass could revive."""
    for key in ACTIONABLE_STAKE_FIELDS:
        ticket[key] = 0.0
    ticket["quote_actionable"] = False
    ticket["quote_block_reason"] = reason
    ticket["bet_action"] = "NO BET"


def apply_live_quote_gate(
    tickets: list[dict[str, Any]],
    *,
    live: bool,
    now: datetime | None = None,
) -> None:
    """Live quotes that fail identity or freshness stay $0. Paper stakes are left alone."""
    if not live:
        for ticket in tickets:
            match = str(ticket.get("quote_match") or "").strip().lower()
            if match in _ADVISORY_MATCHES:
                clear_actionable_stakes(ticket, reason=match)
        return
    for ticket in tickets:
        reason = live_quote_block_reason(ticket, now=now)
        if reason:
            clear_actionable_stakes(ticket, reason=reason)


def read_quote_cache(path: Path | str, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a quote file. Reading does not stamp or rewrite quote_timestamp."""
    return pd.read_csv(path, parse_dates=parse_dates or [])


def price_for_selection(prices: dict[str, float], fighter: str) -> float | None:
    """One outcome price for this fighter. Surname or substring hits are not a price."""
    hits = [
        float(px)
        for name, px in prices.items()
        if names_match_actionable(str(name), fighter)
    ]
    if len(hits) != 1 or not math.isfinite(hits[0]) or hits[0] <= 1.0:
        return None
    return hits[0]


def classify_book_price(*, bookmaker_count: int, bookmaker: str) -> dict[str, str]:
    """A multi-book average is consensus, not an executable price from one book."""
    if bookmaker_count == 1 and bookmaker and bookmaker.strip().lower() not in _NOT_A_BOOK:
        return {"bookmaker": bookmaker.strip(), "price_kind": "book", "odds_source": bookmaker.strip()}
    return {"bookmaker": "consensus", "price_kind": "consensus", "odds_source": "consensus"}
