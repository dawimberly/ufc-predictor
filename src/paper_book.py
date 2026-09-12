"""Automated theoretical paper-betting book.

A self-contained paper account that auto-"places" the model's money tickets
(Deep Blue = BET THIS, Sky Blue = TINY PAPER BET), prices and settles them off
The Odds API odds, and tracks a running balance + equity curve.

Design:
- Starts from a fixed bankroll (``config.PAPER_BOOK_START_BANKROLL``, default $1,000).
- Independent of the manual dashboard bankroll and of any real book you bet on
  yourself (you bet MyBookie manually; this book is theoretical, Odds-API priced).
- Fail-closed: never places without a usable Odds API price; props only settle
  when the fight's round data is available; incomplete rows stay open.
- Idempotent: a bet id derived from (event, fight, market, selection) means a
  card can be re-analyzed without double-placing.
- Reads cached odds only — it never pulls The Odds API, so it costs 0 credits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

BOOK_FIELDS = [
    "bet_id",
    "placed_at",
    "event",
    "fight",
    "fight_id",
    "fighter_1",
    "fighter_2",
    "market_type",
    "selection",
    "pick",
    "tier",
    "book",
    "odds",          # decimal, from The Odds API
    "stake",
    "edge_pct",
    "status",        # open | settled
    "result",        # win | loss | push | ""
    "correct",       # 1 | 0 | ""
    "pnl",
    "settled_at",
    "balance_after",
]

_ACTIONABLE_TIERS = {"blue", "sky_blue"}


# --------------------------------------------------------------------------- #
# Paths / small utils
# --------------------------------------------------------------------------- #
def book_csv_path() -> Path:
    return Path(getattr(config, "PAPER_BOOK_CSV", config.DATA_DIR / "paper_book.csv"))


def state_json_path() -> Path:
    return Path(
        getattr(config, "PAPER_BOOK_STATE_JSON", config.DATA_DIR / "paper_book_state.json")
    )


def start_bankroll() -> float:
    try:
        return float(getattr(config, "PAPER_BOOK_START_BANKROLL", 1000.0) or 1000.0)
    except (TypeError, ValueError):
        return 1000.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(val: Any, default: float | None = None) -> float | None:
    try:
        if val is None or val == "":
            return default
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f


def _clean(name: Any) -> str:
    try:
        from src.prediction_bank import _clean as _bank_clean

        return _bank_clean(name)
    except Exception:
        import re

        return re.sub(r"\s+", " ", str(name or "")).strip()


def _fighters_match(a: str, b: str) -> bool:
    try:
        from src.prediction_bank import _fighters_match as _bank_match

        return _bank_match(a, b)
    except Exception:
        return _clean(a).lower() == _clean(b).lower() and bool(_clean(a))


def _last_token(name: str) -> str:
    parts = _clean(name).lower().split()
    return parts[-1] if parts else ""


def _bet_id(event: str, fight_id: str, market: str, selection: str) -> str:
    raw = "|".join(
        [_clean(event).lower(), _clean(fight_id).lower(), str(market).lower(), str(selection).lower()]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Ticket helpers
# --------------------------------------------------------------------------- #
def _ticket_book(ticket: dict[str, Any]) -> str:
    try:
        from src.bet_slip import ticket_book

        return ticket_book(ticket)
    except Exception:
        return str(ticket.get("book") or "").strip().lower()


def _ticket_market(ticket: dict[str, Any]) -> str:
    try:
        from src.bet_slip import ticket_market_type

        return ticket_market_type(ticket)
    except Exception:
        return str(ticket.get("market_type") or "moneyline").strip().lower()


def _ticket_selection(ticket: dict[str, Any]) -> str:
    try:
        from src.bet_slip import ticket_selection

        sel = ticket_selection(ticket)
        if sel:
            return sel
    except Exception:
        pass
    return str(ticket.get("pick") or ticket.get("selection") or "").strip()


def _ticket_fight_id(ticket: dict[str, Any]) -> str:
    try:
        from src.bet_slip import ticket_fight_id

        fid = ticket_fight_id(ticket)
        if fid:
            return fid
    except Exception:
        pass
    return str(ticket.get("fight_id") or ticket.get("fight") or "").strip()


def _decimal_odds(ticket: dict[str, Any]) -> float | None:
    """Decimal odds from the ticket, preferring an explicit decimal price."""
    for key in ("decimal_odds", "odds", "combined_odds"):
        val = _safe_float(ticket.get(key))
        if val is not None and val > 1.0:
            return val
    am = ticket.get("american_odds")
    if am not in (None, "", "-"):
        try:
            from src.parlay_builder import american_to_decimal

            dec = _safe_float(american_to_decimal(am))
            if dec is not None and dec > 1.0:
                return dec
        except Exception:
            pass
    return None


def _parse_fighters(ticket: dict[str, Any]) -> tuple[str, str]:
    f1 = str(ticket.get("fighter_1") or ticket.get("fighter1") or "").strip()
    f2 = str(ticket.get("fighter_2") or ticket.get("fighter2") or "").strip()
    if f1 and f2:
        return f1, f2
    fight = str(ticket.get("fight") or "").strip()
    for sep in (" vs ", " vs. ", " v "):
        if sep in fight:
            a, b = fight.split(sep, 1)
            return a.strip(), b.strip()
    return f1, f2


def is_money_ticket(ticket: dict[str, Any]) -> bool:
    """Deep Blue / Sky Blue with a positive stake — never fun/advisory."""
    tier = str(ticket.get("bet_tier") or ticket.get("tier") or "").strip().lower()
    if tier not in _ACTIONABLE_TIERS:
        return False
    if ticket.get("fun_bet") or ticket.get("advisory"):
        return False
    stake = _safe_float(ticket.get("suggested_stake")) or _safe_float(ticket.get("stake_usd")) or 0.0
    return stake > 0


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def load_book(path: Path | str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else book_csv_path()
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _save_book(rows: list[dict[str, Any]], path: Path | str | None = None) -> None:
    p = Path(path) if path else book_csv_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BOOK_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in BOOK_FIELDS})


# --------------------------------------------------------------------------- #
# Place
# --------------------------------------------------------------------------- #
def place_paper_tickets(
    tickets: list[dict[str, Any]] | None,
    *,
    event: str = "",
    path: Path | str | None = None,
) -> dict[str, int]:
    """Open a paper position for each Odds-API-priced money ticket.

    Idempotent: a ticket already on the book (open or settled) is skipped.
    Returns counts: placed / skipped_existing / skipped_no_odds / skipped_book / considered.
    """
    if not bool(getattr(config, "PAPER_BOOK_ENABLED", True)):
        return {"placed": 0, "skipped_existing": 0, "skipped_no_odds": 0, "skipped_book": 0, "considered": 0}

    p = Path(path) if path else book_csv_path()
    rows = load_book(p)
    existing_ids = {str(r.get("bet_id") or "") for r in rows}
    allowed_books = set(getattr(config, "PAPER_BOOK_ALLOWED_BOOKS", ("odds api", "")))

    placed = skipped_existing = skipped_no_odds = skipped_book = considered = 0
    for ticket in tickets or []:
        if not isinstance(ticket, dict) or not is_money_ticket(ticket):
            continue
        considered += 1

        if _ticket_book(ticket) not in allowed_books:
            skipped_book += 1
            continue

        odds = _decimal_odds(ticket)
        if odds is None:
            skipped_no_odds += 1
            continue

        market = _ticket_market(ticket)
        selection = _ticket_selection(ticket)
        fight_id = _ticket_fight_id(ticket)
        ev = str(ticket.get("event") or event or "").strip()
        bet_id = _bet_id(ev, fight_id, market, selection)
        if bet_id in existing_ids:
            skipped_existing += 1
            continue

        f1, f2 = _parse_fighters(ticket)
        stake = _safe_float(ticket.get("suggested_stake")) or _safe_float(ticket.get("stake_usd")) or 0.0
        rows.append(
            {
                "bet_id": bet_id,
                "placed_at": _utc_now(),
                "event": ev,
                "fight": str(ticket.get("fight") or f"{f1} vs {f2}").strip(),
                "fight_id": fight_id,
                "fighter_1": f1,
                "fighter_2": f2,
                "market_type": market,
                "selection": selection,
                "pick": str(ticket.get("pick") or selection).strip(),
                "tier": str(ticket.get("bet_tier") or ticket.get("tier") or "").strip().lower(),
                "book": _ticket_book(ticket) or "odds api",
                "odds": f"{float(odds):.4f}",
                "stake": f"{float(stake):.2f}",
                "edge_pct": _fmt_edge(ticket),
                "status": "open",
                "result": "",
                "correct": "",
                "pnl": "",
                "settled_at": "",
                "balance_after": "",
            }
        )
        existing_ids.add(bet_id)
        placed += 1

    if placed:
        _save_book(rows, p)
        logger.info("paper_book placed=%d event=%r", placed, event)
    return {
        "placed": placed,
        "skipped_existing": skipped_existing,
        "skipped_no_odds": skipped_no_odds,
        "skipped_book": skipped_book,
        "considered": considered,
    }


def _fmt_edge(ticket: dict[str, Any]) -> str:
    edge = _safe_float(ticket.get("edge_pct"))
    if edge is None:
        e = _safe_float(ticket.get("edge"))
        if e is not None:
            edge = e * 100.0 if abs(e) <= 1.5 else e
    return f"{edge:+.1f}" if edge is not None else ""


# --------------------------------------------------------------------------- #
# Settle
# --------------------------------------------------------------------------- #
def _prop_over_1_5(hit: Any) -> bool | None:
    """Resolve an Over 1.5 rounds prop from a completed fight row.

    Over wins when the fight reaches round 2+ (or goes to decision). Returns
    None when round data is unavailable (fail-closed → stays open).
    """
    def _get(k: str) -> Any:
        try:
            return hit.get(k)
        except Exception:
            return None

    method = str(_get("method") or _get("finish") or "").lower()
    if "dec" in method:  # decision → went the distance → Over
        return True
    for key in ("round", "finish_round", "ending_round", "end_round", "rounds"):
        r = _safe_float(_get(key))
        if r is not None and r > 0:
            return r >= 2
    return None


def settle_paper_book(
    *,
    historical: Any = None,
    path: Path | str | None = None,
) -> dict[str, int]:
    """Match open positions to completed fights, compute PnL off the taken odds,
    and roll the balance/equity forward. Reuses ``settlement.compute_pnl``.
    """
    from src.settlement import compute_pnl

    p = Path(path) if path else book_csv_path()
    rows = load_book(p)
    open_rows = [r for r in rows if str(r.get("status") or "").lower() != "settled"]
    if not open_rows:
        _write_state(rows)
        return {"settled": 0, "open": 0, "matched": 0}

    if historical is None:
        try:
            from src.data_loader import load_fights

            historical = load_fights()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("paper_book settle skipped — cannot load fights: %s", exc)
            return {"settled": 0, "open": len(open_rows), "matched": 0}

    if historical is None or getattr(historical, "empty", True) or "winner" not in historical.columns:
        return {"settled": 0, "open": len(open_rows), "matched": 0}

    f1c = "fighter_1" if "fighter_1" in historical.columns else "fighter1"
    f2c = "fighter_2" if "fighter_2" in historical.columns else "fighter2"
    hist = historical.dropna(subset=["winner"]).copy()
    hist_index: dict[tuple[str, str], list[Any]] = {}
    for _, h in hist.iterrows():
        a, b = _last_token(str(h[f1c])), _last_token(str(h[f2c]))
        if not a or not b:
            continue
        hist_index.setdefault(tuple(sorted((a, b))), []).append(h)

    settled = matched = 0
    for row in rows:
        if str(row.get("status") or "").lower() == "settled":
            continue
        f1, f2 = str(row.get("fighter_1") or ""), str(row.get("fighter_2") or "")
        candidates = hist_index.get(tuple(sorted((_last_token(f1), _last_token(f2)))), [])
        hit = None
        for h in candidates:
            hf1, hf2 = str(h[f1c]), str(h[f2c])
            if (_fighters_match(f1, hf1) and _fighters_match(f2, hf2)) or (
                _fighters_match(f1, hf2) and _fighters_match(f2, hf1)
            ):
                hit = h
                break
        if hit is None:
            continue
        matched += 1

        market = str(row.get("market_type") or "moneyline").lower()
        correct: bool | None
        if market == "prop":
            correct = _prop_over_1_5(hit)
        else:
            actual = str(hit.get("winner") or "")
            correct = _fighters_match(str(row.get("pick") or ""), actual) if actual else None
        if correct is None:
            # fail-closed: not enough data to resolve yet — leave open
            continue

        stake = _safe_float(row.get("stake"))
        odds = _safe_float(row.get("odds"))
        pnl = compute_pnl(correct=bool(correct), stake=stake, opening_odds=odds)
        if pnl is None:
            continue

        row["status"] = "settled"
        row["result"] = "win" if correct else "loss"
        row["correct"] = "1" if correct else "0"
        row["pnl"] = f"{pnl:.4f}"
        row["settled_at"] = _utc_now()
        settled += 1

    state = _write_state(rows)
    if settled:
        _save_book(rows, p)
    logger.info("paper_book settled=%d matched=%d balance=%.2f", settled, matched, state["balance"])
    return {"settled": settled, "open": len(open_rows) - settled, "matched": matched}


# --------------------------------------------------------------------------- #
# State / summary / equity
# --------------------------------------------------------------------------- #
def _write_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _compute_summary(rows)
    try:
        sp = state_json_path()
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("paper_book state write failed: %s", exc)
    return summary


def _compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = start_bankroll()
    settled = [r for r in rows if str(r.get("status") or "").lower() == "settled"]
    open_rows = [r for r in rows if str(r.get("status") or "").lower() != "settled"]

    # Settle in chronological order to build the equity curve.
    settled.sort(key=lambda r: str(r.get("settled_at") or ""))
    balance = start
    realized_pnl = 0.0
    settled_stake = 0.0
    wins = 0
    equity_curve: list[dict[str, Any]] = [{"at": "start", "balance": round(start, 2)}]
    for r in settled:
        pnl = _safe_float(r.get("pnl")) or 0.0
        realized_pnl += pnl
        balance += pnl
        settled_stake += _safe_float(r.get("stake")) or 0.0
        if str(r.get("correct")) == "1":
            wins += 1
        r["balance_after"] = f"{balance:.2f}"
        equity_curve.append({"at": r.get("settled_at") or "", "balance": round(balance, 2)})

    open_exposure = sum(_safe_float(r.get("stake")) or 0.0 for r in open_rows)
    n_settled = len(settled)
    return {
        "start_bankroll": round(start, 2),
        "balance": round(balance, 2),
        "realized_pnl": round(realized_pnl, 2),
        "roi_pct": round(100.0 * realized_pnl / settled_stake, 2) if settled_stake > 0 else 0.0,
        "open_count": len(open_rows),
        "open_exposure": round(open_exposure, 2),
        "settled_count": n_settled,
        "wins": wins,
        "losses": n_settled - wins,
        "hit_rate_pct": round(100.0 * wins / n_settled, 1) if n_settled else 0.0,
        "equity_curve": equity_curve,
        "updated_at": _utc_now(),
    }


def paper_book_summary(path: Path | str | None = None) -> dict[str, Any]:
    """Totals for the paper account (balance, ROI, hit rate, equity curve)."""
    return _compute_summary(load_book(path))


def open_positions(path: Path | str | None = None) -> list[dict[str, Any]]:
    return [r for r in load_book(path) if str(r.get("status") or "").lower() != "settled"]


def settled_positions(path: Path | str | None = None) -> list[dict[str, Any]]:
    return [r for r in load_book(path) if str(r.get("status") or "").lower() == "settled"]


# --------------------------------------------------------------------------- #
# Adapter for the analysis pipeline
# --------------------------------------------------------------------------- #
def money_tickets_from_result(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Best-effort extraction of sized money tickets from a full-analysis result."""
    if not isinstance(result, dict):
        return []
    tickets: list[dict[str, Any]] = []
    alerts = result.get("alerts") if isinstance(result.get("alerts"), dict) else {}
    for key in ("singles", "prop_singles", "items", "bet_slip"):
        seq = alerts.get(key) if isinstance(alerts, dict) else None
        if seq is None:
            seq = result.get(key)
        if isinstance(seq, list):
            tickets.extend(t for t in seq if isinstance(t, dict))
    return [t for t in tickets if is_money_ticket(t)]


def auto_run_from_result(
    result: dict[str, Any] | None,
    *,
    event: str = "",
) -> dict[str, Any]:
    """Place this card's money tickets then settle any finished ones (fail-soft)."""
    if not bool(getattr(config, "PAPER_BOOK_ENABLED", True)):
        return {"enabled": False}
    out: dict[str, Any] = {"enabled": True}
    try:
        ev = event or str((result or {}).get("event_name") or (result or {}).get("event") or "")
        out["placed"] = place_paper_tickets(money_tickets_from_result(result), event=ev)
    except Exception as exc:  # pragma: no cover - fail-soft
        logger.warning("paper_book auto-place failed: %s", exc)
        out["place_error"] = str(exc)
    try:
        out["settled"] = settle_paper_book()
    except Exception as exc:  # pragma: no cover - fail-soft
        logger.warning("paper_book auto-settle failed: %s", exc)
        out["settle_error"] = str(exc)
    return out
