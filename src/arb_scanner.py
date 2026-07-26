"""Cross-book moneyline and totals arbitrage scanner."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

import config
from src.predictor import _fighter_name_key, _names_match

BOOK_ML_FETCHERS: dict[str, tuple[str, str]] = {
    "BetNow.eu": ("src.odds_providers.betnow_scraper", "fetch_betnow_odds"),
    "DraftKings": ("src.odds_providers.draftkings", "fetch_draftkings_odds"),
    "MyBookie": ("src.odds_providers.mybookie_scraper", "fetch_mybookie_odds"),
}

# Opposite sides for round totals (1.5 line).
_TOTALS_OVER_KEYS = frozenset({"over_1_5_rounds"})
_TOTALS_UNDER_KEYS = frozenset({"round_1_finish", "under_1_5_rounds"})


def fight_pair_key(fighter_a: str, fighter_b: str) -> tuple[str, str]:
    ka, kb = _fighter_name_key(fighter_a), _fighter_name_key(fighter_b)
    return tuple(sorted((ka, kb)))


def _canonical_fighter(name: str, existing: dict[str, str]) -> str:
    key = _fighter_name_key(name)
    for k, display in existing.items():
        if _names_match(k, name) or k == key:
            return display
    existing[key] = str(name).strip()
    return existing[key]


def _american_from_decimal(decimal: float) -> str:
    try:
        decimal = float(decimal)
    except (TypeError, ValueError):
        return "-"
    if decimal != decimal or decimal <= 1:  # NaN or invalid
        return "-"
    if decimal >= 2.0:
        return f"+{int(round((decimal - 1) * 100))}"
    return str(int(round(-100 / (decimal - 1))))


def arb_math(odds_a: float, odds_b: float, *, stake_total: float) -> dict[str, float]:
    """Stake split and profit for a two-outcome market."""
    if odds_a <= 1 or odds_b <= 1:
        return {
            "inv_sum": 999.0,
            "overround_pct": 999.0,
            "profit_pct": 0.0,
            "stake_a": 0.0,
            "stake_b": 0.0,
            "payout": 0.0,
        }
    inv = 1.0 / odds_a + 1.0 / odds_b
    stake_a = stake_total * (1.0 / odds_a) / inv
    stake_b = stake_total * (1.0 / odds_b) / inv
    payout = stake_a * odds_a
    profit_pct = (payout / stake_total - 1.0) * 100.0 if stake_total > 0 else 0.0
    return {
        "inv_sum": inv,
        "overround_pct": max(0.0, (inv - 1.0) * 100.0),
        "profit_pct": profit_pct if inv < 1.0 else 0.0,
        "stake_a": stake_a,
        "stake_b": stake_b,
        "payout": payout,
    }


def _ml_quote_rows(df: pd.DataFrame, book: str) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        f1 = str(row.get("fighter_1", "")).strip()
        f2 = str(row.get("fighter_2", "")).strip()
        try:
            o1 = float(row.get("f1_odds", 0) or 0)
            o2 = float(row.get("f2_odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not f1 or not f2 or o1 <= 1 or o2 <= 1:
            continue
        out.append(
            {
                "book": book,
                "fighter_1": f1,
                "fighter_2": f2,
                "f1_odds": o1,
                "f2_odds": o2,
                "pair_key": fight_pair_key(f1, f2),
            }
        )
    return out


def collect_moneyline_quotes(
    *,
    books: dict[str, Any] | None = None,
    force_refresh: bool = False,
    budget_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Gather ML quotes from book payloads and/or live fetches."""
    import importlib

    quotes: list[dict[str, Any]] = []
    errors: list[str] = []
    enabled = config.enabled_books_from_budget(budget_state) if budget_state else None

    def _book_allowed(book_name: str) -> bool:
        if enabled is not None and book_name not in enabled and book_name != "Odds API":
            return False
        if book_name == "MyBookie" and not config.MYBOOKIE_ENABLED:
            return False
        if book_name == "BetNow.eu" and not getattr(config, "BETNOW_ENABLED", False):
            return False
        if book_name == "DraftKings" and not getattr(config, "DRAFTKINGS_ENABLED", False):
            return False
        return True

    if books:
        for book_name, payload in books.items():
            if book_name == "Overview":
                continue
            if not _book_allowed(book_name):
                continue
            preds = payload.get("predictions") if isinstance(payload, dict) else None
            if isinstance(preds, pd.DataFrame) and not preds.empty:
                if "f1_odds" in preds.columns and preds["f1_odds"].notna().any():
                    quotes.extend(_ml_quote_rows(preds, book_name))
                    continue
            # Prefer payload odds — never live-fetch when ODDS_FETCH_ONCE (arb was burning DK credits).
            if bool(getattr(config, "ODDS_FETCH_ONCE", True)):
                continue
            mod_path, fn_name = BOOK_ML_FETCHERS.get(book_name, ("", ""))
            if not mod_path:
                continue
            try:
                mod = importlib.import_module(mod_path)
                odds_df = getattr(mod, fn_name)(force_refresh=force_refresh)
                quotes.extend(_ml_quote_rows(odds_df, book_name))
            except Exception as exc:
                errors.append(f"{book_name}: {exc}")

    if not quotes and not bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        for book_name, (mod_path, fn_name) in BOOK_ML_FETCHERS.items():
            if not _book_allowed(book_name):
                continue
            try:
                mod = importlib.import_module(mod_path)
                odds_df = getattr(mod, fn_name)(force_refresh=force_refresh)
                quotes.extend(_ml_quote_rows(odds_df, book_name))
            except Exception as exc:
                errors.append(f"{book_name}: {exc}")

    return quotes, errors


def _prop_quote_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prop_key = str(row.get("prop_key", "")).strip()
        if prop_key not in _TOTALS_OVER_KEYS and prop_key not in _TOTALS_UNDER_KEYS:
            continue
        try:
            dec = float(row.get("decimal_odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if dec <= 1:
            continue
        f1 = str(row.get("fighter_1", "")).strip()
        f2 = str(row.get("fighter_2", "")).strip()
        if not f1 or not f2:
            continue
        side = "over" if prop_key in _TOTALS_OVER_KEYS else "under"
        out.append(
            {
                "book": str(row.get("bookmaker", "")).strip() or "Unknown",
                "fighter_1": f1,
                "fighter_2": f2,
                "pair_key": fight_pair_key(f1, f2),
                "side": side,
                "prop_key": prop_key,
                "selection": str(row.get("selection", prop_key)),
                "decimal_odds": dec,
            }
        )
    return out


def collect_totals_quotes(
    *,
    force_refresh: bool = False,
    budget_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Gather O/U 1.5 round quotes from live prop feeds."""
    from src.props import fetch_live_prop_odds

    quotes: list[dict[str, Any]] = []
    errors: list[str] = []
    if not config.ENABLE_PROPS:
        return quotes, errors

    # ODDS_FETCH_ONCE: never trigger DraftKings per-event Odds API pulls from arb scan.
    if force_refresh and bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        force_refresh = False

    enabled = config.enabled_books_from_budget(budget_state) if budget_state else None
    for book in ("Odds API", "BetNow.eu", "DraftKings", "MyBookie"):
        if book != "Odds API" and enabled is not None and book not in enabled:
            continue
        if book == "MyBookie" and not config.MYBOOKIE_ENABLED:
            continue
        if book == "BetNow.eu" and not getattr(config, "BETNOW_ENABLED", False):
            continue
        if book == "DraftKings" and not getattr(config, "DRAFTKINGS_ENABLED", False):
            continue
        try:
            prop_df = fetch_live_prop_odds(book, force_refresh=force_refresh)
            quotes.extend(_prop_quote_rows(prop_df))
        except Exception as exc:
            errors.append(f"{book} props: {exc}")
    return quotes, errors


def _card_pair_keys(combined: pd.DataFrame | None) -> set[tuple[str, str]] | None:
    if combined is None or combined.empty:
        return None
    keys: set[tuple[str, str]] = set()
    for _, row in combined.iterrows():
        f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
        f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
        if f1 and f2:
            keys.add(fight_pair_key(f1, f2))
    return keys or None


def scan_moneyline_arbs(
    quotes: list[dict[str, Any]],
    *,
    card_keys: set[tuple[str, str]] | None = None,
    near_margin_pct: float | None = None,
    stake_total: float | None = None,
) -> list[dict[str, Any]]:
    """Find best cross-book ML prices per fight."""
    near = config.ARB_NEAR_MARGIN_PCT if near_margin_pct is None else near_margin_pct
    stake = config.ARB_STAKE_TOTAL if stake_total is None else stake_total

    by_fight: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for q in quotes:
        key = q["pair_key"]
        if card_keys is not None and key not in card_keys:
            continue
        by_fight.setdefault(key, []).append(q)

    rows: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}

    for pair_key, fight_quotes in by_fight.items():
        if len(fight_quotes) < 2:
            continue

        sides: dict[str, list[tuple[str, float, str]]] = {}
        display_f1 = ""
        display_f2 = ""

        for q in fight_quotes:
            f1 = _canonical_fighter(q["fighter_1"], name_map)
            f2 = _canonical_fighter(q["fighter_2"], name_map)
            display_f1, display_f2 = f1, f2
            k1, k2 = _fighter_name_key(f1), _fighter_name_key(f2)
            sides.setdefault(k1, []).append((q["book"], float(q["f1_odds"]), f1))
            sides.setdefault(k2, []).append((q["book"], float(q["f2_odds"]), f2))

        if len(sides) != 2:
            continue

        fighter_keys = list(sides.keys())
        best_a = max(sides[fighter_keys[0]], key=lambda x: x[1])
        best_b = max(sides[fighter_keys[1]], key=lambda x: x[1])

        math = arb_math(best_a[1], best_b[1], stake_total=stake)
        is_arb = math["inv_sum"] < 1.0
        is_near = not is_arb and math["overround_pct"] <= near

        if not is_arb and not is_near:
            continue

        rows.append(
            {
                "market": "moneyline",
                "fight": f"{display_f1} vs {display_f2}",
                "fighter_a": best_a[2],
                "fighter_b": best_b[2],
                "side_a": {
                    "fighter": best_a[2],
                    "book": best_a[0],
                    "odds": best_a[1],
                    "american": _american_from_decimal(best_a[1]),
                },
                "side_b": {
                    "fighter": best_b[2],
                    "book": best_b[0],
                    "odds": best_b[1],
                    "american": _american_from_decimal(best_b[1]),
                },
                "is_arb": is_arb,
                "is_near": is_near,
                "overround_pct": math["overround_pct"],
                "profit_pct": math["profit_pct"],
                "stake_a": math["stake_a"],
                "stake_b": math["stake_b"],
                "payout": math["payout"],
                "stake_total": stake,
            }
        )

    rows.sort(key=lambda r: (not r["is_arb"], r["overround_pct"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def scan_totals_arbs(
    quotes: list[dict[str, Any]],
    *,
    card_keys: set[tuple[str, str]] | None = None,
    near_margin_pct: float | None = None,
    stake_total: float | None = None,
) -> list[dict[str, Any]]:
    """Find cross-book O/U 1.5 round arbs (Over vs Under)."""
    near = config.ARB_NEAR_MARGIN_PCT if near_margin_pct is None else near_margin_pct
    stake = config.ARB_STAKE_TOTAL if stake_total is None else stake_total

    by_fight: dict[tuple[str, str], dict[str, list[tuple[str, float, str]]]] = {}
    display_names: dict[tuple[str, str], tuple[str, str]] = {}

    for q in quotes:
        key = q["pair_key"]
        if card_keys is not None and key not in card_keys:
            continue
        side = q["side"]
        by_fight.setdefault(key, {"over": [], "under": []})
        by_fight[key][side].append((q["book"], float(q["decimal_odds"]), q["selection"]))
        if key not in display_names:
            display_names[key] = (q["fighter_1"], q["fighter_2"])

    rows: list[dict[str, Any]] = []
    for key, sides in by_fight.items():
        if not sides["over"] or not sides["under"]:
            continue
        best_over = max(sides["over"], key=lambda x: x[1])
        best_under = max(sides["under"], key=lambda x: x[1])
        f1, f2 = display_names.get(key, ("", ""))
        math = arb_math(best_over[1], best_under[1], stake_total=stake)
        is_arb = math["inv_sum"] < 1.0
        is_near = not is_arb and math["overround_pct"] <= near
        if not is_arb and not is_near:
            continue
        rows.append(
            {
                "market": "totals_1_5",
                "fight": f"{f1} vs {f2}",
                "fighter_a": best_over[2],
                "fighter_b": best_under[2],
                "side_a": {
                    "fighter": "Over 1.5",
                    "book": best_over[0],
                    "odds": best_over[1],
                    "american": _american_from_decimal(best_over[1]),
                },
                "side_b": {
                    "fighter": "Under 1.5",
                    "book": best_under[0],
                    "odds": best_under[1],
                    "american": _american_from_decimal(best_under[1]),
                },
                "is_arb": is_arb,
                "is_near": is_near,
                "overround_pct": math["overround_pct"],
                "profit_pct": math["profit_pct"],
                "stake_a": math["stake_a"],
                "stake_b": math["stake_b"],
                "payout": math["payout"],
                "stake_total": stake,
            }
        )

    rows.sort(key=lambda r: (not r["is_arb"], r["overround_pct"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def arb_row_alert_key(row: dict[str, Any]) -> str:
    """Stable key for deduping arb alert notifications."""
    sa = row.get("side_a") or {}
    sb = row.get("side_b") or {}
    return "|".join(
        [
            str(row.get("market", "")),
            str(row.get("fight", "")),
            str(sa.get("book", "")),
            str(sb.get("book", "")),
            f"{float(sa.get('odds', 0) or 0):.3f}",
            f"{float(sb.get('odds', 0) or 0):.3f}",
        ]
    )


def arb_row_profit_pct(row: dict[str, Any]) -> float:
    if not row.get("is_arb"):
        return 0.0
    return float(row.get("profit_pct", 0) or 0)


def is_dk_mybookie_row(row: dict[str, Any]) -> bool:
    """True when one side is DraftKings and the other is MyBookie."""
    sa = row.get("side_a") or {}
    sb = row.get("side_b") or {}
    books = {str(sa.get("book", "")), str(sb.get("book", ""))}
    return "DraftKings" in books and "MyBookie" in books


def strong_arb_rows(
    scan: dict[str, Any],
    *,
    threshold_pct: float | None = None,
    dk_mybookie_only: bool = False,
) -> list[dict[str, Any]]:
    """Rows with true arb profit at or above the alert threshold."""
    floor = config.ARB_ALERT_THRESHOLD_PCT if threshold_pct is None else threshold_pct
    rows = list(scan.get("moneyline") or []) + list(scan.get("props") or [])
    out: list[dict[str, Any]] = []
    for row in rows:
        if arb_row_profit_pct(row) < floor:
            continue
        if dk_mybookie_only and not is_dk_mybookie_row(row):
            continue
        out.append(row)
    out.sort(key=lambda r: arb_row_profit_pct(r), reverse=True)
    return out


def format_arb_alert_message(row: dict[str, Any]) -> str:
    """Human-readable toast body for a strong arb row."""
    sa = row.get("side_a") or {}
    sb = row.get("side_b") or {}
    profit = arb_row_profit_pct(row)
    market = "Moneyline" if row.get("market") == "moneyline" else "O/U 1.5"
    return (
        f"{market} arb +{profit:.2f}% — {row.get('fight', '')}\n"
        f"{sa.get('fighter', '')} @ {str(sa.get('book', '')).replace('.eu', '')} "
        f"({sa.get('american', '')})\n"
        f"{sb.get('fighter', '')} @ {str(sb.get('book', '')).replace('.eu', '')} "
        f"({sb.get('american', '')})\n"
        f"Stakes ${float(row.get('stake_a', 0)):.0f} / ${float(row.get('stake_b', 0)):.0f} "
        f"(on ${float(row.get('stake_total', config.ARB_STAKE_TOTAL)):.0f} total)"
    )


def scan_cross_book_arbs(
    *,
    books: dict[str, Any] | None = None,
    combined: pd.DataFrame | None = None,
    force_refresh: bool = False,
    budget_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full arb scan for loaded card — moneyline + totals when props enabled."""
    card_keys = _card_pair_keys(combined)
    ml_quotes, ml_errors = collect_moneyline_quotes(
        books=books,
        force_refresh=force_refresh,
        budget_state=budget_state,
    )
    moneyline = scan_moneyline_arbs(ml_quotes, card_keys=card_keys)

    props: list[dict[str, Any]] = []
    prop_errors: list[str] = []
    if config.ENABLE_PROPS:
        totals_quotes, prop_errors = collect_totals_quotes(
            force_refresh=force_refresh,
            budget_state=budget_state,
        )
        props = scan_totals_arbs(totals_quotes, card_keys=card_keys)

    books_scanned = sorted({q["book"] for q in ml_quotes})
    true_arbs = sum(1 for r in moneyline + props if r.get("is_arb"))
    near = sum(1 for r in moneyline + props if r.get("is_near"))

    return {
        "moneyline": moneyline,
        "props": props,
        "meta": {
            "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "books_scanned": books_scanned,
            "ml_quotes": len(ml_quotes),
            "true_arb_count": true_arbs,
            "near_count": near,
            "stake_total": config.ARB_STAKE_TOTAL,
            "near_margin_pct": config.ARB_NEAR_MARGIN_PCT,
        },
        "errors": ml_errors + prop_errors,
    }
