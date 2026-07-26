"""Gane eye-poke / foul scenario analysis for Pereira vs Gane."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.predictor import _names_match
from src.props import fetch_live_prop_odds, method_probs_from_row

GANE_NAME = "Ciryl Gane"
PEREIRA_NAME = "Alex Pereira"

BOOKS: tuple[str, ...] = ("BetNow.eu", "DraftKings", "MyBookie")

# Closest live proxy for a foul/eye-poke Gane win (books rarely list the exact market).
PRIMARY_PROP_KEY = "moneyline"
PRIMARY_PROP_LABEL = "Ciryl Gane Moneyline"
METHOD_PROP_KEY = "fighter_ko"
METHOD_PROP_LABEL = "Ciryl Gane by KO/TKO (Yes)"

ASPINALL_EXPLANATION = (
    "At UFC 321 (Aspinall vs Gane), the bout ended in Round 1 when Tom Aspinall could not "
    "continue after an accidental eye poke. Ciryl Gane was awarded the win — a rare "
    "injury/foul stoppage, not a clean KO. Books almost never post 'win by eye poke'; "
    "Gane moneyline is the broadest proxy (any Gane win). Where listed, 'Gane by KO/TKO' "
    "(sometimes including DQ) is the closest method market — verify your book's rules."
)

SPECULATIVE_STAKE_USD = 2.50


def _clip_prob(p: float, lo: float = 0.03, hi: float = 0.97) -> float:
    return float(max(lo, min(hi, p)))


def _fighter_rate(row: pd.Series, prefix: str, key: str, default: float) -> float:
    for col in (f"{prefix}_{key}", f"f1_{key}" if prefix == "f1" else f"f2_{key}"):
        if col in row.index and pd.notna(row.get(col)):
            return float(row[col])
    return default


def find_pereira_gane_row(predictions: pd.DataFrame) -> pd.Series | None:
    """Locate Alex Pereira vs Ciryl Gane in a predictions table."""
    if predictions is None or predictions.empty:
        return None
    for _, row in predictions.iterrows():
        f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
        f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
        names = {f1.lower(), f2.lower()}
        if GANE_NAME.lower() in names and PEREIRA_NAME.lower() in names:
            return row
    return None


def _gane_side(row: pd.Series) -> str:
    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    return "f1" if _names_match(f1, GANE_NAME) else "f2"


def gane_win_probability(row: pd.Series) -> float:
    side = _gane_side(row)
    p1 = float(row.get("prob_f1_win", row.get("predicted_prob", 0.5)) or 0.5)
    if pd.notna(row.get("prob_f2_win")):
        p2 = float(row["prob_f2_win"])
    else:
        p2 = 1.0 - p1
    return _clip_prob(p2 if side == "f2" else p1, 0.05, 0.85)


def gane_ko_probability(row: pd.Series) -> float:
    """Joint prob: Gane wins AND fight ends by KO/TKO (method proxy for foul stoppage)."""
    side = _gane_side(row)
    win_p = gane_win_probability(row)
    if side == "f1":
        ko_rate = _clip_prob(_fighter_rate(row, "f1", "ko_rate", 0.18), 0.05, 0.55)
    else:
        ko_rate = _clip_prob(_fighter_rate(row, "f2", "ko_rate", 0.18), 0.05, 0.55)
    return _clip_prob(win_p * ko_rate, 0.02, 0.35)


def _format_american(decimal: float | None) -> str:
    if decimal is None or decimal <= 1:
        return "—"
    if decimal >= 2.0:
        am = int(round((decimal - 1.0) * 100))
        return f"+{am}"
    am = int(round(-100.0 / (decimal - 1.0)))
    return str(am)


def _ml_odds_from_row(row: pd.Series | None, gane_side: str) -> float | None:
    if row is None:
        return None
    key = "f1_odds" if gane_side == "f1" else "f2_odds"
    val = row.get(key)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        dec = float(val)
        return dec if dec > 1.0 else None
    except (TypeError, ValueError):
        return None


def _lookup_gane_ko_prop(prop_odds: pd.DataFrame, row: pd.Series) -> dict[str, Any] | None:
    if prop_odds is None or prop_odds.empty:
        return None
    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
    for _, prow in prop_odds.iterrows():
        pf1 = str(prow.get("fighter_1", ""))
        pf2 = str(prow.get("fighter_2", ""))
        aligned = _names_match(f1, pf1) and _names_match(f2, pf2)
        swapped = _names_match(f1, pf2) and _names_match(f2, pf1)
        if not aligned and not swapped:
            continue
        if str(prow.get("prop_key", "")) != METHOD_PROP_KEY:
            continue
        sel = str(prow.get("selection", ""))
        if GANE_NAME.lower() not in sel.lower():
            continue
        dec = float(prow.get("decimal_odds", 0) or 0)
        if dec <= 1:
            continue
        implied = float(prow.get("implied_prob") or (1.0 / dec))
        return {
            "decimal_odds": dec,
            "american_odds": prow.get("american_odds"),
            "implied_prob": implied,
            "selection": sel,
            "prop_label": METHOD_PROP_LABEL,
        }
    return None


def _quote_edge(model_prob: float, decimal: float | None) -> tuple[float | None, float | None]:
    if decimal is None or decimal <= 1:
        return None, None
    implied = 1.0 / decimal
    return model_prob, model_prob - implied


def build_gane_foul_scenario(
    *,
    overview_predictions: pd.DataFrame | None = None,
    books: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build dashboard payload for the Gane foul/eye-poke speculative scenario.

    Uses overview predictions for model probs; per-book tables for ML and prop lines.
    """
    books = books or {}
    preds_df = overview_predictions if overview_predictions is not None else pd.DataFrame()
    base_row = find_pereira_gane_row(preds_df)
    if base_row is None:
        for book_data in books.values():
            preds = book_data.get("predictions")
            if isinstance(preds, pd.DataFrame):
                base_row = find_pereira_gane_row(preds)
                if base_row is not None:
                    break

    if base_row is None:
        return {
            "found": False,
            "fight_label": f"{PEREIRA_NAME} vs {GANE_NAME}",
            "message": "Pereira vs Gane not on the loaded card — refresh when the fight is listed.",
        }

    gane_side = _gane_side(base_row)
    gane_ml_prob = gane_win_probability(base_row)
    gane_ko_prob = gane_ko_probability(base_row)
    fight_label = (
        f"{base_row.get('fighter_1', PEREIRA_NAME)} vs {base_row.get('fighter_2', GANE_NAME)}"
    )

    book_quotes: dict[str, dict[str, Any]] = {}
    for book in BOOKS:
        book_data = books.get(book, {})
        preds = book_data.get("predictions")
        brow = find_pereira_gane_row(preds) if isinstance(preds, pd.DataFrame) else None
        ml_dec = _ml_odds_from_row(brow, gane_side)

        prop_odds = fetch_live_prop_odds(book, force_refresh=False)
        ko_prop = _lookup_gane_ko_prop(prop_odds, base_row)

        book_quotes[book] = {
            "moneyline_decimal": ml_dec,
            "moneyline_american": _format_american(ml_dec),
            "method_decimal": ko_prop.get("decimal_odds") if ko_prop else None,
            "method_american": (
                _format_american(ko_prop["decimal_odds"])
                if ko_prop and ko_prop.get("decimal_odds")
                else "—"
            ),
            "method_selection": ko_prop.get("selection", "") if ko_prop else "",
            "method_available": ko_prop is not None,
        }

    # Pick best proxy: prefer ML (covers foul wins); fall back to method prop if ML missing.
    candidates: list[dict[str, Any]] = []
    for book, q in book_quotes.items():
        if q.get("moneyline_decimal"):
            dec = float(q["moneyline_decimal"])
            model_p, edge = _quote_edge(gane_ml_prob, dec)
            candidates.append(
                {
                    "book": book,
                    "prop_key": PRIMARY_PROP_KEY,
                    "prop_label": PRIMARY_PROP_LABEL,
                    "decimal_odds": dec,
                    "american": _format_american(dec),
                    "model_prob": model_p,
                    "edge": edge,
                    "score": (edge or -999) + 0.001 * dec,
                }
            )
        if q.get("method_decimal"):
            dec = float(q["method_decimal"])
            model_p, edge = _quote_edge(gane_ko_prob, dec)
            candidates.append(
                {
                    "book": book,
                    "prop_key": METHOD_PROP_KEY,
                    "prop_label": METHOD_PROP_LABEL,
                    "decimal_odds": dec,
                    "american": q.get("method_american", _format_american(dec)),
                    "model_prob": model_p,
                    "edge": edge,
                    "score": (edge or -999) + 0.002 * dec,
                }
            )

    if candidates:
        ml_candidates = [c for c in candidates if c["prop_key"] == PRIMARY_PROP_KEY]
        if ml_candidates:
            best = max(ml_candidates, key=lambda c: c["decimal_odds"])
        else:
            best = max(candidates, key=lambda c: c["decimal_odds"])
    else:
        probs = method_probs_from_row(base_row)
        synth_ml = max(1.05, 1.0 / max(gane_ml_prob * 1.08, 0.05))
        best = {
            "book": "Model (synthetic)",
            "prop_key": PRIMARY_PROP_KEY,
            "prop_label": PRIMARY_PROP_LABEL,
            "decimal_odds": synth_ml,
            "american": _format_american(synth_ml),
            "model_prob": gane_ml_prob,
            "edge": gane_ml_prob - (1.0 / synth_ml),
            "score": 0.0,
        }
        for book in BOOKS:
            book_quotes.setdefault(book, {})
            book_quotes[book].setdefault("moneyline_decimal", None)

    return {
        "found": True,
        "fight_label": fight_label,
        "event_name": str(base_row.get("event_name", base_row.get("event", ""))),
        "gane_ml_prob": gane_ml_prob,
        "gane_ko_prob": gane_ko_prob,
        "best_bet": best,
        "book_quotes": book_quotes,
        "explanation": ASPINALL_EXPLANATION,
        "suggested_stake_usd": SPECULATIVE_STAKE_USD,
        "stake_range": "$2–$3",
        "risk_label": "HIGH RISK / SPECULATIVE",
    }
