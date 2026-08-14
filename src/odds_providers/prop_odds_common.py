"""Shared helpers for live UFC prop odds ingestion."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.predictor import _implied_probs, _names_match, _to_decimal_odds

PROP_ODDS_COLUMNS = [
    "fighter_1",
    "fighter_2",
    "prop_key",
    "selection",
    "decimal_odds",
    "implied_prob",
    "american_odds",
    "market_key",
    "point",
    "rotation",
    "bookmaker",
    "odds_source",
]


def parse_american_odds(text: str) -> float | None:
    """Parse +450 / -110 from text."""
    if not text:
        return None
    clean = str(text).strip().replace("\u2212", "-")
    m = re.search(r"(?<![0-9T])([+-]\d{2,4})(?![0-9])", clean)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def american_to_decimal(american: float) -> float:
    if american >= 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def implied_from_decimal(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return np.nan
    return float(1.0 / decimal_odds)


def normalize_totals_point(point: float | None) -> float | None:
    if point is None or not np.isfinite(point):
        return None
    return float(point)


def map_rounds_total(side: str, point: float | None) -> tuple[str, str] | None:
    """
    Map Over/Under + rounds line to (prop_key, selection).

    Uses the actual totals point (1.5 / 2.5 / 3.5 / 4.5). Callers must not
    hard-code Over 1.5 when ``data-points`` / API point is something else.
    """
    pt = normalize_totals_point(point)
    if pt is None:
        return None
    side_l = str(side or "").strip().lower()
    if side_l in ("o", "over"):
        side_l = "over"
    elif side_l in ("u", "under"):
        side_l = "under"
    else:
        return None

    # Snap common MMA half-round lines
    for candidate in (1.5, 2.5, 3.5, 4.5, 0.5):
        if abs(pt - candidate) < 0.01:
            pt = candidate
            break

    pt_tag = f"{pt:.1f}".replace(".", "_")  # 2.5 -> 2_5
    if side_l == "over":
        return f"over_{pt_tag}_rounds", f"Over {pt:g}"
    return f"under_{pt_tag}_rounds", f"Under {pt:g}"


def remap_totals_prop_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Fix rows where selection/prop_key say 1.5 but ``point`` is a different line."""
    if df is None or df.empty or "point" not in df.columns:
        return df
    out = df.copy()
    for idx, row in out.iterrows():
        market = str(row.get("market_key", "") or "").lower()
        prop_key = str(row.get("prop_key", "") or "")
        if market and market != "totals" and "round" not in prop_key:
            continue
        if prop_key in ("fighter_ko", "fighter_sub", "fighter_decision", "goes_to_decision", "finish", "ko_tko", "submission"):
            continue
        pt = normalize_totals_point(row.get("point"))
        if pt is None:
            continue
        sel = str(row.get("selection", "") or "").lower()
        side = "over" if ("over" in sel or sel.startswith("o ")) else None
        if side is None and ("under" in sel or sel.startswith("u ") or prop_key in ("round_1_finish", "under_1_5_rounds")):
            side = "under"
        if side is None:
            if prop_key.startswith("over_"):
                side = "over"
            elif prop_key.startswith("under_") or prop_key == "round_1_finish":
                side = "under"
        if side is None:
            continue
        mapped = map_rounds_total(side, pt)
        if mapped is None:
            continue
        key, selection = mapped
        # Keep legacy alias row for Under 1.5 ↔ R1 finish consumers
        if key == "under_1_5_rounds" and prop_key == "round_1_finish":
            out.at[idx, "selection"] = "Under 1.5 / R1 Finish"
            continue
        out.at[idx, "prop_key"] = key
        out.at[idx, "selection"] = selection
    return out


def prop_row(
    *,
    fighter_1: str,
    fighter_2: str,
    prop_key: str,
    selection: str,
    decimal_odds: float,
    bookmaker: str,
    odds_source: str = "live",
    market_key: str = "",
    point: float | None = None,
    rotation: str = "",
    american_odds: float | None = None,
) -> dict[str, Any]:
    am = american_odds
    if am is None and decimal_odds > 1:
        if decimal_odds >= 2.0:
            am = float(round((decimal_odds - 1.0) * 100))
        else:
            am = float(round(-100.0 / (decimal_odds - 1.0)))
    return {
        "fighter_1": fighter_1.strip(),
        "fighter_2": fighter_2.strip(),
        "prop_key": prop_key,
        "selection": selection,
        "decimal_odds": round(float(decimal_odds), 3),
        "implied_prob": implied_from_decimal(decimal_odds),
        "american_odds": am,
        "market_key": market_key,
        "point": point,
        "rotation": rotation,
        "bookmaker": bookmaker,
        "odds_source": odds_source,
    }


def empty_prop_odds_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PROP_ODDS_COLUMNS)


# Under 1.5 rounds ≡ fight ends in round 1 for matching live quotes.
PROP_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "round_1_finish": ("round_1_finish", "under_1_5_rounds"),
    "under_1_5_rounds": ("under_1_5_rounds", "round_1_finish"),
}


def lookup_prop_odds_row(
    fighter_1: str,
    fighter_2: str,
    prop_key: str,
    odds: pd.DataFrame,
    *,
    selection: str | None = None,
    fighter_name: str | None = None,
) -> pd.Series | None:
    """Find a prop line for a fight, optionally matching selection / method fighter."""
    if odds is None or odds.empty:
        return None
    keys = PROP_KEY_ALIASES.get(str(prop_key), (str(prop_key),))
    named = str(fighter_name or "").strip()
    fallback: pd.Series | None = None
    for _, row in odds.iterrows():
        f1 = str(row.get("fighter_1", ""))
        f2 = str(row.get("fighter_2", ""))
        aligned = _names_match(fighter_1, f1) and _names_match(fighter_2, f2)
        swapped = _names_match(fighter_1, f2) and _names_match(fighter_2, f1)
        if not aligned and not swapped:
            continue
        if str(row.get("prop_key", "")) not in keys:
            continue
        row_sel = str(row.get("selection", "") or "")
        if selection is not None and row_sel.lower() != selection.lower():
            continue
        if named:
            sel_fighter = row_sel
            if sel_fighter.lower().endswith(" yes"):
                sel_fighter = sel_fighter[:-4].strip()
            if _names_match(named, sel_fighter) or named.lower() in sel_fighter.lower():
                return row
            # Keep first fight+key match as fallback when selection fighter unknown
            if fallback is None:
                fallback = row
            continue
        return row
    return fallback


def attach_prop_odds_to_predictions(
    predictions: pd.DataFrame,
    prop_odds: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-prop live quote columns to prediction rows."""
    if predictions.empty:
        return predictions.copy()
    out = predictions.copy()
    for key in set(prop_odds.get("prop_key", pd.Series(dtype=str)).tolist()) if not prop_odds.empty else []:
        out[f"prop_live_odds_{key}"] = np.nan
        out[f"prop_live_implied_{key}"] = np.nan
        out[f"prop_odds_source_{key}"] = ""
    if prop_odds.empty:
        return out

    for idx, row in out.iterrows():
        f1 = str(row.get("fighter_1", row.get("fighter1", "")))
        f2 = str(row.get("fighter_2", row.get("fighter2", "")))
        for prop_key in prop_odds["prop_key"].unique():
            match = lookup_prop_odds_row(f1, f2, str(prop_key), prop_odds)
            if match is None:
                continue
            out.at[idx, f"prop_live_odds_{prop_key}"] = float(match["decimal_odds"])
            out.at[idx, f"prop_live_implied_{prop_key}"] = float(match["implied_prob"])
            out.at[idx, f"prop_odds_source_{prop_key}"] = str(match.get("odds_source", "live"))
    return out


def prop_odds_summary(prop_odds: pd.DataFrame) -> dict[str, int]:
    if prop_odds is None or prop_odds.empty:
        return {"live": 0, "synthetic": 0}
    src = prop_odds.get("odds_source", pd.Series(dtype=str)).astype(str).str.lower()
    live = int(src.isin(["live", "the_odds_api"]).sum())
    return {"live": live, "synthetic": max(0, len(prop_odds) - live)}
