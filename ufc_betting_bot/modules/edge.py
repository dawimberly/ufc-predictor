"""Edge and implied probability from historical / closing odds.

Strict as-of evaluation only accepts a validated opening price. Closing-only
and unlabeled (unproven) quotes are refused so selection, sizing, and
settlement never mix price timings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def odds_to_decimal(price: float) -> float:
    if not np.isfinite(price):
        return np.nan
    if abs(price) > 100:
        if price >= 100:
            return 1.0 + price / 100.0
        return 1.0 + 100.0 / abs(price)
    return float(price)


def has_valid_odds(row: pd.Series) -> bool:
    if "f1_odds" not in row.index or "f2_odds" not in row.index:
        return False
    o1, o2 = row.get("f1_odds"), row.get("f2_odds")
    if pd.isna(o1) or pd.isna(o2):
        return False
    try:
        o1_f, o2_f = float(o1), float(o2)
    except (TypeError, ValueError):
        return False
    if abs(o1_f) > 100 or abs(o2_f) > 100:
        return abs(o1_f) >= 100 and abs(o2_f) >= 100
    return o1_f > 1 and o2_f > 1


def fight_decimal_odds(row: pd.Series) -> tuple[float, float] | None:
    if not has_valid_odds(row):
        return None
    f1 = odds_to_decimal(float(row["f1_odds"]))
    f2 = odds_to_decimal(float(row["f2_odds"]))
    if not np.isfinite(f1) or not np.isfinite(f2) or f1 <= 1 or f2 <= 1:
        return None
    return f1, f2


def decimal_implied_prob(f1_odds: float, f2_odds: float) -> tuple[float, float]:
    p1 = 1 / f1_odds if f1_odds > 0 else np.nan
    p2 = 1 / f2_odds if f2_odds > 0 else np.nan
    total = p1 + p2
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    return p1 / total, p2 / total


def market_probs(row: pd.Series) -> tuple[float, float] | None:
    """De-vigged implied probabilities from ``f1_odds`` / ``f2_odds``.

    For strict historical as-of work use :func:`validated_as_of_price` instead.
    """
    decimal = fight_decimal_odds(row)
    if decimal is None:
        return None
    p1, p2 = decimal_implied_prob(*decimal)
    if np.isfinite(p1) and np.isfinite(p2):
        return p1, p2
    return None


_OPENING_ODDS_PAIRS = (
    ("f1_opening_odds", "f2_opening_odds"),
    ("opening_f1_odds", "opening_f2_odds"),
    ("open_f1_odds", "open_f2_odds"),
)
_CLOSING_ODDS_PAIRS = (
    ("f1_closing_odds", "f2_closing_odds"),
    ("closing_f1_odds", "closing_f2_odds"),
)


@dataclass(frozen=True)
class AsOfMarketPrice:
    """One validated historical price used for selection, sizing, and settlement."""

    market_p1: float
    market_p2: float
    decimal_f1: float
    decimal_f2: float
    timing: str

    def decimals_for_side(self, side: str) -> float:
        return self.decimal_f1 if side == "f1" else self.decimal_f2


def _decimals_from_named_pair(
    row: pd.Series, left: str, right: str
) -> tuple[float, float] | None:
    if left not in row.index or right not in row.index:
        return None
    renamed = row.copy()
    renamed["f1_odds"] = row[left]
    renamed["f2_odds"] = row[right]
    return fight_decimal_odds(renamed)


def _probs_from_named_pair(row: pd.Series, left: str, right: str) -> tuple[float, float] | None:
    decimals = _decimals_from_named_pair(row, left, right)
    if decimals is None:
        return None
    p1, p2 = decimal_implied_prob(*decimals)
    if np.isfinite(p1) and np.isfinite(p2):
        return p1, p2
    return None


def validated_as_of_price(row: pd.Series) -> tuple[AsOfMarketPrice | None, str]:
    """Strict as-of market price a historical bet could have used.

    Opening prices win when present. Closing-only and unlabeled ``f1_odds`` /
    ``f2_odds`` (unproven timing) are refused — they must not feed selection,
    sizing, or settlement under a strict as-of claim.
    """
    for left, right in _OPENING_ODDS_PAIRS:
        decimals = _decimals_from_named_pair(row, left, right)
        if decimals is None:
            continue
        p1, p2 = decimal_implied_prob(*decimals)
        if np.isfinite(p1) and np.isfinite(p2):
            return (
                AsOfMarketPrice(
                    market_p1=p1,
                    market_p2=p2,
                    decimal_f1=decimals[0],
                    decimal_f2=decimals[1],
                    timing="opening",
                ),
                "opening",
            )

    closing_pair: tuple[str, str] | None = None
    for left, right in _CLOSING_ODDS_PAIRS:
        if _decimals_from_named_pair(row, left, right) is not None:
            closing_pair = (left, right)
            break

    stored_decimals = fight_decimal_odds(row) if has_valid_odds(row) else None
    if closing_pair is not None:
        left, right = closing_pair
        if stored_decimals is None:
            return None, "closing_not_as_of"
        try:
            same = abs(float(row["f1_odds"]) - float(row[left])) <= 1e-6 and abs(
                float(row["f2_odds"]) - float(row[right])
            ) <= 1e-6
        except (TypeError, ValueError):
            same = False
        if same:
            return None, "closing_not_as_of"
        # Distinct unlabeled f1_odds alongside a closing pair is still unproven:
        # without an opening label we will not treat it as as-of.
        return None, "unproven"

    if stored_decimals is not None:
        return None, "unproven"
    return None, "missing"


def as_of_market_probs(row: pd.Series) -> tuple[tuple[float, float] | None, str]:
    """Market probabilities for strict as-of evaluation (opening only)."""
    price, timing = validated_as_of_price(row)
    if price is None:
        return None, timing
    return (price.market_p1, price.market_p2), timing


def as_of_price_from_row(row: pd.Series) -> AsOfMarketPrice | None:
    """Reuse a price already attached by walk-forward, else validate from columns.

    Ensures selection / sizing / settlement read the same validated opening
    price when ``as_of_decimal_*`` and ``odds_timing`` were written upstream.
    """
    timing = str(row.get("odds_timing") or "").strip()
    if timing and timing != "opening":
        return None

    dec_f1 = row.get("as_of_decimal_f1")
    dec_f2 = row.get("as_of_decimal_f2")
    m1 = row.get("implied_prob_f1")
    m2 = row.get("implied_prob_f2")
    try:
        if (
            timing == "opening"
            and pd.notna(dec_f1)
            and pd.notna(dec_f2)
            and pd.notna(m1)
            and pd.notna(m2)
        ):
            d1, d2 = float(dec_f1), float(dec_f2)
            p1, p2 = float(m1), float(m2)
            if d1 > 1 and d2 > 1 and np.isfinite(p1) and np.isfinite(p2):
                return AsOfMarketPrice(
                    market_p1=p1,
                    market_p2=p2,
                    decimal_f1=d1,
                    decimal_f2=d2,
                    timing="opening",
                )
    except (TypeError, ValueError):
        pass

    price, resolved = validated_as_of_price(row)
    if price is None or resolved != "opening":
        return None
    return price


def compute_edge(
    model_p1: float,
    model_p2: float,
    market_p1: float,
    market_p2: float,
) -> dict[str, float]:
    edge_f1 = model_p1 - market_p1
    edge_f2 = model_p2 - market_p2
    if edge_f1 >= edge_f2:
        return {
            "bet_side": "f1",
            "edge": edge_f1,
            "best_edge": max(edge_f1, edge_f2),
            "edge_f1": edge_f1,
            "edge_f2": edge_f2,
        }
    return {
        "bet_side": "f2",
        "edge": edge_f2,
        "best_edge": max(edge_f1, edge_f2),
        "edge_f1": edge_f1,
        "edge_f2": edge_f2,
    }


def raw_kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction before scaling and caps."""
    if decimal_odds <= 1 or not np.isfinite(prob) or prob <= 0 or prob >= 1:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob
    return max(0.0, (b * prob - q) / b)
