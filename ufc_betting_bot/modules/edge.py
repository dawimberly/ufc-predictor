"""Edge and implied probability from historical / closing odds.

Strict as-of evaluation only accepts a validated opening price that was
available at or before the fight's prediction cutoff. Closing-only and
unlabeled (unproven) quotes are refused so selection, sizing, and settlement
never mix price timings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

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
_OPENING_QUOTE_TIME_KEYS = (
    "opening_odds_observed_at",
    "opening_quote_timestamp",
    "opening_odds_timestamp",
    "f1_opening_odds_timestamp",
)
_EVENT_CUTOFF_KEYS = (
    "commence_time",
    "event_date",
    "date",
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


def _parse_utc_timestamp(value, *, require_explicit_tz: bool = False) -> datetime | None:
    """Parse to timezone-aware UTC.

    When ``require_explicit_tz`` is True (opening quote times), naive datetimes
    and strings without a timezone offset are rejected. Fight-start / as_of
    calendar dates may omit a tz and are interpreted as UTC midnight.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None

    if require_explicit_tz:
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return None
            return value.tz_convert("UTC").to_pydatetime()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return None
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            try:
                ts = pd.to_datetime(value, utc=False, errors="coerce")
            except (TypeError, ValueError):
                return None
            if pd.isna(ts) or getattr(ts, "tzinfo", None) is None:
                return None
            return ts.tz_convert("UTC").to_pydatetime()
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        except (TypeError, ValueError):
            return None
        if pd.isna(ts):
            return None
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def event_prediction_cutoff(row: pd.Series) -> datetime | None:
    """Fight-start bound: commence time when present, else event/date (UTC)."""
    for key in _EVENT_CUTOFF_KEYS:
        if key not in row.index:
            continue
        parsed = _parse_utc_timestamp(row.get(key), require_explicit_tz=False)
        if parsed is not None:
            return parsed
    return None


def opening_quote_observed_at(row: pd.Series) -> datetime | None:
    """When the opening quote was observed (timezone-aware required)."""
    for key in _OPENING_QUOTE_TIME_KEYS:
        if key not in row.index:
            continue
        parsed = _parse_utc_timestamp(row.get(key), require_explicit_tz=True)
        if parsed is not None:
            return parsed
    return None


def effective_prediction_cutoff(
    row: pd.Series,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> datetime | None:
    """Cutoff used for as-of checks: never later than fight start.

    A caller-supplied ``as_of`` cannot extend past ``commence_time`` /
    ``event_date``. Post-fight cutoffs therefore cannot authorize post-fight
    quotes.
    """
    fight_start = event_prediction_cutoff(row)
    if as_of is None:
        return fight_start
    parsed_as_of = _parse_utc_timestamp(as_of, require_explicit_tz=False)
    if parsed_as_of is None:
        return None
    if fight_start is None:
        return None
    return min(parsed_as_of, fight_start)


def _decimals_from_named_pair(
    row: pd.Series, left: str, right: str
) -> tuple[float, float] | None:
    if left not in row.index or right not in row.index:
        return None
    renamed = row.copy()
    renamed["f1_odds"] = row[left]
    renamed["f2_odds"] = row[right]
    return fight_decimal_odds(renamed)


def validated_as_of_price(
    row: pd.Series,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> tuple[AsOfMarketPrice | None, str]:
    """Strict as-of market price a historical bet could have used.

    Opening prices win only when:
    1. Named opening odds exist and are valid, and
    2. An opening quote timestamp is timezone-aware and ≤ the prediction
       cutoff, which is never later than fight start (``min(as_of,
       commence/event_date)``).

    A bare opening *label* without a timezone-aware timestamp is ``unproven``.
    Closing-only and unlabeled ``f1_odds`` / ``f2_odds`` are refused.
    """
    cutoff = effective_prediction_cutoff(row, as_of=as_of)

    for left, right in _OPENING_ODDS_PAIRS:
        decimals = _decimals_from_named_pair(row, left, right)
        if decimals is None:
            continue
        p1, p2 = decimal_implied_prob(*decimals)
        if not (np.isfinite(p1) and np.isfinite(p2)):
            continue
        observed = opening_quote_observed_at(row)
        if observed is None:
            # Label without timestamped availability is not proof of as-of access.
            return None, "unproven"
        if cutoff is None:
            return None, "unproven"
        if observed > cutoff:
            return None, "unproven"
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
        return None, "unproven"

    if stored_decimals is not None:
        return None, "unproven"
    return None, "missing"


def as_of_market_probs(
    row: pd.Series,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> tuple[tuple[float, float] | None, str]:
    """Market probabilities for strict as-of evaluation (opening only)."""
    price, timing = validated_as_of_price(row, as_of=as_of)
    if price is None:
        return None, timing
    return (price.market_p1, price.market_p2), timing


def as_of_price_from_row(
    row: pd.Series,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> AsOfMarketPrice | None:
    """Validated opening price for selection / sizing / settlement.

    Always re-runs :func:`validated_as_of_price` so attached
    ``odds_timing`` / ``as_of_decimal_*`` fields cannot bypass rejection.
    When those fields are present they must match the revalidated opening
    decimals within a tight tolerance.
    """
    timing = str(row.get("odds_timing") or "").strip()
    if timing and timing not in ("opening",):
        return None

    price, resolved = validated_as_of_price(row, as_of=as_of)
    if price is None or resolved != "opening":
        return None

    dec_f1 = row.get("as_of_decimal_f1")
    dec_f2 = row.get("as_of_decimal_f2")
    try:
        if pd.notna(dec_f1) and pd.notna(dec_f2):
            if (
                abs(float(dec_f1) - price.decimal_f1) > 1e-6
                or abs(float(dec_f2) - price.decimal_f2) > 1e-6
            ):
                return None
    except (TypeError, ValueError):
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


def as_of_eligibility_counts(predictions: pd.DataFrame) -> dict[str, int]:
    """Total fights and as-of exclusion tallies for reporting."""
    if predictions is None or len(predictions) == 0:
        return {
            "n_total": 0,
            "n_as_of_eligible": 0,
            "n_opening": 0,
            "n_closing_not_as_of": 0,
            "n_unproven": 0,
            "n_missing": 0,
            "n_other": 0,
        }
    timing = (
        predictions["odds_timing"].astype(str)
        if "odds_timing" in predictions.columns
        else pd.Series(["missing"] * len(predictions), index=predictions.index)
    )
    counts = timing.value_counts(dropna=False).to_dict()
    n_opening = int(counts.get("opening", 0))
    return {
        "n_total": int(len(predictions)),
        "n_as_of_eligible": n_opening,
        "n_opening": n_opening,
        "n_closing_not_as_of": int(counts.get("closing_not_as_of", 0)),
        "n_unproven": int(counts.get("unproven", 0)),
        "n_missing": int(counts.get("missing", 0)),
        "n_other": int(
            len(predictions)
            - n_opening
            - int(counts.get("closing_not_as_of", 0))
            - int(counts.get("unproven", 0))
            - int(counts.get("missing", 0))
        ),
    }
