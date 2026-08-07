"""High-value prediction features (Phase 1) — leakage-safe, optional model block.

Computed on long history / matchups using prior fights only. Gated into the model
via ``config.ENABLE_HIGH_VALUE_FEATURES`` + ``HIGH_VALUE_FEATURE_COLUMNS``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Phase 1 layoff thresholds (parallel to legacy 10d / 180d flags).
HV_SHORT_NOTICE_DAYS = 21
HV_LONG_LAYOFF_DAYS = 365

# Approximate career peak ages by division (years).
DIVISION_PEAK_AGE: dict[str, float] = {
    "flyweight": 28.0,
    "bantamweight": 29.0,
    "featherweight": 29.0,
    "lightweight": 30.0,
    "welterweight": 30.0,
    "middleweight": 31.0,
    "light heavyweight": 32.0,
    "heavyweight": 33.0,
    "open weight": 32.0,
    "catch weight": 30.0,
    "women": 29.0,
    "women's strawweight": 28.0,
    "women's flyweight": 28.0,
    "women's bantamweight": 29.0,
    "women's featherweight": 29.0,
}

# Diff columns added to the optional model block.
HIGH_VALUE_DIFF_COLUMNS: tuple[str, ...] = (
    "hv_short_notice_flag_diff",
    "hv_long_layoff_flag_diff",
    "first_fight_new_wc_flag_diff",
    "finish_rate_l5_diff",
    "division_age_adj_diff",
    "hv_td_pressure_diff",
    "hv_control_clash",
    "wins_vs_better_record_l5_diff",
    "ko_losses_career_flag_diff",
)

# Per-fighter rolling / as-of columns produced on long history.
HV_FIGHTER_STAT_FIELDS: tuple[str, ...] = (
    "hv_short_notice_flag",
    "hv_long_layoff_flag",
    "first_fight_new_wc_flag",
    "finish_rate_l5",
    "division_age_adj",
    "wins_vs_better_record_l5",
    "ko_losses_career_flag",
)


def normalize_weight_class(wc: Any) -> str:
    text = str(wc or "").strip().lower()
    if not text:
        return ""
    if "women" in text or text.startswith("w"):
        for key in (
            "women's strawweight",
            "women's flyweight",
            "women's bantamweight",
            "women's featherweight",
        ):
            if key.replace("women's ", "") in text or key in text:
                return key
        return "women"
    for key in DIVISION_PEAK_AGE:
        if key in text:
            return key
    return text


def division_peak_age(weight_class: Any) -> float:
    key = normalize_weight_class(weight_class)
    if key in DIVISION_PEAK_AGE:
        return float(DIVISION_PEAK_AGE[key])
    if key.startswith("women"):
        return float(DIVISION_PEAK_AGE["women"])
    return 30.0


def hv_layoff_flags(days_since_last: float) -> tuple[float, float]:
    """Return (short_notice_≤21d, long_layoff_≥365d). Debut / NaN → (0, 0)."""
    try:
        if days_since_last is None or (isinstance(days_since_last, float) and np.isnan(days_since_last)):
            return 0.0, 0.0
        days = float(days_since_last)
    except (TypeError, ValueError):
        return 0.0, 0.0
    short_flag = 1.0 if days <= HV_SHORT_NOTICE_DAYS else 0.0
    long_flag = 1.0 if days >= HV_LONG_LAYOFF_DAYS else 0.0
    return short_flag, long_flag


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def apply_hv_rolling_extras(history: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    """Add Phase 1 rolling / as-of columns onto long fighter history (prior fights only)."""
    if history is None or history.empty:
        return history

    out = history
    g = out.groupby("fighter", group_keys=False)
    last5 = 5

    # --- Layoff flags (21d / 365d), parallel to legacy ---
    if "days_since_last_fight" in out.columns:
        pairs = out["days_since_last_fight"].map(hv_layoff_flags)
        out["hv_short_notice_flag"] = [p[0] for p in pairs]
        out["hv_long_layoff_flag"] = [p[1] for p in pairs]
    else:
        out["hv_short_notice_flag"] = 0.0
        out["hv_long_layoff_flag"] = 0.0

    # --- First fight in a new weight class (vs prior bout class) ---
    if "weight_class" in out.columns:
        prev_wc = g["weight_class"].shift(1)
        cur = out["weight_class"].map(normalize_weight_class)
        prev = prev_wc.map(normalize_weight_class)
        out["first_fight_new_wc_flag"] = np.where(
            prev.isna() | (prev == "") | (cur == ""),
            0.0,
            (cur != prev).astype(float),
        )
    else:
        out["first_fight_new_wc_flag"] = 0.0

    # --- finish_rate_l5 (explicit last-5 window) ---
    if "finish" in out.columns:
        out["finish_rate_l5"] = g["finish"].apply(lambda s: _shifted_rolling_mean(s, last5))
    elif "finish_rate" in out.columns:
        out["finish_rate_l5"] = out["finish_rate"]
    else:
        out["finish_rate_l5"] = np.nan

    # --- Division-adjusted age ---
    if "age" in out.columns:
        peaks = out.get("weight_class", pd.Series("", index=out.index)).map(division_peak_age)
        out["division_age_adj"] = out["age"] - peaks
    else:
        out["division_age_adj"] = np.nan

    # --- KO losses career flag (any prior KO/TKO loss) ---
    if "is_ko" in out.columns and "won" in out.columns:
        ko_loss = ((out["won"] == 0) & (out["is_ko"] == 1)).astype(float)
        out["_ko_loss"] = ko_loss
        out["ko_losses_career"] = out.groupby("fighter", group_keys=False)["_ko_loss"].apply(
            lambda s: s.shift(1).cumsum()
        )
        out.drop(columns=["_ko_loss"], inplace=True, errors="ignore")
        out["ko_losses_career_flag"] = (out["ko_losses_career"].fillna(0) > 0).astype(float)
    else:
        out["ko_losses_career"] = np.nan
        out["ko_losses_career_flag"] = 0.0

    # --- Wins vs better-record opponents (L5 rate, prior only) ---
    fid = "fight_id" if "fight_id" in out.columns else None
    if (
        fid
        and "win_rate" in out.columns
        and "won" in out.columns
        and "opponent" in out.columns
    ):
        opp = out[[fid, "fighter", "win_rate"]].rename(
            columns={"fighter": "opponent", "win_rate": "opp_asof_win_rate"}
        )
        merged = out[[fid, "opponent", "won", "win_rate", "fighter"]].merge(
            opp, on=[fid, "opponent"], how="left"
        )
        beat_better = (
            (merged["won"] == 1)
            & merged["opp_asof_win_rate"].notna()
            & merged["win_rate"].notna()
            & (merged["opp_asof_win_rate"] > merged["win_rate"])
        ).astype(float)
        beat_better.index = out.index
        out["beat_better_record"] = beat_better
        out["wins_vs_better_record_l5"] = out.groupby("fighter", group_keys=False)[
            "beat_better_record"
        ].apply(lambda s: _shifted_rolling_mean(s, last5))
    else:
        out["wins_vs_better_record_l5"] = np.nan

    return out


def build_hv_matchup_features(
    f1: dict[str, float],
    f2: dict[str, float],
) -> dict[str, float]:
    """Differential / interaction features for the HV block."""

    def _f(d: dict[str, float], key: str, default: float = np.nan) -> float:
        try:
            v = d.get(key, default)
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    f1_td_acc, f2_td_acc = _f(f1, "td_acc"), _f(f2, "td_acc")
    f1_td_def, f2_td_def = _f(f1, "td_defense"), _f(f2, "td_defense")
    f1_ctrl, f2_ctrl = _f(f1, "control_time_per_min", 0.0), _f(f2, "control_time_per_min", 0.0)
    # Prefer NaN-safe product: missing control → 0 for clash intensity only
    if np.isnan(f1_ctrl):
        f1_ctrl = 0.0
    if np.isnan(f2_ctrl):
        f2_ctrl = 0.0

    td_pressure_f1 = f2_td_acc * f1_td_def  # opp td_acc × own td_def
    td_pressure_f2 = f1_td_acc * f2_td_def
    if np.isnan(td_pressure_f1) or np.isnan(td_pressure_f2):
        hv_td_pressure_diff = np.nan
    else:
        hv_td_pressure_diff = float(td_pressure_f1 - td_pressure_f2)

    # control_against ≈ opponent control; × opp_control_avg ≈ opp_control^2 proxy, use product of both controls
    hv_control_clash = float(f1_ctrl * f2_ctrl)

    return {
        "hv_short_notice_flag_diff": _f(f1, "hv_short_notice_flag", 0.0)
        - _f(f2, "hv_short_notice_flag", 0.0),
        "hv_long_layoff_flag_diff": _f(f1, "hv_long_layoff_flag", 0.0)
        - _f(f2, "hv_long_layoff_flag", 0.0),
        "first_fight_new_wc_flag_diff": _f(f1, "first_fight_new_wc_flag", 0.0)
        - _f(f2, "first_fight_new_wc_flag", 0.0),
        "finish_rate_l5_diff": _f(f1, "finish_rate_l5") - _f(f2, "finish_rate_l5"),
        "division_age_adj_diff": _f(f1, "division_age_adj") - _f(f2, "division_age_adj"),
        "hv_td_pressure_diff": hv_td_pressure_diff,
        "hv_control_clash": hv_control_clash,
        "wins_vs_better_record_l5_diff": _f(f1, "wins_vs_better_record_l5")
        - _f(f2, "wins_vs_better_record_l5"),
        "ko_losses_career_flag_diff": _f(f1, "ko_losses_career_flag", 0.0)
        - _f(f2, "ko_losses_career_flag", 0.0),
    }


def log_hv_coverage(features: pd.DataFrame, *, year: int = 2025, label: str = "") -> None:
    """Log non-null coverage %% for HV diffs on a calendar-year card sample."""
    if features is None or features.empty:
        logger.info("HV coverage [%s]: empty frame", label or "n/a")
        return
    date_col = "event_date" if "event_date" in features.columns else "date"
    sample = features
    if date_col in features.columns:
        dts = pd.to_datetime(features[date_col], errors="coerce")
        sample = features.loc[dts.dt.year == year]
    if sample.empty:
        sample = features
        logger.info(
            "HV coverage [%s]: no %s rows — using full frame n=%s",
            label or "n/a",
            year,
            len(sample),
        )
    else:
        logger.info(
            "HV coverage [%s]: year=%s n=%s",
            label or "n/a",
            year,
            len(sample),
        )
    for col in HIGH_VALUE_DIFF_COLUMNS:
        if col not in sample.columns:
            logger.info("  %s: MISSING", col)
            continue
        nn = float(sample[col].notna().mean() * 100.0)
        logger.info("  %s: %.1f%% non-null", col, nn)
