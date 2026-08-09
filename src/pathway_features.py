"""Win/loss pathway + condition/style features — leakage-safe, UFC-scoped.

Computed on long history using prior fights only. Gated into the model via
``config.ENABLE_PATHWAY_FEATURES`` + ``PATHWAY_FEATURE_COLUMNS``.

Benter-style gaps covered here:
- method win/loss pathways (L5 + career)
- R1 finish / distance rates
- cardio proxies (decision-loss, late-finish timing skew)
- quality of last loss (opponent pre-fight Elo)
- matchup clashes (KO/TD/sub pressure)
- bout flags: stance mismatch, is_five_round (bout-level)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Diff / interaction / bout columns added to the optional model block.
PATHWAY_DIFF_COLUMNS: tuple[str, ...] = (
    "ko_win_rate_l5_diff",
    "ko_win_rate_career_diff",
    "sub_win_rate_l5_diff",
    "sub_win_rate_career_diff",
    "dec_win_rate_l5_diff",
    "dec_win_rate_career_diff",
    "ko_loss_rate_l5_diff",
    "ko_loss_rate_career_diff",
    "sub_loss_rate_l5_diff",
    "sub_loss_rate_career_diff",
    "dec_loss_rate_l5_diff",
    "dec_loss_rate_career_diff",
    "r1_finish_rate_l5_diff",
    "r1_finish_rate_career_diff",
    "late_finish_rate_l5_diff",
    "late_finish_rate_career_diff",
    "distance_rate_l5_diff",
    "distance_rate_career_diff",
    "cardio_decay_proxy_diff",
    "finish_timing_skew_diff",
    "last_loss_opp_elo_diff",
    # Key clash interactions (f1-centric minus f2-centric)
    "path_opp_ko_x_own_ko_loss",
    "path_opp_td_att_x_own_td_def",
    "path_opp_sub_x_own_sub_loss",
    "path_pace_product_diff",
    # Bout / style flags (copied onto both sides or bout-level)
    "path_stance_mismatch",
    "is_five_round",
)

# Per-fighter as-of columns produced on long history.
PATHWAY_FIGHTER_STAT_FIELDS: tuple[str, ...] = (
    "ko_win_rate_l5",
    "ko_win_rate_career",
    "sub_win_rate_l5",
    "sub_win_rate_career",
    "dec_win_rate_l5",
    "dec_win_rate_career",
    "ko_loss_rate_l5",
    "ko_loss_rate_career",
    "sub_loss_rate_l5",
    "sub_loss_rate_career",
    "dec_loss_rate_l5",
    "dec_loss_rate_career",
    "r1_finish_rate_l5",
    "r1_finish_rate_career",
    "late_finish_rate_l5",
    "late_finish_rate_career",
    "distance_rate_l5",
    "distance_rate_career",
    "cardio_decay_proxy",
    "finish_timing_skew",
    "last_loss_opp_elo",
    "td_att_rate_l5",
    "td_att_rate_career",
    "sub_att_rate_l5",
    "sub_att_rate_career",
    "pace_l5",
    "pace_career",
)


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shifted_expanding_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def _f(d: dict[str, float], key: str, default: float = np.nan) -> float:
    try:
        v = d.get(key, default)
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _last_prior_loss_opp_elo(won: pd.Series, opp_elo: pd.Series) -> pd.Series:
    """As-of Elo of opponent in fighter's most recent prior loss (shifted)."""
    loss_elo = opp_elo.where(won.fillna(1).astype(int) == 0)
    # Forward-fill loss Elo within fighter, then shift so current fight sees prior only.
    filled = loss_elo.ffill()
    return filled.shift(1)


def apply_pathway_rolling_extras(history: pd.DataFrame) -> pd.DataFrame:
    """Add pathway / condition columns onto long fighter history (prior only).

    Prefer calling this *after* Greco/CompuBox fills and SOS Elo attach so
    pace / last_loss_opp_elo have usable inputs.
    """
    if history is None or history.empty:
        return history

    out = history

    won = out["won"].fillna(0).astype(int) if "won" in out.columns else pd.Series(0, index=out.index)
    is_ko = out["is_ko"].fillna(0).astype(int) if "is_ko" in out.columns else pd.Series(0, index=out.index)
    is_sub = out["is_sub"].fillna(0).astype(int) if "is_sub" in out.columns else pd.Series(0, index=out.index)
    is_dec = out["is_dec"].fillna(0).astype(int) if "is_dec" in out.columns else pd.Series(0, index=out.index)

    out["_path_ko_win"] = ((won == 1) & (is_ko == 1)).astype(float)
    out["_path_sub_win"] = ((won == 1) & (is_sub == 1)).astype(float)
    out["_path_dec_win"] = ((won == 1) & (is_dec == 1)).astype(float)
    out["_path_ko_loss"] = ((won == 0) & (is_ko == 1)).astype(float)
    out["_path_sub_loss"] = ((won == 0) & (is_sub == 1)).astype(float)
    out["_path_dec_loss"] = ((won == 0) & (is_dec == 1)).astype(float)
    out["_path_went_distance"] = (is_dec == 1).astype(float)

    if "round" in out.columns:
        rnd = pd.to_numeric(out["round"], errors="coerce")
        finished = (is_ko == 1) | (is_sub == 1)
        if "finish" in out.columns:
            finished = finished | (pd.to_numeric(out["finish"], errors="coerce").fillna(0) > 0)
        out["_path_r1_finish"] = ((rnd == 1) & finished).astype(float)
        # Late-round finish (R3+) — cardio / championship-rounds proxy vs R1
        out["_path_late_finish"] = ((rnd >= 3) & finished).astype(float)
    else:
        out["_path_r1_finish"] = 0.0
        out["_path_late_finish"] = 0.0

    if "takedowns_attempted" in out.columns:
        out["_path_td_att"] = pd.to_numeric(out["takedowns_attempted"], errors="coerce")
    else:
        out["_path_td_att"] = np.nan

    # Sub "attempts" proxy: submission finishes by this fighter (offensive sub rate)
    out["_path_sub_att"] = out["_path_sub_win"].astype(float)

    # Pace: prefer fight-level SLpM; fall back to rolled column after Greco fill
    if "sig_strikes_per_min" in out.columns and out["sig_strikes_per_min"].notna().any():
        out["_path_pace"] = pd.to_numeric(out["sig_strikes_per_min"], errors="coerce")
    elif "sig_strikes_per_min_roll" in out.columns:
        out["_path_pace"] = pd.to_numeric(out["sig_strikes_per_min_roll"], errors="coerce")
    else:
        out["_path_pace"] = np.nan

    g = out.groupby("fighter", group_keys=False)
    last5 = 5
    rate_specs: list[tuple[str, str]] = [
        ("ko_win_rate", "_path_ko_win"),
        ("sub_win_rate", "_path_sub_win"),
        ("dec_win_rate", "_path_dec_win"),
        ("ko_loss_rate", "_path_ko_loss"),
        ("sub_loss_rate", "_path_sub_loss"),
        ("dec_loss_rate", "_path_dec_loss"),
        ("r1_finish_rate", "_path_r1_finish"),
        ("late_finish_rate", "_path_late_finish"),
        ("distance_rate", "_path_went_distance"),
        ("td_att_rate", "_path_td_att"),
        ("sub_att_rate", "_path_sub_att"),
        ("pace", "_path_pace"),
    ]
    for name, src in rate_specs:
        out[f"{name}_l5"] = g[src].apply(lambda s: _shifted_rolling_mean(s, last5))
        out[f"{name}_career"] = g[src].apply(_shifted_expanding_mean)

    # Cardio decay proxy (no per-round CompuBox required):
    # decision-loss rate × distance rate (fails late when fights go long).
    out["cardio_decay_proxy"] = (
        out["dec_loss_rate_l5"].fillna(0.0) * out["distance_rate_l5"].fillna(0.0)
    )
    # Positive skew = early finisher; negative = more late finishes
    out["finish_timing_skew"] = (
        out["r1_finish_rate_l5"] - out["late_finish_rate_l5"]
    )

    # Quality of last loss — opponent pre-fight Elo (SOS attach provides column)
    opp_elo_col = None
    for cand in ("_opp_elo_prefight", "opp_elo_prefight", "avg_opp_elo"):
        if cand in out.columns:
            opp_elo_col = cand
            break
    if opp_elo_col is not None:
        parts: list[pd.Series] = []
        for _, grp in out.groupby("fighter", sort=False):
            parts.append(_last_prior_loss_opp_elo(grp["won"], grp[opp_elo_col]))
        out["last_loss_opp_elo"] = pd.concat(parts).reindex(out.index)
    else:
        out["last_loss_opp_elo"] = np.nan

    drop_tmp = [c for c in out.columns if c.startswith("_path_")]
    out.drop(columns=drop_tmp, inplace=True, errors="ignore")
    return out


def build_pathway_matchup_features(
    f1: dict[str, float],
    f2: dict[str, float],
    *,
    scheduled_rounds: Any = None,
) -> dict[str, float]:
    """Differential + clash pathway features for the optional PATH block."""

    def _diff(key: str) -> float:
        a, b = _f(f1, key), _f(f2, key)
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return float(a - b)

    def _clash(opp_key: str, own_key: str) -> float:
        f1_c = _f(f2, opp_key) * _f(f1, own_key)
        f2_c = _f(f1, opp_key) * _f(f2, own_key)
        if np.isnan(f1_c) or np.isnan(f2_c):
            return np.nan
        return float(f1_c - f2_c)

    # Stance mismatch (southpaw vs orthodox) — also in BASE; PATH re-exposes
    f1_sw, f2_sw = _f(f1, "stance_southpaw", 0.0), _f(f2, "stance_southpaw", 0.0)
    f1_or, f2_or = _f(f1, "stance_orthodox", 0.0), _f(f2, "stance_orthodox", 0.0)
    stance_mismatch = float(
        (f1_sw == 1 and f2_or == 1) or (f1_or == 1 and f2_sw == 1)
    )

    try:
        sr = float(scheduled_rounds) if scheduled_rounds is not None else np.nan
    except (TypeError, ValueError):
        sr = np.nan
    is_five = 1.0 if (not np.isnan(sr) and sr >= 5) else 0.0

    # Pace product differential
    p1, p2 = _f(f1, "pace_l5"), _f(f2, "pace_l5")
    if np.isnan(p1) or np.isnan(p2):
        # Fall back to career pace or sig_strikes_per_min
        p1 = _f(f1, "pace_career")
        if np.isnan(p1):
            p1 = _f(f1, "sig_strikes_per_min")
        p2 = _f(f2, "pace_career")
        if np.isnan(p2):
            p2 = _f(f2, "sig_strikes_per_min")
    if np.isnan(p1) or np.isnan(p2):
        pace_product_diff = np.nan
    else:
        pace_product_diff = float(p1 - p2) * float((p1 + p2) / 2.0)

    # TD pressure: prefer td_att_rate; fall back to td_acc
    td_att_1 = _f(f1, "td_att_rate_career")
    if np.isnan(td_att_1):
        td_att_1 = _f(f1, "td_acc")
    td_att_2 = _f(f2, "td_att_rate_career")
    if np.isnan(td_att_2):
        td_att_2 = _f(f2, "td_acc")
    f1_td = td_att_2 * _f(f1, "td_defense")
    f2_td = td_att_1 * _f(f2, "td_defense")
    if np.isnan(f1_td) or np.isnan(f2_td):
        opp_td_x_own_def = np.nan
    else:
        opp_td_x_own_def = float(f1_td - f2_td)

    return {
        "ko_win_rate_l5_diff": _diff("ko_win_rate_l5"),
        "ko_win_rate_career_diff": _diff("ko_win_rate_career"),
        "sub_win_rate_l5_diff": _diff("sub_win_rate_l5"),
        "sub_win_rate_career_diff": _diff("sub_win_rate_career"),
        "dec_win_rate_l5_diff": _diff("dec_win_rate_l5"),
        "dec_win_rate_career_diff": _diff("dec_win_rate_career"),
        "ko_loss_rate_l5_diff": _diff("ko_loss_rate_l5"),
        "ko_loss_rate_career_diff": _diff("ko_loss_rate_career"),
        "sub_loss_rate_l5_diff": _diff("sub_loss_rate_l5"),
        "sub_loss_rate_career_diff": _diff("sub_loss_rate_career"),
        "dec_loss_rate_l5_diff": _diff("dec_loss_rate_l5"),
        "dec_loss_rate_career_diff": _diff("dec_loss_rate_career"),
        "r1_finish_rate_l5_diff": _diff("r1_finish_rate_l5"),
        "r1_finish_rate_career_diff": _diff("r1_finish_rate_career"),
        "late_finish_rate_l5_diff": _diff("late_finish_rate_l5"),
        "late_finish_rate_career_diff": _diff("late_finish_rate_career"),
        "distance_rate_l5_diff": _diff("distance_rate_l5"),
        "distance_rate_career_diff": _diff("distance_rate_career"),
        "cardio_decay_proxy_diff": _diff("cardio_decay_proxy"),
        "finish_timing_skew_diff": _diff("finish_timing_skew"),
        "last_loss_opp_elo_diff": _diff("last_loss_opp_elo"),
        "path_opp_ko_x_own_ko_loss": _clash("ko_win_rate_career", "ko_loss_rate_career"),
        "path_opp_td_att_x_own_td_def": opp_td_x_own_def,
        "path_opp_sub_x_own_sub_loss": _clash("sub_att_rate_career", "sub_loss_rate_career"),
        "path_pace_product_diff": pace_product_diff,
        "path_stance_mismatch": stance_mismatch,
        "is_five_round": is_five,
    }


def log_pathway_coverage(
    features: pd.DataFrame, *, year: int = 2025, label: str = ""
) -> dict[str, float]:
    """Log / return non-null coverage %% for pathway diffs on a year sample."""
    coverage: dict[str, float] = {}
    if features is None or features.empty:
        logger.info("Pathway coverage [%s]: empty frame", label or "n/a")
        return coverage
    date_col = "event_date" if "event_date" in features.columns else "date"
    sample = features
    if date_col in features.columns:
        dts = pd.to_datetime(features[date_col], errors="coerce")
        sample = features.loc[dts.dt.year == year]
    if sample.empty:
        sample = features
    logger.info(
        "Pathway coverage [%s]: year=%s n=%s",
        label or "n/a",
        year,
        len(sample),
    )
    for col in PATHWAY_DIFF_COLUMNS:
        if col not in sample.columns:
            coverage[col] = 0.0
            logger.info("  %s: MISSING", col)
            continue
        nn = float(sample[col].notna().mean() * 100.0)
        coverage[col] = nn
        logger.info("  %s: %.1f%% non-null", col, nn)
    return coverage
