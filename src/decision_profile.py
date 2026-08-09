"""Fighter decision-profile features (Phase 1) — judge-agnostic.

Leakage-safe as-of rates from method labels. Not added to FEATURE_COLUMNS
unless a Phase 3 keep rule passes. Pathway flags stay untouched.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

DECISION_PROFILE_FIGHTER_FIELDS: tuple[str, ...] = (
    "dec_win_rate_l5",
    "dec_win_rate_career",
    "dec_loss_rate_l5",
    "dec_loss_rate_career",
    "split_dec_win_rate_l5",
    "split_dec_win_rate_career",
    "split_dec_loss_rate_l5",
    "split_dec_loss_rate_career",
    "decision_finish_share_l5",
    "decision_finish_share_career",
)

DECISION_PROFILE_DIFF_COLUMNS: tuple[str, ...] = (
    "dec_win_rate_l5_diff",
    "dec_win_rate_career_diff",
    "dec_loss_rate_l5_diff",
    "dec_loss_rate_career_diff",
    "split_dec_win_rate_l5_diff",
    "split_dec_win_rate_career_diff",
    "split_dec_loss_rate_l5_diff",
    "split_dec_loss_rate_career_diff",
    "decision_finish_share_l5_diff",
    "decision_finish_share_career_diff",
)


def _method_is_dec(method: Any) -> bool:
    s = str(method or "").upper()
    return "DEC" in s or "DECISION" in s


def _method_is_split(method: Any) -> bool:
    s = str(method or "").upper()
    return "SPLIT" in s or s.replace(" ", "") in {"S-DEC", "S_DEC", "SDEC"}


def _method_is_finish(method: Any) -> bool:
    s = str(method or "").upper()
    if _method_is_dec(method):
        return False
    return any(k in s for k in ("KO", "TKO", "SUB", "SUBMISSION", "DQ"))


def _shifted_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _shifted_expanding_mean(s: pd.Series) -> pd.Series:
    return s.shift(1).expanding(min_periods=1).mean()


def apply_decision_profile_rolling(history: pd.DataFrame) -> pd.DataFrame:
    """Add decision-profile columns onto long fighter history (prior only)."""
    if history is None or history.empty:
        return history
    out = history.copy()
    won = out["won"].fillna(0).astype(int) if "won" in out.columns else pd.Series(0, index=out.index)

    if "is_dec" in out.columns:
        is_dec = out["is_dec"].fillna(0).astype(int)
    elif "method" in out.columns:
        is_dec = out["method"].map(lambda m: int(_method_is_dec(m)))
    else:
        is_dec = pd.Series(0, index=out.index)

    if "method" in out.columns:
        is_split = out["method"].map(lambda m: int(_method_is_split(m)))
        is_finish = out["method"].map(lambda m: int(_method_is_finish(m)))
    else:
        is_ko = out["is_ko"].fillna(0).astype(int) if "is_ko" in out.columns else 0
        is_sub = out["is_sub"].fillna(0).astype(int) if "is_sub" in out.columns else 0
        is_finish = ((is_ko == 1) | (is_sub == 1)).astype(int)
        is_split = pd.Series(0, index=out.index)

    out["_dp_dec_win"] = ((won == 1) & (is_dec == 1)).astype(float)
    out["_dp_dec_loss"] = ((won == 0) & (is_dec == 1)).astype(float)
    out["_dp_split_win"] = ((won == 1) & (is_split == 1)).astype(float)
    out["_dp_split_loss"] = ((won == 0) & (is_split == 1)).astype(float)
    # Among wins: share that went to decision (needs cards) vs finish
    out["_dp_win"] = (won == 1).astype(float)
    out["_dp_dec_win_only"] = out["_dp_dec_win"]
    out["_dp_finish_win"] = ((won == 1) & (is_finish == 1)).astype(float)

    g = out.groupby("fighter", group_keys=False)
    for name, src in (
        ("dec_win_rate", "_dp_dec_win"),
        ("dec_loss_rate", "_dp_dec_loss"),
        ("split_dec_win_rate", "_dp_split_win"),
        ("split_dec_loss_rate", "_dp_split_loss"),
    ):
        out[f"{name}_l5"] = g[src].apply(lambda s: _shifted_rolling_mean(s, 5))
        out[f"{name}_career"] = g[src].apply(_shifted_expanding_mean)

    # decision_finish_share = dec_wins / (dec_wins + finish_wins) among prior wins
    def _share(dec_w: pd.Series, fin_w: pd.Series, window: int | None) -> pd.Series:
        d = dec_w.shift(1)
        f = fin_w.shift(1)
        if window is None:
            d_sum = d.expanding(min_periods=1).sum()
            f_sum = f.expanding(min_periods=1).sum()
        else:
            d_sum = d.rolling(window, min_periods=1).sum()
            f_sum = f.rolling(window, min_periods=1).sum()
        denom = d_sum + f_sum
        return (d_sum / denom).where(denom > 0)

    parts = []
    for _, gdf in out.groupby("fighter", sort=False):
        gdf = gdf.copy()
        gdf["decision_finish_share_l5"] = _share(gdf["_dp_dec_win"], gdf["_dp_finish_win"], 5)
        gdf["decision_finish_share_career"] = _share(
            gdf["_dp_dec_win"], gdf["_dp_finish_win"], None
        )
        parts.append(gdf)
    out = pd.concat(parts).sort_index()
    return out


def attach_decision_profile_diffs(features: pd.DataFrame) -> pd.DataFrame:
    """Compute f1-f2 diffs when side columns exist on a wide matrix."""
    if features is None or features.empty:
        return features
    out = features.copy()

    def _diff(col: str) -> pd.Series:
        a = f"f1_{col}"
        b = f"f2_{col}"
        if a in out.columns and b in out.columns:
            return pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")
        # already on history sides merged with prefixes
        return pd.Series(np.nan, index=out.index)

    for col in DECISION_PROFILE_FIGHTER_FIELDS:
        diff_name = f"{col}_diff"
        if diff_name not in out.columns:
            series = _diff(col)
            if series.notna().any():
                out[diff_name] = series
    return out


def attach_decision_profile_to_wide(
    features: pd.DataFrame,
    *,
    fights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Ensure f1_/f2_ decision-profile fields + diffs on a wide fight matrix.

    Builds a lightweight long history from fights (or feature method/winner),
    applies ``apply_decision_profile_rolling``, pivots to sides, merges.
    """
    if features is None or features.empty:
        return features

    from src.data_loader import clean_fighter_name, load_fights

    out = features.copy()
    fid = config.FIGHT_ID_COLUMN
    date_col = config.DATE_COLUMN
    f1c = "fighter_1" if "fighter_1" in out.columns else "fighter1"
    f2c = "fighter_2" if "fighter_2" in out.columns else "fighter2"

    src = fights if isinstance(fights, pd.DataFrame) and not fights.empty else None
    if src is None:
        try:
            src = load_fights()
        except Exception:
            src = out

    date_src = date_col if date_col in src.columns else (
        "event_date" if "event_date" in src.columns else None
    )
    sf1 = "fighter_1" if "fighter_1" in src.columns else "fighter1"
    sf2 = "fighter_2" if "fighter_2" in src.columns else "fighter2"
    if not date_src or sf1 not in src.columns or sf2 not in src.columns:
        logger.warning("decision_profile attach skipped — missing fighters/date")
        return out

    work = src[
        [c for c in (fid, date_src, sf1, sf2, "method", "winner") if c in src.columns]
    ].copy()
    work[date_src] = pd.to_datetime(work[date_src], errors="coerce")
    work = work.dropna(subset=[date_src]).sort_values(date_src)

    rows: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        a = clean_fighter_name(r.get(sf1))
        b = clean_fighter_name(r.get(sf2))
        if not a or not b:
            continue
        winner = clean_fighter_name(r.get("winner")) if "winner" in work.columns else ""
        method = r.get("method")
        for name in (a, b):
            if not winner:
                won = np.nan
            else:
                won = 1 if name == winner else 0
            rows.append(
                {
                    fid: r.get(fid),
                    date_col: r[date_src],
                    "fighter": name,
                    "won": won,
                    "method": method,
                    "is_dec": int(_method_is_dec(method)),
                }
            )
    hist = pd.DataFrame(rows).dropna(subset=["won"])
    if hist.empty:
        logger.warning("decision_profile attach: empty history")
        return out
    hist["won"] = hist["won"].astype(int)
    hist = apply_decision_profile_rolling(hist)

    side_cols = list(DECISION_PROFILE_FIGHTER_FIELDS)
    keep = [fid, "fighter"] + [c for c in side_cols if c in hist.columns]
    slim = hist[keep].drop_duplicates([fid, "fighter"], keep="last")

    # Drop prior decision-profile side/diff cols so DEC arm uses this module's values
    drop_cols = []
    for x in side_cols:
        for c in (f"f1_{x}", f"f2_{x}", f"{x}_diff"):
            if c in out.columns:
                drop_cols.append(c)
    out = out.drop(columns=drop_cols, errors="ignore")

    left = out[[fid, f1c, f2c]].copy()
    left["_f1k"] = left[f1c].map(clean_fighter_name)
    left["_f2k"] = left[f2c].map(clean_fighter_name)

    s = slim.copy()
    f1_cols = {c: f"f1_{c}" for c in side_cols if c in s.columns}
    f2_cols = {c: f"f2_{c}" for c in side_cols if c in s.columns}
    s1 = s.rename(columns={"fighter": "_f1k", **f1_cols})
    s2 = s.rename(columns={"fighter": "_f2k", **f2_cols})
    merged = left.merge(s1[[fid, "_f1k"] + list(f1_cols.values())], on=[fid, "_f1k"], how="left")
    merged = merged.merge(s2[[fid, "_f2k"] + list(f2_cols.values())], on=[fid, "_f2k"], how="left")

    for c in list(f1_cols.values()) + list(f2_cols.values()):
        if c in merged.columns:
            out[c] = merged[c].to_numpy()

    out = attach_decision_profile_diffs(out)
    for col in side_cols:
        a, b, d = f"f1_{col}", f"f2_{col}", f"{col}_diff"
        if a in out.columns and b in out.columns:
            out[d] = pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")

    present = [c for c in DECISION_PROFILE_DIFF_COLUMNS if c in out.columns]
    logger.info(
        "decision_profile attached diffs=%s/%s",
        len(present),
        len(DECISION_PROFILE_DIFF_COLUMNS),
    )
    return out


def log_decision_profile_coverage(
    df: pd.DataFrame, *, year: int | None = None, label: str = ""
) -> dict[str, Any]:
    work = df
    if year is not None and config.DATE_COLUMN in df.columns:
        dts = pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")
        work = df.loc[dts.dt.year == year]
    n = max(len(work), 1)
    stats: dict[str, Any] = {"n": int(len(work)), "label": label or str(year)}
    for col in DECISION_PROFILE_DIFF_COLUMNS:
        if col in work.columns:
            s = pd.to_numeric(work[col], errors="coerce")
            stats[col] = {
                "nonnull_pct": float(s.notna().mean()),
                "nonzero_pct": float((s.fillna(0) != 0).mean()),
            }
    # fighter-level from long history
    for col in ("dec_win_rate_career", "split_dec_win_rate_career", "decision_finish_share_career"):
        if col in work.columns:
            s = pd.to_numeric(work[col], errors="coerce")
            stats[col] = {
                "nonnull_pct": float(s.notna().mean()),
                "mean": float(s.mean()) if s.notna().any() else float("nan"),
            }
    logger.info("decision_profile coverage %s: %s", label or year, {k: stats[k] for k in list(stats)[:8]})
    return stats
