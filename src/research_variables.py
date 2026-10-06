"""Leakage-safe predictors that were on disk but not in the training list.

Every column here is known before the bell:
- prior reversals, total-strike pace, significant-strike share, round-3 pace
- home-country from Greco event locations (prior events only)
- prior win/loss streaks, rounds fought, title bouts, split-decision rate
- division rank, pound-for-pound rank, listed weight, empty-arena flag

Market prices and the current fight's result are not used.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data_loader import clean_fighter_name

logger = logging.getLogger(__name__)

# Unranked fighters sit just outside the published top 15.
UNRANKED_RANK = 16.0

RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    "rev_rate_diff",
    "total_strikes_per_min_diff",
    "sig_share_diff",
    "pace_decay_diff",
    "home_country_diff",
    "home_country_rate_diff",
    "event_outside_usa",
    "split_dec_rate_l5_diff",
    "split_dec_rate_career_diff",
    "win_streak_diff",
    "lose_streak_diff",
    "longest_win_streak_diff",
    "rounds_fought_diff",
    "title_bouts_diff",
    "rank_diff",
    "p4p_rank_diff",
    "listed_weight_diff",
    "empty_arena",
    "is_womens",
)

_COUNT_DIFFS = (
    "win_streak_diff",
    "lose_streak_diff",
    "longest_win_streak_diff",
    "rounds_fought_diff",
    "title_bouts_diff",
)
_RATE_DIFFS = (
    "split_dec_rate_l5_diff",
    "split_dec_rate_career_diff",
    "rev_rate_diff",
    "total_strikes_per_min_diff",
    "sig_share_diff",
    "pace_decay_diff",
)


def _name_key(value: object) -> str:
    return clean_fighter_name(value).casefold()


def _event_key(value: object) -> str:
    text = str(value or "").casefold().replace(".", "")
    return re.sub(r"\s+", " ", text).strip()


def _is_split_method(value: object) -> bool:
    text = str(value or "").upper().replace("_", "-")
    return "SPLIT" in text or "S-DEC" in text


def _parse_of_landed(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.extract(r"(\d+)\s*of\s*(\d+)")
    return pd.to_numeric(parts[0], errors="coerce")


def _parse_lbs(value: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    if not match:
        return np.nan
    return float(match.group(1))


def _rank_advantage(own: pd.Series, opp: pd.Series) -> pd.Series:
    """Positive when ``own`` is ranked ahead of ``opp`` (lower number is better)."""
    return opp.fillna(UNRANKED_RANK) - own.fillna(UNRANKED_RANK)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Unreadable %s: %s", path, exc)
        return pd.DataFrame()


def _load_rankings(*, download: bool) -> pd.DataFrame:
    path = config.CACHE_DIR / "ufc-master.csv"
    if path.is_file():
        return _read_csv(path)
    if not download:
        return pd.DataFrame()
    url = str(getattr(config, "ULTIMATE_UFC_DATASET_URL", "") or "")
    if not url:
        return pd.DataFrame()
    try:
        import requests

        response = requests.get(url, timeout=max(getattr(config, "REQUEST_TIMEOUT_SEC", 30), 60))
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return _read_csv(path)
    except Exception as exc:
        logger.warning("Rankings file unavailable: %s", exc)
        return pd.DataFrame()


def _history_diffs(features: pd.DataFrame) -> pd.DataFrame:
    """Prior-only streak, split-decision, rounds, and title diffs. Debut counts are 0."""
    fid = config.FIGHT_ID_COLUMN
    date_col = config.DATE_COLUMN
    work = features.dropna(subset=[fid, date_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    work["_k1"] = work["fighter_1"].map(_name_key)
    work["_k2"] = work["fighter_2"].map(_name_key)
    work["_day"] = work[date_col].dt.normalize()
    hist = work.drop_duplicates(["_day", "_k1", "_k2"], keep="first")
    won = pd.to_numeric(hist.get(config.TARGET_COLUMN), errors="coerce")
    method = hist["method"] if "method" in hist.columns else pd.Series("", index=hist.index)
    rnd = pd.to_numeric(hist["round"], errors="coerce") if "round" in hist.columns else pd.Series(np.nan, index=hist.index)
    title = (
        pd.to_numeric(hist["is_title_fight"], errors="coerce").fillna(0)
        if "is_title_fight" in hist.columns
        else pd.Series(0.0, index=hist.index)
    )

    f1 = pd.DataFrame(
        {
            fid: hist[fid].to_numpy(),
            "_day": hist["_day"].to_numpy(),
            "fighter": hist["_k1"].to_numpy(),
            "won": won.to_numpy(),
            "is_split": method.map(_is_split_method).astype(float).to_numpy(),
            "round": rnd.to_numpy(),
            "is_title": title.to_numpy(),
        }
    )
    f2_won = np.where(np.isfinite(won.to_numpy()), 1.0 - won.to_numpy(), np.nan)
    f2 = f1.copy()
    f2["fighter"] = hist["_k2"].to_numpy()
    f2["won"] = f2_won
    long = pd.concat([f1, f2], ignore_index=True)
    long = long[long["fighter"].astype(bool)]

    pieces: list[pd.DataFrame] = []
    for _fighter, grp in long.groupby("fighter", sort=False):
        grp = grp.sort_values(["_day", fid])
        won_arr = pd.to_numeric(grp["won"], errors="coerce").to_numpy()
        n = len(grp)
        win_s = np.zeros(n)
        lose_s = np.zeros(n)
        longest = np.zeros(n)
        cur_w = cur_l = 0
        best = 0
        for i, value in enumerate(won_arr):
            win_s[i] = cur_w
            lose_s[i] = cur_l
            longest[i] = best
            if value == 1:
                cur_w += 1
                cur_l = 0
                best = max(best, cur_w)
            elif value == 0:
                cur_l += 1
                cur_w = 0
            else:
                cur_w = 0
                cur_l = 0
        split = pd.to_numeric(grp["is_split"], errors="coerce")
        out = grp[[fid, "fighter"]].copy()
        out["win_streak"] = win_s
        out["lose_streak"] = lose_s
        out["longest_win_streak"] = longest
        out["split_l5"] = split.shift(1).rolling(5, min_periods=1).mean().to_numpy()
        out["split_career"] = split.shift(1).expanding(min_periods=1).mean().to_numpy()
        out["rounds_prior"] = pd.to_numeric(grp["round"], errors="coerce").shift(1).expanding(min_periods=1).sum().to_numpy()
        out["titles_prior"] = pd.to_numeric(grp["is_title"], errors="coerce").shift(1).expanding(min_periods=1).sum().to_numpy()
        pieces.append(out)
    if not pieces:
        return pd.DataFrame(columns=[fid, *_COUNT_DIFFS, *_RATE_DIFFS[:2]])
    side = pd.concat(pieces, ignore_index=True)

    def _pull(keys: pd.Series, col: str) -> np.ndarray:
        table = side.drop_duplicates([fid, "fighter"], keep="last").set_index([fid, "fighter"])[col]
        idx = pd.MultiIndex.from_arrays([hist[fid].to_numpy(), keys.to_numpy()])
        return pd.to_numeric(table.reindex(idx).to_numpy(), errors="coerce")

    diffs = pd.DataFrame(
        {
            fid: hist[fid].to_numpy(),
            "_day": hist["_day"].to_numpy(),
            "_k1": hist["_k1"].to_numpy(),
            "_k2": hist["_k2"].to_numpy(),
        }
    )
    diffs["win_streak_diff"] = np.nan_to_num(_pull(hist["_k1"], "win_streak"), nan=0.0) - np.nan_to_num(
        _pull(hist["_k2"], "win_streak"), nan=0.0
    )
    diffs["lose_streak_diff"] = np.nan_to_num(_pull(hist["_k1"], "lose_streak"), nan=0.0) - np.nan_to_num(
        _pull(hist["_k2"], "lose_streak"), nan=0.0
    )
    diffs["longest_win_streak_diff"] = np.nan_to_num(_pull(hist["_k1"], "longest_win_streak"), nan=0.0) - np.nan_to_num(
        _pull(hist["_k2"], "longest_win_streak"), nan=0.0
    )
    diffs["rounds_fought_diff"] = np.nan_to_num(_pull(hist["_k1"], "rounds_prior"), nan=0.0) - np.nan_to_num(
        _pull(hist["_k2"], "rounds_prior"), nan=0.0
    )
    diffs["title_bouts_diff"] = np.nan_to_num(_pull(hist["_k1"], "titles_prior"), nan=0.0) - np.nan_to_num(
        _pull(hist["_k2"], "titles_prior"), nan=0.0
    )
    left_l5, right_l5 = _pull(hist["_k1"], "split_l5"), _pull(hist["_k2"], "split_l5")
    left_car, right_car = _pull(hist["_k1"], "split_career"), _pull(hist["_k2"], "split_career")
    diffs["split_dec_rate_l5_diff"] = left_l5 - right_l5
    diffs["split_dec_rate_career_diff"] = left_car - right_car
    return diffs.drop_duplicates(["_day", "_k1", "_k2"], keep="first")


def _greco_prior_table(stats: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Entering (pre-bout) and after-bout rolling means, one row per fighter-bout."""
    if stats is None or stats.empty or events is None or events.empty:
        return pd.DataFrame()
    raw = stats.rename(columns=str.upper).copy()
    ev = events.rename(columns=str.upper).copy()
    needed = {"EVENT", "BOUT", "ROUND", "FIGHTER", "REV.", "SIG.STR.", "TOTAL STR."}
    if not needed.issubset(raw.columns) or "EVENT" not in ev.columns or "DATE" not in ev.columns:
        return pd.DataFrame()
    raw["fighter_key"] = raw["FIGHTER"].map(_name_key)
    raw["rev"] = pd.to_numeric(raw["REV."], errors="coerce").fillna(0.0)
    raw["sig_l"] = _parse_of_landed(raw["SIG.STR."])
    raw["tot_l"] = _parse_of_landed(raw["TOTAL STR."])
    raw["round_num"] = pd.to_numeric(raw["ROUND"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    raw = raw[raw["fighter_key"].astype(bool)]
    bout = raw.groupby(["EVENT", "BOUT", "fighter_key"], as_index=False).agg(
        rev=("rev", "sum"),
        sig_l=("sig_l", "sum"),
        tot_l=("tot_l", "sum"),
        max_round=("round_num", "max"),
    )
    r1 = (
        raw.loc[raw["round_num"] == 1]
        .groupby(["EVENT", "BOUT", "fighter_key"], as_index=False)["sig_l"]
        .sum()
        .rename(columns={"sig_l": "r1"})
    )
    late = (
        raw.loc[raw["round_num"] >= 3]
        .groupby(["EVENT", "BOUT", "fighter_key"], as_index=False)["sig_l"]
        .sum()
        .rename(columns={"sig_l": "late"})
    )
    bout = bout.merge(r1, on=["EVENT", "BOUT", "fighter_key"], how="left")
    bout = bout.merge(late, on=["EVENT", "BOUT", "fighter_key"], how="left")
    minutes = bout["max_round"].fillna(3).clip(lower=1) * 5.0
    bout["total_strikes_per_min"] = bout["tot_l"] / minutes
    bout["sig_share"] = np.where(bout["tot_l"] > 0, bout["sig_l"] / bout["tot_l"], np.nan)
    bout["pace_decay"] = np.where(bout["r1"] > 0, bout["late"] / bout["r1"], np.nan)
    dates = ev.assign(_ek=ev["EVENT"].map(_event_key)).drop_duplicates("_ek", keep="last")
    bout["_ek"] = bout["EVENT"].map(_event_key)
    bout = bout.merge(dates[["_ek", "DATE"]], on="_ek", how="left")
    bout["day"] = pd.to_datetime(bout["DATE"], errors="coerce").dt.normalize()
    bout = bout.dropna(subset=["fighter_key", "day"]).sort_values(["fighter_key", "day"])
    metric = {
        "rev": "rev_rate",
        "total_strikes_per_min": "total_strikes_per_min",
        "sig_share": "sig_share",
        "pace_decay": "pace_decay",
    }
    grouped = bout.groupby("fighter_key", sort=False)
    for src, name in metric.items():
        series = pd.to_numeric(bout[src], errors="coerce")
        bout[f"enter_{name}"] = grouped[src].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        bout[f"after_{name}"] = series.groupby(bout["fighter_key"]).transform(
            lambda s: s.expanding(min_periods=1).mean()
        )
    keep = ["fighter_key", "day"] + [f"enter_{n}" for n in metric.values()] + [f"after_{n}" for n in metric.values()]
    return bout[keep].drop_duplicates(["fighter_key", "day"], keep="last")


def _map_greco(features: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    fid = config.FIGHT_ID_COLUMN
    if prior is None or prior.empty:
        return pd.DataFrame(columns=[fid, *_RATE_DIFFS[2:]])
    day = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce").dt.normalize()
    frames = []
    for side, name_col in (("1", "fighter_1"), ("2", "fighter_2")):
        left = pd.DataFrame(
            {
                fid: features[fid].to_numpy(),
                "fighter_key": features[name_col].map(_name_key).to_numpy(),
                "day": day.to_numpy(),
            }
        )
        left.loc[~left["fighter_key"].astype(bool), "fighter_key"] = np.nan
        exact = left.merge(prior, on=["fighter_key", "day"], how="left", indicator=True)
        hit = exact["_merge"].eq("both")
        values = pd.DataFrame({fid: left[fid]})
        for name in ("rev_rate", "total_strikes_per_min", "sig_share", "pace_decay"):
            values[name] = exact[f"enter_{name}"].to_numpy()
        miss_mask = ~hit.to_numpy()
        miss = left.loc[miss_mask].dropna(subset=["fighter_key", "day"]).sort_values(["day", "fighter_key"])
        if not miss.empty:
            after_names = ("rev_rate", "total_strikes_per_min", "sig_share", "pace_decay")
            after_cols = ["fighter_key", "day"] + [f"after_{n}" for n in after_names]
            right = prior[after_cols].dropna(subset=["fighter_key", "day"]).sort_values(["day", "fighter_key"])
            filled = pd.merge_asof(
                miss,
                right,
                on="day",
                by="fighter_key",
                direction="backward",
                allow_exact_matches=False,
            )
            filled = filled.drop_duplicates(fid, keep="last").set_index(fid)
            miss_ids = values.loc[miss_mask, fid]
            for name in after_names:
                values.loc[miss_mask, name] = miss_ids.map(filled[f"after_{name}"]).to_numpy()
        values = values.groupby(fid, as_index=False).first()
        values = values.rename(columns={c: f"s{side}_{c}" for c in values.columns if c != fid})
        frames.append(values)
    wide = frames[0].merge(frames[1], on=fid, how="outer")
    out = pd.DataFrame({fid: wide[fid]})
    for name, dest in (
        ("rev_rate", "rev_rate_diff"),
        ("total_strikes_per_min", "total_strikes_per_min_diff"),
        ("sig_share", "sig_share_diff"),
        ("pace_decay", "pace_decay_diff"),
    ):
        out[dest] = pd.to_numeric(wide[f"s1_{name}"], errors="coerce") - pd.to_numeric(wide[f"s2_{name}"], errors="coerce")
    return out


def _attach_location(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return features
    ev = events.rename(columns=str.upper)
    if "EVENT" not in ev.columns or "LOCATION" not in ev.columns:
        return features
    out = features.copy()
    ev = ev.copy()
    ev["_ek"] = ev["EVENT"].map(_event_key)
    ev["day"] = pd.to_datetime(ev["DATE"], errors="coerce").dt.normalize() if "DATE" in ev.columns else pd.NaT
    by_name = ev.drop_duplicates("_ek", keep="last").set_index("_ek")["LOCATION"]
    event_col = "event" if "event" in out.columns else ("event_name" if "event_name" in out.columns else None)
    loc = pd.Series(pd.NA, index=out.index, dtype="object")
    if event_col:
        loc = out[event_col].map(_event_key).map(by_name).astype("object")
    if "DATE" in ev.columns and config.DATE_COLUMN in out.columns:
        by_day = (
            ev.dropna(subset=["day"])
            .groupby("day")["LOCATION"]
            .agg(lambda s: s.iloc[0] if s.nunique() == 1 else pd.NA)
        )
        days = pd.to_datetime(out[config.DATE_COLUMN], errors="coerce").dt.normalize()
        day_loc = days.map(by_day).astype("object")
        # Assign into an object series. Series.where would cast city names to NaN.
        blank = loc.isna() | loc.astype(str).str.strip().str.lower().isin(["", "nan", "none", "<na>"])
        loc = loc.copy()
        loc.loc[blank.to_numpy()] = day_loc.loc[blank.to_numpy()].to_numpy()
    text = loc.astype(str).str.strip()
    bad = loc.isna() | text.str.lower().isin(["", "nan", "none", "<na>", "nat"])
    loc = text.astype("object")
    loc.loc[bad.to_numpy()] = pd.NA
    out["location"] = loc
    return out


def _rank_and_weight(features: pd.DataFrame, rankings: pd.DataFrame, tott: pd.DataFrame) -> pd.DataFrame:
    fid = config.FIGHT_ID_COLUMN
    out = pd.DataFrame({fid: features[fid].to_numpy()})
    out["rank_diff"] = np.nan
    out["p4p_rank_diff"] = np.nan
    out["listed_weight_diff"] = np.nan
    out["empty_arena"] = 0.0
    if rankings is not None and not rankings.empty and {"R_fighter", "B_fighter", "date"}.issubset(rankings.columns):
        rank = rankings.copy()
        rank["day"] = pd.to_datetime(rank["date"], errors="coerce").dt.normalize()
        rank["r_key"] = rank["R_fighter"].map(_name_key)
        rank["b_key"] = rank["B_fighter"].map(_name_key)
        rank = rank[(rank["r_key"].astype(bool)) & (rank["b_key"].astype(bool))]
        rank = rank.drop_duplicates(["day", "r_key", "b_key"], keep="last")
        rank["_hit"] = 1
        for col, default in (
            ("R_match_weightclass_rank", np.nan),
            ("B_match_weightclass_rank", np.nan),
            ("R_Pound-for-Pound_rank", np.nan),
            ("B_Pound-for-Pound_rank", np.nan),
            ("R_Weight_lbs", np.nan),
            ("B_Weight_lbs", np.nan),
            ("empty_arena", np.nan),
        ):
            if col not in rank.columns:
                rank[col] = default
            else:
                rank[col] = pd.to_numeric(rank[col], errors="coerce")
        left = pd.DataFrame(
            {
                fid: features[fid].to_numpy(),
                "day": pd.to_datetime(features[config.DATE_COLUMN], errors="coerce").dt.normalize().to_numpy(),
                "k1": features["fighter_1"].map(_name_key).to_numpy(),
                "k2": features["fighter_2"].map(_name_key).to_numpy(),
            }
        )
        red = left.merge(rank, left_on=["day", "k1", "k2"], right_on=["day", "r_key", "b_key"], how="left")
        blue = left.merge(rank, left_on=["day", "k1", "k2"], right_on=["day", "b_key", "r_key"], how="left")
        use_red = red["_hit"].eq(1)
        f1_rank = red["R_match_weightclass_rank"].where(use_red, blue["B_match_weightclass_rank"])
        f2_rank = red["B_match_weightclass_rank"].where(use_red, blue["R_match_weightclass_rank"])
        f1_p4p = red["R_Pound-for-Pound_rank"].where(use_red, blue["B_Pound-for-Pound_rank"])
        f2_p4p = red["B_Pound-for-Pound_rank"].where(use_red, blue["R_Pound-for-Pound_rank"])
        f1_lbs = red["R_Weight_lbs"].where(use_red, blue["B_Weight_lbs"])
        f2_lbs = red["B_Weight_lbs"].where(use_red, blue["R_Weight_lbs"])
        arena = red["empty_arena"].where(use_red, blue["empty_arena"])
        matched = use_red | blue["_hit"].eq(1)
        out["rank_diff"] = _rank_advantage(f1_rank, f2_rank).where(matched)
        out["p4p_rank_diff"] = _rank_advantage(f1_p4p, f2_p4p).where(matched)
        both_lbs = f1_lbs.notna() & f2_lbs.notna()
        out["listed_weight_diff"] = (f1_lbs - f2_lbs).where(both_lbs)
        out["empty_arena"] = arena.fillna(0).where(matched, 0).astype(float)
        out[fid] = left[fid].to_numpy()
    if tott is not None and not tott.empty:
        tape = tott.rename(columns=str.upper)
        if {"FIGHTER", "WEIGHT"}.issubset(tape.columns):
            tape = tape.copy()
            tape["fighter_key"] = tape["FIGHTER"].map(_name_key)
            tape["lbs"] = tape["WEIGHT"].map(_parse_lbs)
            tape = tape.dropna(subset=["fighter_key"]).drop_duplicates("fighter_key", keep="last")
            lookup = tape.set_index("fighter_key")["lbs"]
            missing = out["listed_weight_diff"].isna().to_numpy()
            if missing.any():
                w1 = features["fighter_1"].map(_name_key).map(lookup).to_numpy()
                w2 = features["fighter_2"].map(_name_key).map(lookup).to_numpy()
                fallback = np.where(np.isfinite(w1) & np.isfinite(w2), w1 - w2, np.nan)
                current = np.array(out["listed_weight_diff"], dtype=float, copy=True)
                current[missing] = fallback[missing]
                out["listed_weight_diff"] = current
    return out.drop_duplicates(fid, keep="first")


def attach_research_variables(
    features: pd.DataFrame,
    *,
    stats: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    tott: pd.DataFrame | None = None,
    download: bool = False,
) -> pd.DataFrame:
    """Add research columns. Missing sources leave those columns empty."""
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        return features
    fid = config.FIGHT_ID_COLUMN
    needed = {fid, config.DATE_COLUMN, "fighter_1", "fighter_2"}
    if not needed.issubset(features.columns):
        return features

    greco_dir = Path(getattr(config, "UFCSTATS_GRECO_CACHE_DIR", config.CACHE_DIR / "ufcstats_greco"))
    if stats is None:
        stats = _read_csv(greco_dir / "ufc_fight_stats.csv")
    if events is None:
        events = _read_csv(greco_dir / "ufc_event_details.csv")
    if rankings is None:
        rankings = _load_rankings(download=download)
    if tott is None:
        tott = _read_csv(greco_dir / "ufc_fighter_tott.csv")

    out = _attach_location(features, events)
    try:
        from src.home_country import attach_home_country_features

        out = attach_home_country_features(out)
    except Exception as exc:
        logger.warning("Home-country attach skipped: %s", exc)

    history = _history_diffs(out)
    if history is not None and not history.empty:
        out["_day"] = pd.to_datetime(out[config.DATE_COLUMN], errors="coerce").dt.normalize()
        out["_k1"] = out["fighter_1"].map(_name_key)
        out["_k2"] = out["fighter_2"].map(_name_key)
        keep = [c for c in history.columns if c not in {fid, "_day", "_k1", "_k2"}]
        out = out.drop(columns=[c for c in keep if c in out.columns], errors="ignore")
        out = out.merge(history[["_day", "_k1", "_k2", *keep]], on=["_day", "_k1", "_k2"], how="left")
        out = out.drop(columns=["_day", "_k1", "_k2"], errors="ignore")

    pieces = [_map_greco(out, _greco_prior_table(stats, events)), _rank_and_weight(out, rankings, tott)]
    for piece in pieces:
        if piece is None or piece.empty or fid not in piece.columns:
            continue
        add_cols = [c for c in piece.columns if c != fid]
        out = out.drop(columns=[c for c in add_cols if c in out.columns], errors="ignore")
        out = out.merge(piece[[fid, *add_cols]], on=fid, how="left")

    wc = out["weight_class"] if "weight_class" in out.columns else pd.Series("", index=out.index)
    out["is_womens"] = wc.astype(str).str.contains(r"women|female", case=False, regex=True).astype(float)
    country = out["event_country"] if "event_country" in out.columns else pd.Series("", index=out.index)
    country_text = country.fillna("").astype(str).str.strip()
    out["event_outside_usa"] = np.where(country_text.eq(""), np.nan, country_text.ne("usa").astype(float))
    if "scheduled_rounds" in out.columns:
        rounds = pd.to_numeric(out["scheduled_rounds"], errors="coerce")
        out["is_five_round"] = np.where(rounds >= 5, 1.0, 0.0)
    if "stance_matchup" in out.columns:
        out["path_stance_mismatch"] = pd.to_numeric(out["stance_matchup"], errors="coerce")

    for col in RESEARCH_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    filled = {col: float(pd.to_numeric(out[col], errors="coerce").notna().mean()) for col in RESEARCH_FEATURE_COLUMNS}
    logger.info("Research variable coverage: %s", ", ".join(f"{k}={v:.0%}" for k, v in filled.items()))
    return _dedupe_bouts(out)


def _dedupe_bouts(features: pd.DataFrame) -> pd.DataFrame:
    """One row per date and fighter pair.

    The feature file stores some bouts twice (two fight ids, same result).
    An overturned official result beats the in-cage method. A row with both
    prices beats a row without prices.
    """
    fid_cols = [config.DATE_COLUMN, "fighter_1", "fighter_2"]
    if any(c not in features.columns for c in fid_cols):
        return features
    work = features.copy()
    work["_day"] = pd.to_datetime(work[config.DATE_COLUMN], errors="coerce").dt.normalize()
    work["_k1"] = work["fighter_1"].map(_name_key)
    work["_k2"] = work["fighter_2"].map(_name_key)
    method = work["method"] if "method" in work.columns else ""
    work["_overturn"] = pd.Series(method, index=work.index).astype(str).str.contains("overturn", case=False).astype(int)
    if {"f1_odds", "f2_odds"}.issubset(work.columns):
        priced = pd.to_numeric(work["f1_odds"], errors="coerce").notna() & pd.to_numeric(work["f2_odds"], errors="coerce").notna()
        work["_priced"] = priced.astype(int)
    else:
        work["_priced"] = 0
    work["_pref"] = work["_overturn"] * 2 + work["_priced"]
    work = work.sort_values(["_day", "_k1", "_k2", "_pref"], ascending=[True, True, True, False], kind="mergesort")
    work = work.drop_duplicates(["_day", "_k1", "_k2"], keep="first")
    return work.drop(columns=["_day", "_k1", "_k2", "_overturn", "_priced", "_pref"], errors="ignore")
