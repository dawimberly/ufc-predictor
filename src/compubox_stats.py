"""CompuBox-style striking features.

Prefer real CompuBox numbers from ``data/cache/compubox_striking.csv`` when present;
otherwise derive the same fields from Greco/UFCStats bout detail (KD, HEAD/BODY/LEG,
DISTANCE/CLINCH/GROUND).

Reuses the SportsBettingBot ``CompuBoxLine`` differential helpers when importable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

import config
from src.data_loader import _download_greco_csv, _parse_of_fraction, clean_fighter_name

logger = logging.getLogger(__name__)

COMPUBOX_CACHE = config.CACHE_DIR / "compubox_striking.csv"

# Detailed striking fields exposed to feature engineering.
COMPUBOX_STAT_FIELDS = (
    "kd_rate",
    "head_strike_pct",
    "body_strike_pct",
    "leg_strike_pct",
    "distance_strike_pct",
    "clinch_strike_pct",
    "ground_strike_pct",
    "power_proxy",
    "sig_strikes_per_min",
    "sig_strike_acc",
)


@dataclass
class CompuBoxLine:
    """Per-fighter striking snapshot (aligned with SportsBettingBot skeleton)."""

    fighter: str
    sig_strikes_landed: float = 0.0
    sig_strikes_attempted: float = 0.0
    knockdowns: float = 0.0
    head_pct: float = 0.0
    body_pct: float = 0.0
    leg_pct: float = 0.0
    avg_fight_time_min: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.sig_strikes_attempted <= 0:
            return 0.0
        return self.sig_strikes_landed / self.sig_strikes_attempted

    @property
    def volume_per_min(self) -> float:
        if self.avg_fight_time_min <= 0:
            return 0.0
        return self.sig_strikes_landed / self.avg_fight_time_min

    @property
    def power_proxy(self) -> float:
        if self.sig_strikes_landed <= 0:
            return 0.0
        return self.knockdowns / max(self.sig_strikes_landed, 1.0)


def differential(a: CompuBoxLine, b: CompuBoxLine) -> dict[str, float]:
    return {
        "acc_diff": a.accuracy - b.accuracy,
        "volume_diff": a.volume_per_min - b.volume_per_min,
        "power_diff": a.power_proxy - b.power_proxy,
        "head_diff": a.head_pct - b.head_pct,
    }


try:
    # Prefer SportsBettingBot definitions when present on path / sports_bot package.
    from sports_bot.models.compubox import (  # type: ignore
        CompuBoxLine as _SBCompuBoxLine,
        differential as _sb_differential,
    )

    CompuBoxLine = _SBCompuBoxLine  # type: ignore[misc, assignment]
    differential = _sb_differential  # type: ignore[assignment]
except Exception:
    pass


def _safe_share(part: float, whole: float) -> float:
    if whole is None or not np.isfinite(whole) or whole <= 0:
        return np.nan
    if part is None or not np.isfinite(part):
        return np.nan
    return float(part) / float(whole)


def load_compubox_csv(*, force_refresh: bool = False) -> pd.DataFrame:
    """Optional real CompuBox dump. Expected columns include fighter + bout_date."""
    del force_refresh  # cache file is local-only
    if not COMPUBOX_CACHE.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(COMPUBOX_CACHE)
    except Exception as exc:
        logger.warning("CompuBox CSV unreadable: %s", exc)
        return pd.DataFrame()
    if df.empty:
        return df
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for want in (
        "fighter",
        "bout_date",
        "event",
        "kd",
        "sig_landed",
        "sig_attempted",
        "head_landed",
        "body_landed",
        "leg_landed",
        "distance_landed",
        "clinch_landed",
        "ground_landed",
        "fight_minutes",
    ):
        if want in cols:
            rename[cols[want]] = want
        elif want.replace("_", "") in {k.replace("_", ""): k for k in cols}:
            pass
    df = df.rename(columns=rename)
    if "fighter" not in df.columns:
        return pd.DataFrame()
    df["fighter"] = df["fighter"].map(clean_fighter_name)
    if "bout_date" in df.columns:
        df["bout_date"] = pd.to_datetime(df["bout_date"], errors="coerce")
    return df


def _bout_from_compubox_row(row: pd.Series) -> dict[str, float]:
    sig_l = float(pd.to_numeric(row.get("sig_landed"), errors="coerce") or np.nan)
    sig_a = float(pd.to_numeric(row.get("sig_attempted"), errors="coerce") or np.nan)
    kd = float(pd.to_numeric(row.get("kd"), errors="coerce") or 0.0)
    mins = float(pd.to_numeric(row.get("fight_minutes"), errors="coerce") or np.nan)
    head = float(pd.to_numeric(row.get("head_landed"), errors="coerce") or np.nan)
    body = float(pd.to_numeric(row.get("body_landed"), errors="coerce") or np.nan)
    leg = float(pd.to_numeric(row.get("leg_landed"), errors="coerce") or np.nan)
    dist = float(pd.to_numeric(row.get("distance_landed"), errors="coerce") or np.nan)
    clinch = float(pd.to_numeric(row.get("clinch_landed"), errors="coerce") or np.nan)
    ground = float(pd.to_numeric(row.get("ground_landed"), errors="coerce") or np.nan)
    return {
        "kd_rate": kd,
        "head_strike_pct": _safe_share(head, sig_l),
        "body_strike_pct": _safe_share(body, sig_l),
        "leg_strike_pct": _safe_share(leg, sig_l),
        "distance_strike_pct": _safe_share(dist, sig_l),
        "clinch_strike_pct": _safe_share(clinch, sig_l),
        "ground_strike_pct": _safe_share(ground, sig_l),
        "power_proxy": _safe_share(kd, sig_l),
        "sig_strikes_per_min": (sig_l / mins) if mins and mins > 0 and np.isfinite(sig_l) else np.nan,
        "sig_strike_acc": _safe_share(sig_l, sig_a),
        "_source": "compubox",
    }


@lru_cache(maxsize=1)
def load_detailed_bout_striking(*, force_refresh: bool = False) -> pd.DataFrame:
    """Per-fighter per-bout CompuBox-style striking (real CompuBox preferred, else Greco)."""
    rows: list[dict[str, Any]] = []

    # Real CompuBox overrides first
    cb = load_compubox_csv()
    if not cb.empty and "bout_date" in cb.columns:
        for _, row in cb.iterrows():
            if not row.get("fighter") or pd.isna(row.get("bout_date")):
                continue
            stats = _bout_from_compubox_row(row)
            rows.append(
                {
                    "fighter": row["fighter"],
                    "bout_date": row["bout_date"],
                    "event": row.get("event"),
                    **{k: v for k, v in stats.items() if not str(k).startswith("_")},
                    "source": "compubox",
                }
            )

    # Greco / UFCStats detail
    try:
        stats = _download_greco_csv("ufc_fight_stats.csv", force_refresh=force_refresh)
        events = _download_greco_csv("ufc_event_details.csv", force_refresh=force_refresh)
    except Exception as exc:
        logger.warning("Greco detailed striking unavailable: %s", exc)
        stats = pd.DataFrame()
        events = pd.DataFrame()

    if not stats.empty and not events.empty:
        stats = stats.rename(columns=str.upper)
        events = events.rename(columns=str.upper)
        for col in ("SIG.STR.", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"):
            if col not in stats.columns:
                stats[col] = "0 of 0"
        landed, attempted = zip(*stats["SIG.STR."].map(_parse_of_fraction))
        stats["sig_landed"] = landed
        stats["sig_attempted"] = attempted
        for dest, src in (
            ("head_l", "HEAD"),
            ("body_l", "BODY"),
            ("leg_l", "LEG"),
            ("dist_l", "DISTANCE"),
            ("clinch_l", "CLINCH"),
            ("ground_l", "GROUND"),
        ):
            vals, _ = zip(*stats[src].map(_parse_of_fraction))
            stats[dest] = vals
        stats["kd"] = pd.to_numeric(stats.get("KD", 0), errors="coerce").fillna(0.0)
        stats["fighter"] = stats["FIGHTER"].map(clean_fighter_name)
        stats["_rnd"] = (
            stats["ROUND"].astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(3.0)
        )

        bout = stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False).agg(
            sig_landed=("sig_landed", "sum"),
            sig_attempted=("sig_attempted", "sum"),
            head_l=("head_l", "sum"),
            body_l=("body_l", "sum"),
            leg_l=("leg_l", "sum"),
            dist_l=("dist_l", "sum"),
            clinch_l=("clinch_l", "sum"),
            ground_l=("ground_l", "sum"),
            kd=("kd", "sum"),
            max_round=("_rnd", "max"),
        )
        bout["fighter"] = bout["FIGHTER"].map(clean_fighter_name)
        bout["fight_minutes"] = bout["max_round"].fillna(3) * 5.0
        event_dates = events.set_index("EVENT")["DATE"]
        bout["bout_date"] = pd.to_datetime(bout["EVENT"].map(event_dates), errors="coerce")

        # Skip Greco rows already covered by real CompuBox (same fighter+date)
        covered = set()
        if rows:
            for r in rows:
                covered.add((r["fighter"], pd.Timestamp(r["bout_date"]).normalize()))

        for _, row in bout.iterrows():
            if pd.isna(row.get("bout_date")) or not row.get("fighter"):
                continue
            key = (row["fighter"], pd.Timestamp(row["bout_date"]).normalize())
            if key in covered:
                continue
            sig_l = float(row["sig_landed"])
            sig_a = float(row["sig_attempted"])
            mins = float(row["fight_minutes"]) if row["fight_minutes"] else np.nan
            kd = float(row["kd"])
            rows.append(
                {
                    "fighter": row["fighter"],
                    "bout_date": row["bout_date"],
                    "event": row.get("EVENT"),
                    "kd_rate": kd,
                    "head_strike_pct": _safe_share(float(row["head_l"]), sig_l),
                    "body_strike_pct": _safe_share(float(row["body_l"]), sig_l),
                    "leg_strike_pct": _safe_share(float(row["leg_l"]), sig_l),
                    "distance_strike_pct": _safe_share(float(row["dist_l"]), sig_l),
                    "clinch_strike_pct": _safe_share(float(row["clinch_l"]), sig_l),
                    "ground_strike_pct": _safe_share(float(row["ground_l"]), sig_l),
                    "power_proxy": _safe_share(kd, sig_l),
                    "sig_strikes_per_min": (sig_l / mins) if mins and mins > 0 else np.nan,
                    "sig_strike_acc": _safe_share(sig_l, sig_a),
                    "source": "greco",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["fighter", "bout_date", *COMPUBOX_STAT_FIELDS, "source"])
    out = pd.DataFrame(rows)
    out["bout_date"] = pd.to_datetime(out["bout_date"], errors="coerce")
    out = out.dropna(subset=["fighter", "bout_date"])
    return out.sort_values(["fighter", "bout_date"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _pre_fight_table(window: int = 5) -> pd.DataFrame:
    bout = load_detailed_bout_striking()
    if bout.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for fighter, grp in bout.groupby("fighter"):
        grp = grp.sort_values("bout_date").reset_index(drop=True)
        for i in range(len(grp)):
            prior = grp.iloc[max(0, i - window) : i]
            if prior.empty:
                continue
            row: dict[str, Any] = {
                "fighter": fighter,
                "as_of_date": grp.iloc[i]["bout_date"],
                "source": prior["source"].iloc[-1] if "source" in prior.columns else "greco",
            }
            for field in COMPUBOX_STAT_FIELDS:
                if field in prior.columns:
                    row[field] = prior[field].mean()
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    return out.sort_values(["fighter", "as_of_date"]).reset_index(drop=True)


def fill_history_from_compubox(history: pd.DataFrame, *, window: int = 5) -> pd.DataFrame:
    """Fill NaN CompuBox-style columns with leakage-safe pre-fight rolling means."""
    if history.empty or "fighter" not in history.columns:
        return history
    date_col = config.DATE_COLUMN
    if date_col not in history.columns:
        return history

    table = _pre_fight_table(window)
    out = history.reset_index(drop=True).copy()
    for field in COMPUBOX_STAT_FIELDS:
        if field not in out.columns:
            # Don't overwrite core sig_strike fields already managed by Greco unless missing.
            if field in ("sig_strikes_per_min", "sig_strike_acc"):
                continue
            out[field] = np.nan

    if table.empty:
        return out

    filled = 0
    # Detailed fields always; core striking only fills NaNs (never overwrite UFCStats/Greco).
    targets: list[tuple[str, str]] = [
        ("kd_rate", "kd_rate"),
        ("head_strike_pct", "head_strike_pct"),
        ("body_strike_pct", "body_strike_pct"),
        ("leg_strike_pct", "leg_strike_pct"),
        ("distance_strike_pct", "distance_strike_pct"),
        ("clinch_strike_pct", "clinch_strike_pct"),
        ("ground_strike_pct", "ground_strike_pct"),
        ("power_proxy", "power_proxy"),
        ("sig_strike_acc", "sig_strike_acc"),
        ("sig_strikes_per_min", "sig_strikes_per_min_roll"),
    ]

    for fighter, grp in out.groupby("fighter", sort=False):
        g = table[table["fighter"] == clean_fighter_name(str(fighter))].sort_values("as_of_date")
        if g.empty:
            continue
        valid = grp[date_col].notna()
        if not valid.any():
            continue
        idx = g["as_of_date"].searchsorted(grp.loc[valid, date_col].values, side="right") - 1
        mask_idx = grp.index[valid]
        for src_field, target in targets:
            if src_field not in g.columns:
                continue
            if target not in out.columns:
                if target in {
                    "kd_rate",
                    "head_strike_pct",
                    "body_strike_pct",
                    "leg_strike_pct",
                    "distance_strike_pct",
                    "clinch_strike_pct",
                    "ground_strike_pct",
                    "power_proxy",
                }:
                    out[target] = np.nan
                else:
                    continue
            values = np.where(idx >= 0, g[src_field].to_numpy()[np.maximum(idx, 0)], np.nan)
            values = np.where(idx >= 0, values, np.nan)
            for i, row_i in enumerate(mask_idx):
                if pd.isna(out.at[row_i, target]) and pd.notna(values[i]):
                    out.at[row_i, target] = float(values[i])
                    filled += 1
    if filled:
        logger.info("CompuBox-style fill: %s cells", filled)
    return out


def apply_compubox_to_features(features: pd.DataFrame, *, window: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Fill missing f1_/f2_ CompuBox-style columns on wide feature rows."""
    if features.empty:
        return features, np.zeros(0, dtype=bool)
    date_col = config.DATE_COLUMN
    out = features.copy()
    touched = np.zeros(len(out), dtype=bool)
    table = _pre_fight_table(window)
    if table.empty:
        return out, touched

    detail_fields = [
        "kd_rate",
        "head_strike_pct",
        "body_strike_pct",
        "leg_strike_pct",
        "distance_strike_pct",
        "clinch_strike_pct",
        "ground_strike_pct",
        "power_proxy",
    ]
    # Ensure columns exist
    for prefix in ("f1", "f2"):
        for field in detail_fields:
            col = f"{prefix}_{field}"
            if col not in out.columns:
                out[col] = np.nan

    positions = {idx: pos for pos, idx in enumerate(out.index)}
    for prefix, fighter_col in (("f1", "fighter_1"), ("f2", "fighter_2")):
        if fighter_col not in out.columns:
            continue
        for i, row in out.iterrows():
            name = clean_fighter_name(str(row.get(fighter_col) or ""))
            ts = pd.to_datetime(row.get(date_col), errors="coerce")
            if not name or pd.isna(ts):
                continue
            g = table[(table["fighter"] == name) & (table["as_of_date"] < ts.normalize())]
            if g.empty:
                continue
            last = g.iloc[-1]
            for field in detail_fields:
                col = f"{prefix}_{field}"
                if pd.isna(out.at[i, col]) and field in last and pd.notna(last[field]):
                    out.at[i, col] = float(last[field])
                    touched[positions[i]] = True
    return out, touched


def compubox_coverage(fighter_names: list[str] | pd.Series) -> dict[str, float]:
    names = [clean_fighter_name(str(n)) for n in fighter_names if clean_fighter_name(str(n))]
    uniq = set(n for n in names if n)
    if not uniq:
        return {"n": 0.0, "pct_fighters": 0.0, "n_matched": 0.0, "pct_compubox_source": 0.0}
    bout = load_detailed_bout_striking()
    if bout.empty:
        return {"n": float(len(uniq)), "pct_fighters": 0.0, "n_matched": 0.0, "pct_compubox_source": 0.0}
    have = set(bout["fighter"].unique())
    hit = sum(1 for n in uniq if n in have)
    cb_share = float((bout["source"] == "compubox").mean()) if "source" in bout.columns else 0.0
    return {
        "n": float(len(uniq)),
        "pct_fighters": hit / len(uniq),
        "n_matched": float(hit),
        "pct_compubox_source": cb_share,
    }
