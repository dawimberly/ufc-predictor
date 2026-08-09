"""Controversial fight / referee catalog for integrity research.

Uses Greco ``REFEREE`` + method labels (split / majority / DQ / doctor /
overturned / CNC). Not a LightGBM feature. Live skip only when a row already
carries ``referee`` and that ref is flagged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.data_loader import clean_fighter_name

logger = logging.getLogger(__name__)

GRECO_RESULTS = config.CACHE_DIR / "ufcstats_greco" / "ufc_fight_results.csv"
CATALOG_PATH = Path(config.DATA_DIR) / "controversial_catalog.json"
REPORTS = Path(config.DATA_DIR) / "reports"

# Method tokens treated as historically "messy" outcomes (post-hoc labels).
CONTROVERSY_METHOD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"SPLIT|S-DEC|S_DEC", "split_decision"),
    (r"MAJORITY|M-DEC|M_DEC", "majority_decision"),
    (r"\bDQ\b|DISQUAL", "dq"),
    (r"DOCTOR|TKO - DOCTOR", "doctor_stoppage"),
    (r"OVERTURN", "overturned"),
    (r"COULD NOT CONTINUE|CNC|NO CONTEST|\bNC\b", "could_not_continue"),
)


@dataclass(frozen=True)
class ControversyLabel:
    is_controversial: bool
    kinds: tuple[str, ...]
    referee: str = ""
    ref_flagged: bool = False


def method_controversy_kinds(method: Any) -> tuple[str, ...]:
    text = str(method or "").upper()
    if not text.strip():
        return ()
    hits: list[str] = []
    for pat, kind in CONTROVERSY_METHOD_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(kind)
    return tuple(hits)


def is_controversial_method(method: Any) -> bool:
    return bool(method_controversy_kinds(method))


def _bout_key(a: Any, b: Any) -> str:
    x = clean_fighter_name(a).casefold()
    y = clean_fighter_name(b).casefold()
    if not x or not y:
        return ""
    return "|".join(sorted([x, y]))


def _parse_greco_bout(bout: Any) -> tuple[str, str]:
    text = str(bout or "")
    if " vs." in text.lower():
        parts = re.split(r"\s+vs\.?\s+", text, maxsplit=1, flags=re.IGNORECASE)
    elif " vs " in text.lower():
        parts = re.split(r"\s+vs\.?\s+", text, maxsplit=1, flags=re.IGNORECASE)
    else:
        return "", ""
    if len(parts) != 2:
        return "", ""
    return clean_fighter_name(parts[0]), clean_fighter_name(parts[1])


@lru_cache(maxsize=2)
def load_greco_referee_table(path_str: str = "", mtime_ns: int = 0) -> pd.DataFrame:
    path = Path(path_str) if path_str else GRECO_RESULTS
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "BOUT" not in df.columns or "REFEREE" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        f1, f2 = _parse_greco_bout(r.get("BOUT"))
        key = _bout_key(f1, f2)
        if not key:
            continue
        method = str(r.get("METHOD") or "")
        kinds = method_controversy_kinds(method)
        rows.append(
            {
                "bout_key": key,
                "fighter_1": f1,
                "fighter_2": f2,
                "referee": str(r.get("REFEREE") or "").strip(),
                "method": method,
                "event": str(r.get("EVENT") or ""),
                "controversial": bool(kinds),
                "controversy_kinds": "|".join(kinds),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One row per bout_key (last wins)
    return out.drop_duplicates("bout_key", keep="last").reset_index(drop=True)


def attach_referee_and_controversy(df: pd.DataFrame) -> pd.DataFrame:
    """Add referee + controversy columns via Greco bout join."""
    out = df.copy()
    f1c = "fighter_1" if "fighter_1" in out.columns else "fighter1"
    f2c = "fighter_2" if "fighter_2" in out.columns else "fighter2"
    if f1c not in out.columns or f2c not in out.columns:
        out["referee"] = ""
        out["controversy_kinds"] = ""
        out["is_controversial_fight"] = False
        out["ref_flagged"] = False
        return out

    greco_path = GRECO_RESULTS
    mtime = int(greco_path.stat().st_mtime_ns) if greco_path.is_file() else 0
    greco = load_greco_referee_table(str(greco_path), mtime)
    out["_bout_key"] = [
        _bout_key(a, b) for a, b in zip(out[f1c], out[f2c])
    ]
    if "method" in out.columns:
        out["controversy_kinds"] = out["method"].map(
            lambda m: "|".join(method_controversy_kinds(m))
        )
    else:
        out["controversy_kinds"] = ""

    if not greco.empty:
        ref_map = greco.set_index("bout_key")["referee"].to_dict()
        g_kinds = greco.set_index("bout_key")["controversy_kinds"].to_dict()
        out["referee"] = out["_bout_key"].map(lambda k: ref_map.get(k, ""))
        # Prefer Greco kinds when method missing
        miss = out["controversy_kinds"].fillna("").eq("")
        out.loc[miss, "controversy_kinds"] = out.loc[miss, "_bout_key"].map(
            lambda k: g_kinds.get(k, "")
        )
    else:
        out["referee"] = out.get("referee", pd.Series("", index=out.index))

    out["is_controversial_fight"] = out["controversy_kinds"].fillna("").astype(str).str.len() > 0
    flagged = set(flagged_referee_names())
    out["ref_flagged"] = out["referee"].fillna("").map(
        lambda r: clean_fighter_name(r).casefold() in flagged if r else False
    )
    out = out.drop(columns=["_bout_key"], errors="ignore")
    return out


def referee_controversy_stats(
    *,
    min_bouts: int = 40,
    prior_strength: float = 40.0,
) -> pd.DataFrame:
    """
    Empirical-Bayes controversy rate by referee.

    Flags refs whose shrunk rate exceeds league mean by >= 2 posterior SE
    and raw rate also above mean (elevated messy outcomes).
    """
    greco_path = GRECO_RESULTS
    mtime = int(greco_path.stat().st_mtime_ns) if greco_path.is_file() else 0
    g = load_greco_referee_table(str(greco_path), mtime)
    if g.empty:
        return pd.DataFrame()

    work = g[g["referee"].fillna("").str.strip().ne("")].copy()
    league = float(work["controversial"].mean()) if len(work) else 0.0
    rows: list[dict[str, Any]] = []
    for ref, grp in work.groupby("referee"):
        n = int(len(grp))
        k = int(grp["controversial"].sum())
        raw = k / n if n else 0.0
        # Beta-binomial style shrink toward league mean
        post_k = k + prior_strength * league
        post_n = n + prior_strength
        shrunk = post_k / post_n
        # Posterior variance of mean ~ p(1-p)/(n+prior)
        se = float(np.sqrt(max(shrunk * (1.0 - shrunk), 1e-9) / post_n))
        z = (shrunk - league) / se if se > 0 else 0.0
        elevated = bool(n >= min_bouts and z >= 2.0 and raw > league)
        depressed = bool(n >= min_bouts and z <= -2.0 and raw < league)
        watchlist = bool(n >= min_bouts and z >= 1.0 and raw > league)
        rows.append(
            {
                "referee": ref,
                "n_bouts": n,
                "n_controversial": k,
                "raw_rate": raw,
                "shrunk_rate": shrunk,
                "league_rate": league,
                "z_vs_league": z,
                "elevated": elevated,
                "depressed": depressed,
                "watchlist": watchlist,
                "flag": elevated,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["flag", "z_vs_league", "n_bouts"], ascending=[False, False, False]
    )
    return out.reset_index(drop=True)


def flagged_referee_names() -> frozenset[str]:
    catalog = _load_catalog()
    names = {
        clean_fighter_name(r).casefold()
        for r in (catalog.get("flagged_referees") or [])
        if r
    }
    return frozenset(names)


@lru_cache(maxsize=2)
def _load_catalog_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("controversial catalog load failed: %s", exc)
        return {}


def _load_catalog() -> dict[str, Any]:
    p = CATALOG_PATH
    mtime = int(p.stat().st_mtime_ns) if p.is_file() else 0
    return _load_catalog_cached(str(p), mtime)


def build_and_save_catalog(
    *,
    min_bouts: int = 40,
    path: Path | None = None,
) -> dict[str, Any]:
    """Compute elevated refs + method rules; write ``data/controversial_catalog.json``."""
    stats = referee_controversy_stats(min_bouts=min_bouts)
    elevated = stats[stats["flag"]].copy() if not stats.empty else pd.DataFrame()
    watch = (
        stats[stats["watchlist"]].copy()
        if (not stats.empty and "watchlist" in stats.columns)
        else pd.DataFrame()
    )
    flagged_refs = elevated["referee"].tolist() if not elevated.empty else []
    watch_refs = watch["referee"].tolist() if not watch.empty else []

    catalog = {
        "version": 1,
        "updated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        "league_controversy_rate": float(stats["league_rate"].iloc[0]) if len(stats) else None,
        "min_bouts": min_bouts,
        "method_rules": [k for _, k in CONTROVERSY_METHOD_PATTERNS],
        "flagged_referees": flagged_refs,
        "watchlist_referees": watch_refs,
        "referee_stats_elevated": elevated.head(25).to_dict(orient="records")
        if not elevated.empty
        else [],
        "referee_stats_watchlist": watch.head(25).to_dict(orient="records")
        if not watch.empty
        else [],
        "notes": (
            "flagged_referees: EB-shrunk rate >= league + 2 SE, n>=min_bouts (live skip if referee known). "
            "watchlist_referees: >= +1 SE (research only). "
            "Controversy methods: split/majority/DQ/doctor/overturned/CNC. Not a model feature."
        ),
        "user_fight_flags": [
            {
                "fight": "Louie Sutherland vs Jose Montanha",
                "date": "2026-08-08",
                "reason": "user_integrity_flag",
            }
        ],
    }
    out = path or CATALOG_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    _load_catalog_cached.cache_clear()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not stats.empty:
        stats.to_csv(REPORTS / "referee_controversy_rates.csv", index=False)
    logger.info(
        "Controversial catalog: %s elevated / %s watchlist refs (of %s), league_rate=%.3f",
        len(flagged_refs),
        len(watch_refs),
        len(stats),
        catalog["league_controversy_rate"] or 0.0,
    )
    return catalog


def should_skip_for_referee(referee: Any) -> tuple[bool, str]:
    name = clean_fighter_name(str(referee or ""))
    if not name:
        return False, ""
    if name.casefold() in flagged_referee_names():
        return True, f"controversial_ref:{name}"
    return False, ""


def label_row(row: pd.Series | dict[str, Any]) -> ControversyLabel:
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    kinds = method_controversy_kinds(get("method"))
    if not kinds and get("controversy_kinds"):
        kinds = tuple(
            x for x in str(get("controversy_kinds")).split("|") if x
        )
    ref = str(get("referee") or "").strip()
    ref_flagged = False
    if ref:
        ref_flagged = clean_fighter_name(ref).casefold() in flagged_referee_names()
    elif get("ref_flagged") is not None:
        ref_flagged = bool(get("ref_flagged"))
    return ControversyLabel(
        is_controversial=bool(kinds) or ref_flagged,
        kinds=kinds,
        referee=ref,
        ref_flagged=ref_flagged,
    )
