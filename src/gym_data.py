"""Fighter gym profiles: load data/gyms.csv and attach to fight/prediction frames."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

GYM_SIDE_COLS = (
    "gym",
    "gym_location",
    "gym_strengths",
    "gym_notes",
    "local_advantage",
)


def _gyms_csv_path() -> Path:
    path = getattr(config, "GYMS_CSV", None)
    if path is not None:
        return Path(path)
    return Path(config.DATA_DIR) / "gyms.csv"


def _clean_name(name: Any) -> str:
    from src.data_loader import clean_fighter_name

    return clean_fighter_name(name)


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


@lru_cache(maxsize=4)
def load_gym_profiles(path_str: str | None = None) -> pd.DataFrame:
    """Load gyms.csv keyed by cleaned fighter_name."""
    path = Path(path_str) if path_str else _gyms_csv_path()
    if not path.is_file():
        logger.debug("Gyms CSV missing: %s", path)
        return pd.DataFrame(
            columns=["fighter_name", "gym", "location", "strengths", "notes", "fighter_key"]
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read gyms.csv: %s", exc)
        return pd.DataFrame(
            columns=["fighter_name", "gym", "location", "strengths", "notes", "fighter_key"]
        )

    colmap = {
        "fighter": "fighter_name",
        "name": "fighter_name",
        "gym_name": "gym",
        "gym_location": "location",
        "primary_strengths": "strengths",
        "camp_notes": "notes",
        "recent_notes": "notes",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    for col in ("fighter_name", "gym", "location", "strengths", "notes"):
        if col not in df.columns:
            df[col] = ""
    df["fighter_name"] = df["fighter_name"].astype(str)
    df["fighter_key"] = df["fighter_name"].map(_clean_name)
    df = df[df["fighter_key"].astype(bool)].drop_duplicates("fighter_key", keep="last")
    logger.info("Loaded %s gym profiles from %s", len(df), path)
    return df.reset_index(drop=True)


def clear_gym_cache() -> None:
    load_gym_profiles.cache_clear()


def lookup_gym(fighter_name: str, profiles: pd.DataFrame | None = None) -> dict[str, str]:
    """Return gym dict for a fighter (empty strings when unknown)."""
    profiles = profiles if profiles is not None else load_gym_profiles()
    empty = {"gym": "", "location": "", "strengths": "", "notes": ""}
    if profiles is None or profiles.empty:
        return empty
    key = _clean_name(fighter_name)
    if not key:
        return empty

    hit = profiles.loc[profiles["fighter_key"] == key]
    if hit.empty:
        # Fuzzy: last-name + token overlap via data_loader helper
        try:
            from src.data_loader import _fighters_same_person

            for _, row in profiles.iterrows():
                if _fighters_same_person(key, str(row.get("fighter_key") or "")):
                    hit = profiles.loc[[row.name]]
                    break
        except Exception:
            pass
    if hit.empty:
        return empty
    row = hit.iloc[0]
    return {
        "gym": str(row.get("gym") or "").strip(),
        "location": str(row.get("location") or "").strip(),
        "strengths": str(row.get("strengths") or "").strip(),
        "notes": str(row.get("notes") or "").strip(),
    }


# USPS → full state so "Denver, CO" overlaps gym "Englewood Colorado".
_US_STATE_ABBREV = {
    "al": "alabama",
    "ak": "alaska",
    "az": "arizona",
    "ar": "arkansas",
    "ca": "california",
    "co": "colorado",
    "ct": "connecticut",
    "de": "delaware",
    "fl": "florida",
    "ga": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois",
    "in": "indiana",
    "ia": "iowa",
    "ks": "kansas",
    "ky": "kentucky",
    "la": "louisiana",
    "me": "maine",
    "md": "maryland",
    "ma": "massachusetts",
    "mi": "michigan",
    "mn": "minnesota",
    "ms": "mississippi",
    "mo": "missouri",
    "mt": "montana",
    "ne": "nebraska",
    "nv": "nevada",
    "nh": "new hampshire",
    "nj": "new jersey",
    "nm": "new mexico",
    "ny": "new york",
    "nc": "north carolina",
    "nd": "north dakota",
    "oh": "ohio",
    "ok": "oklahoma",
    "or": "oregon",
    "pa": "pennsylvania",
    "ri": "rhode island",
    "sc": "south carolina",
    "sd": "south dakota",
    "tn": "tennessee",
    "tx": "texas",
    "ut": "utah",
    "vt": "vermont",
    "va": "virginia",
    "wa": "washington",
    "wv": "west virginia",
    "wi": "wisconsin",
    "wy": "wyoming",
    "dc": "district of columbia",
}


def _location_tokens(text: str) -> set[str]:
    """Normalize location string into comparable tokens (expand US state abbrevs)."""
    raw = _norm_token(text)
    if not raw:
        return set()
    stop = {
        "usa",
        "uk",
        "uae",
        "the",
        "and",
        "mma",
        "team",
        "gym",
        "fight",
        "club",
        "united",
        "states",
        "of",
        "america",
    }
    out: set[str] = set()
    for t in raw.split():
        if t in stop:
            continue
        expanded = _US_STATE_ABBREV.get(t, t)
        for part in str(expanded).split():
            if part and part not in stop:
                out.add(part)
    return out


def _location_overlap(gym_location: str, event_location: str) -> bool:
    """True when gym city/region shares meaningful tokens with event location."""
    g_toks = _location_tokens(gym_location)
    e_toks = _location_tokens(event_location)
    if not g_toks or not e_toks:
        return False
    # Prefer tokens length >= 4 (cities / full states); fall back to short cities.
    g_long = {t for t in g_toks if len(t) >= 4}
    e_long = {t for t in e_toks if len(t) >= 4}
    if g_long and e_long:
        return bool(g_long & e_long)
    return bool(g_toks & e_toks)


def gym_narrative_line(fighter_name: str, profile: dict[str, str], *, local: bool = False) -> str:
    """Human line for Ollama / briefs, e.g. 'Trains at New Wave — strong grappling'."""
    gym = profile.get("gym") or ""
    if not gym:
        return ""
    strengths = profile.get("strengths") or ""
    primary = strengths.split(",")[0].strip() if strengths else ""
    bits = [f"Trains at {gym}"]
    if primary:
        bits.append(f"strong {primary} advantage")
    elif strengths:
        bits.append(f"strengths: {strengths}")
    if local:
        bits.append("local / proximity advantage")
    notes = profile.get("notes") or ""
    if notes and len(notes) < 80:
        bits.append(notes)
    return " - ".join(bits)


def gym_matchup_summary(
    f1: str,
    f2: str,
    *,
    event_location: str = "",
    profiles: pd.DataFrame | None = None,
) -> str:
    """Short matchup gym read for dashboard Brief / Ollama."""
    profiles = profiles if profiles is not None else load_gym_profiles()
    p1 = lookup_gym(f1, profiles)
    p2 = lookup_gym(f2, profiles)
    parts: list[str] = []
    loc1 = _location_overlap(p1.get("location", ""), event_location)
    loc2 = _location_overlap(p2.get("location", ""), event_location)
    n1 = gym_narrative_line(f1, p1, local=loc1)
    n2 = gym_narrative_line(f2, p2, local=loc2)
    if n1:
        parts.append(f"{f1}: {n1}")
    if n2:
        parts.append(f"{f2}: {n2}")
    if p1.get("gym") and p1.get("gym") == p2.get("gym"):
        parts.append(f"Shared gym ({p1['gym']}) - camp familiarity risk")
    return " | ".join(parts)


def _profile_lookup_maps(profiles: pd.DataFrame) -> dict[str, dict[str, str]]:
    """fighter_key -> gym fields for O(1) merges."""
    if profiles is None or profiles.empty:
        return {}
    maps: dict[str, dict[str, str]] = {}
    for _, row in profiles.iterrows():
        key = str(row.get("fighter_key") or "")
        if not key:
            continue
        maps[key] = {
            "gym": str(row.get("gym") or "").strip(),
            "location": str(row.get("location") or "").strip(),
            "strengths": str(row.get("strengths") or "").strip(),
            "notes": str(row.get("notes") or "").strip(),
        }
    return maps


def attach_gym_features(df: pd.DataFrame, *, event_location_col: str | None = None) -> pd.DataFrame:
    """
    Attach f1_/f2_ gym columns and local_advantage flags to a fight/prediction frame.

    Safe no-op when gyms.csv is missing or fighters unknown.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    profiles = load_gym_profiles()
    f1_col = "fighter_1" if "fighter_1" in out.columns else ("fighter1" if "fighter1" in out.columns else None)
    f2_col = "fighter_2" if "fighter_2" in out.columns else ("fighter2" if "fighter2" in out.columns else None)
    if not f1_col or not f2_col:
        return out

    loc_col = event_location_col
    if loc_col is None:
        for cand in ("location", "event_location", "venue", "event_city"):
            if cand in out.columns:
                loc_col = cand
                break

    lookup = _profile_lookup_maps(profiles)
    empty = {"gym": "", "location": "", "strengths": "", "notes": ""}

    def _resolve(name: Any) -> dict[str, str]:
        key = _clean_name(name)
        if not key:
            return empty
        hit = lookup.get(key)
        if hit:
            return hit
        # Fuzzy fallback only when exact miss (rare)
        return lookup_gym(str(name or ""), profiles)

    f1_keys = out[f1_col].map(_clean_name)
    f2_keys = out[f2_col].map(_clean_name)
    p1 = f1_keys.map(lambda k: lookup.get(k) or empty)
    p2 = f2_keys.map(lambda k: lookup.get(k) or empty)

    # Fill exact misses with fuzzy lookup (small set)
    miss1 = [i for i, k in enumerate(f1_keys) if k and k not in lookup]
    miss2 = [i for i, k in enumerate(f2_keys) if k and k not in lookup]
    if miss1 or miss2:
        p1_list = list(p1)
        p2_list = list(p2)
        for i in miss1:
            p1_list[i] = _resolve(out.iloc[i][f1_col])
        for i in miss2:
            p2_list[i] = _resolve(out.iloc[i][f2_col])
        p1 = pd.Series(p1_list, index=out.index)
        p2 = pd.Series(p2_list, index=out.index)

    out["f1_gym"] = p1.map(lambda d: d.get("gym", ""))
    out["f1_gym_location"] = p1.map(lambda d: d.get("location", ""))
    out["f1_gym_strengths"] = p1.map(lambda d: d.get("strengths", ""))
    out["f1_gym_notes"] = p1.map(lambda d: d.get("notes", ""))
    out["f2_gym"] = p2.map(lambda d: d.get("gym", ""))
    out["f2_gym_location"] = p2.map(lambda d: d.get("location", ""))
    out["f2_gym_strengths"] = p2.map(lambda d: d.get("strengths", ""))
    out["f2_gym_notes"] = p2.map(lambda d: d.get("notes", ""))

    event_locs = (
        out[loc_col].astype(str)
        if loc_col and loc_col in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    out["f1_local_advantage"] = [
        int(_location_overlap(loc, ev))
        for loc, ev in zip(out["f1_gym_location"], event_locs)
    ]
    out["f2_local_advantage"] = [
        int(_location_overlap(loc, ev))
        for loc, ev in zip(out["f2_gym_location"], event_locs)
    ]

    notes: list[str] = []
    for i in range(len(out)):
        r = out.iloc[i]
        f1n = str(r.get(f1_col) or "")
        f2n = str(r.get(f2_col) or "")
        d1 = {
            "gym": str(r.get("f1_gym") or ""),
            "location": str(r.get("f1_gym_location") or ""),
            "strengths": str(r.get("f1_gym_strengths") or ""),
            "notes": str(r.get("f1_gym_notes") or ""),
        }
        d2 = {
            "gym": str(r.get("f2_gym") or ""),
            "location": str(r.get("f2_gym_location") or ""),
            "strengths": str(r.get("f2_gym_strengths") or ""),
            "notes": str(r.get("f2_gym_notes") or ""),
        }
        parts: list[str] = []
        n1 = gym_narrative_line(f1n, d1, local=bool(r.get("f1_local_advantage")))
        n2 = gym_narrative_line(f2n, d2, local=bool(r.get("f2_local_advantage")))
        if n1:
            parts.append(f"{f1n}: {n1}")
        if n2:
            parts.append(f"{f2n}: {n2}")
        if d1.get("gym") and d1.get("gym") == d2.get("gym"):
            parts.append(f"Shared gym ({d1['gym']}) - camp familiarity risk")
        notes.append(" | ".join(parts))
    out["gym_matchup_note"] = notes

    # Numeric diffs usable as optional model features later
    out["local_advantage_diff"] = (
        pd.to_numeric(out["f1_local_advantage"], errors="coerce").fillna(0).astype(int)
        - pd.to_numeric(out["f2_local_advantage"], errors="coerce").fillna(0).astype(int)
    )
    return out


def format_gym_cell(row: pd.Series | dict[str, Any]) -> str:
    """Compact table cell: 'SBG (striking) vs ITT (cardio)'."""
    if isinstance(row, dict):
        row = pd.Series(row)
    g1 = str(row.get("f1_gym") or "").strip()
    g2 = str(row.get("f2_gym") or "").strip()
    s1 = str(row.get("f1_gym_strengths") or "").split(",")[0].strip()
    s2 = str(row.get("f2_gym_strengths") or "").split(",")[0].strip()
    loc1 = bool(row.get("f1_local_advantage"))
    loc2 = bool(row.get("f2_local_advantage"))

    def _side(gym: str, strength: str, local: bool) -> str:
        if not gym:
            return "-"
        bits = [gym]
        if strength:
            bits.append(strength)
        if local:
            bits.append("local")
        return " / ".join(bits)

    left = _side(g1, s1, loc1)
    right = _side(g2, s2, loc2)
    if left == "-" and right == "-":
        return "-"
    return f"{left} vs {right}"
