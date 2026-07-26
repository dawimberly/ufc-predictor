"""Prior-sport background tiers (wrestling, BJJ, boxing, Muay Thai, etc.).

Leakage-safe: career background is treated as a pre-fight known attribute
(same class as height/stance). Unknown → tier 0 / primary=other.

Sources (fail-soft): Wikipedia bio, Sherdog bio, gym strengths/notes,
optional curated ``data/cache/prior_sport_profiles.csv``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

import config
from src.data_loader import _fighters_same_person, clean_fighter_name, ensure_data_dirs

logger = logging.getLogger(__name__)

PRIOR_SPORT_CACHE = config.CACHE_DIR / "prior_sport_profiles.csv"

BASE_SPORTS = (
    "wrestling",
    "bjj",
    "boxing",
    "muay_thai",
    "kickboxing",
    "sambo",
    "judo",
    "other",
)

# Grappling vs striking families for matchup clash.
GRAPPLING_BASES = frozenset({"wrestling", "bjj", "sambo", "judo"})
STRIKING_BASES = frozenset({"boxing", "muay_thai", "kickboxing"})

# Numeric tiers in [0, 1]. 0 = unknown / none.
WRESTLING_TIERS: dict[str, float] = {
    "unknown": 0.0,
    "hs": 0.20,
    "ncaa_d3_d2": 0.40,
    "ncaa_d1": 0.60,
    "ncaa_aa": 0.80,
    "international_olympic": 1.00,
}

BJJ_TITLE_TIERS: dict[str, float] = {
    "unknown": 0.0,
    "none": 0.25,  # belt / training base, no major title
    "state": 0.45,
    "national": 0.70,
    "world": 1.00,
}

BOXING_TIERS: dict[str, float] = {
    "unknown": 0.0,
    "amateur": 0.25,
    "pro_club": 0.50,
    "contender": 0.75,
    "world_level": 1.00,
}

MT_KB_TIERS: dict[str, float] = {
    "unknown": 0.0,
    "regional": 0.33,
    "major_org": 0.66,
    "world_class": 1.00,
}

OTHER_TIERS: dict[str, float] = {
    "unknown": 0.0,
    "amateur": 0.33,
    "national": 0.66,
    "international": 1.00,
}

# Serious background threshold for multi_base (exclude pure "mentioned lightly").
SERIOUS_TIER = 0.40

_CACHE_COLS = [
    "name",
    "primary_base",
    "base_level_tier",
    "base_level_label",
    "multi_base",
    "wrestling_tier",
    "bjj_tier",
    "boxing_tier",
    "muay_thai_tier",
    "kickboxing_tier",
    "sambo_tier",
    "judo_tier",
    "other_tier",
    "sources",
    "notes",
]


@dataclass
class SportBackground:
    """Parsed prior-sport profile for one fighter."""

    name: str = ""
    tiers: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        for sport in BASE_SPORTS:
            self.tiers.setdefault(sport, 0.0)
            self.labels.setdefault(sport, "unknown")

    @property
    def primary_base(self) -> str:
        ranked = sorted(
            ((s, float(self.tiers.get(s, 0.0))) for s in BASE_SPORTS if s != "other"),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if ranked and ranked[0][1] >= 0.15:
            return ranked[0][0]
        if float(self.tiers.get("other", 0.0)) >= 0.15:
            return "other"
        return "other"

    @property
    def base_level_tier(self) -> float:
        primary = self.primary_base
        return float(self.tiers.get(primary, 0.0))

    @property
    def base_level_label(self) -> str:
        return str(self.labels.get(self.primary_base, "unknown"))

    @property
    def multi_base(self) -> float:
        serious = [s for s in BASE_SPORTS if s != "other" and float(self.tiers.get(s, 0.0)) >= SERIOUS_TIER]
        return 1.0 if len(serious) >= 2 else 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "name": clean_fighter_name(self.name),
            "primary_base": self.primary_base,
            "base_level_tier": self.base_level_tier,
            "base_level_label": self.base_level_label,
            "multi_base": self.multi_base,
            "wrestling_tier": self.tiers.get("wrestling", 0.0),
            "bjj_tier": self.tiers.get("bjj", 0.0),
            "boxing_tier": self.tiers.get("boxing", 0.0),
            "muay_thai_tier": self.tiers.get("muay_thai", 0.0),
            "kickboxing_tier": self.tiers.get("kickboxing", 0.0),
            "sambo_tier": self.tiers.get("sambo", 0.0),
            "judo_tier": self.tiers.get("judo", 0.0),
            "other_tier": self.tiers.get("other", 0.0),
            "sources": "|".join(self.sources),
            "notes": self.notes[:500],
        }

    def feature_dict(self) -> dict[str, float]:
        primary = self.primary_base
        out: dict[str, float] = {
            "base_level_tier": self.base_level_tier,
            "multi_base": self.multi_base,
            "base_grappling": 1.0 if primary in GRAPPLING_BASES else 0.0,
            "base_striking": 1.0 if primary in STRIKING_BASES else 0.0,
        }
        for sport in BASE_SPORTS:
            out[f"base_{sport}"] = 1.0 if primary == sport else 0.0
            out[f"{sport}_tier"] = float(self.tiers.get(sport, 0.0))
        return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _bump(bg: SportBackground, sport: str, tier: float, label: str, *, source: str = "") -> None:
    cur = float(bg.tiers.get(sport, 0.0))
    if tier > cur:
        bg.tiers[sport] = float(tier)
        bg.labels[sport] = label
    if source and source not in bg.sources:
        bg.sources.append(source)


def _infer_wrestling(text: str, bg: SportBackground, *, source: str) -> None:
    t = _norm(text)
    if not re.search(r"wrestl|folkstyle|freestyle|greco[- ]roman", t):
        return
    if re.search(
        r"olympi(c|an)|olympic (gold|silver|bronze|team)|world (freestyle|greco).{0,20}(champion|medalist)|senior world",
        t,
    ):
        _bump(bg, "wrestling", WRESTLING_TIERS["international_olympic"], "international_olympic", source=source)
        return
    if re.search(r"all[- ]american|ncaa all american", t):
        _bump(bg, "wrestling", WRESTLING_TIERS["ncaa_aa"], "ncaa_aa", source=source)
        return
    if re.search(r"ncaa.{0,12}(division\s*i|d-?1)\b|division\s*i\b.{0,20}wrestl|d1 wrestl", t):
        _bump(bg, "wrestling", WRESTLING_TIERS["ncaa_d1"], "ncaa_d1", source=source)
        return
    if re.search(r"ncaa.{0,12}(division\s*(ii|iii)|d-?[23])\b|naia wrestl|division\s*(ii|iii).{0,20}wrestl", t):
        _bump(bg, "wrestling", WRESTLING_TIERS["ncaa_d3_d2"], "ncaa_d3_d2", source=source)
        return
    if re.search(r"high school|hs (state )?champion|state champion.{0,20}wrestl|scholastic wrestl", t):
        _bump(bg, "wrestling", WRESTLING_TIERS["hs"], "hs", source=source)
        return
    # Generic wrestling mention without level
    if re.search(r"wrestl(ing|er)|wrestling base|folkstyle", t):
        _bump(bg, "wrestling", max(0.15, bg.tiers.get("wrestling", 0.0)), "unknown", source=source)


def _infer_bjj(text: str, bg: SportBackground, *, source: str) -> None:
    t = _norm(text)
    if not re.search(r"\bbjj\b|brazilian jiu[- ]jitsu|jiu[- ]jitsu|grappling (base|background)", t):
        # Still allow pure title cues with BJJ orgs
        if not re.search(r"ibjjf|adcc", t):
            return
    if re.search(r"ibjjf world|world (bjj|jiu[- ]jitsu) champion|adcc (champion|gold|world)|world champion.{0,30}(bjj|jiu)", t):
        _bump(bg, "bjj", BJJ_TITLE_TIERS["world"], "world", source=source)
        return
    if re.search(r"pan[- ]am|national champion.{0,20}(bjj|jiu)|ibjjf national|brazilian national", t):
        _bump(bg, "bjj", BJJ_TITLE_TIERS["national"], "national", source=source)
        return
    if re.search(r"state champion.{0,20}(bjj|jiu)|state (bjj|jiu)", t):
        _bump(bg, "bjj", BJJ_TITLE_TIERS["state"], "state", source=source)
        return
    if re.search(r"black belt|coral belt|red belt", t):
        _bump(bg, "bjj", BJJ_TITLE_TIERS["none"], "none", source=source)
        return
    if re.search(r"brown belt|purple belt", t):
        _bump(bg, "bjj", 0.20, "none", source=source)
        return
    if re.search(r"\bbjj\b|jiu[- ]jitsu", t):
        _bump(bg, "bjj", max(0.15, bg.tiers.get("bjj", 0.0)), "unknown", source=source)


def _infer_boxing(text: str, bg: SportBackground, *, source: str) -> None:
    t = _norm(text)
    if not re.search(r"box(ing|er)|pugilist", t):
        return
    if re.search(
        r"\b(wba|wbc|ibf|wbo)\b|world (boxing )?titl|undisputed|olympic boxing.{0,20}(gold|medalist)|world champion.{0,20}box",
        t,
    ):
        _bump(bg, "boxing", BOXING_TIERS["world_level"], "world_level", source=source)
        return
    if re.search(r"title (eliminator|contender)|ranked contender|top[- ]10|contender level|prospect of the year", t):
        _bump(bg, "boxing", BOXING_TIERS["contender"], "contender", source=source)
        return
    if re.search(r"pro box|professional box|boxing record|fought (on|under).{0,20}(boxrec|pro)", t):
        _bump(bg, "boxing", BOXING_TIERS["pro_club"], "pro_club", source=source)
        return
    if re.search(r"amateur box|golden gloves|national amateur|usa boxing", t):
        _bump(bg, "boxing", BOXING_TIERS["amateur"], "amateur", source=source)
        return
    if re.search(r"boxing base|boxer", t):
        _bump(bg, "boxing", max(0.15, bg.tiers.get("boxing", 0.0)), "unknown", source=source)


def _infer_muay_thai_kickboxing(text: str, bg: SportBackground, *, source: str) -> None:
    t = _norm(text)
    has_mt = bool(re.search(r"muay thai|thaiboxing|thai boxing", t))
    has_kb = bool(re.search(r"kickbox|k-1|glory|one championship.{0,30}kick|kick boxing", t))
    if not (has_mt or has_kb):
        return

    world = bool(
        re.search(
            r"lumpinee|rajadamnern|stadium champion|glory (world|champion)|k-1 (world|champion)|"
            r"one .{0,20}kickboxing champion|world (muay thai|kickboxing) champion",
            t,
        )
    )
    major = bool(
        re.search(
            r"bellator kickboxing|kunlun|wbc muay thai|ifma|stadium|max muay thai|"
            r"lion fight|combat sports federation",
            t,
        )
    )
    if has_mt:
        if world:
            _bump(bg, "muay_thai", MT_KB_TIERS["world_class"], "world_class", source=source)
        elif major:
            _bump(bg, "muay_thai", MT_KB_TIERS["major_org"], "major_org", source=source)
        else:
            _bump(bg, "muay_thai", MT_KB_TIERS["regional"], "regional", source=source)
    if has_kb:
        if world:
            _bump(bg, "kickboxing", MT_KB_TIERS["world_class"], "world_class", source=source)
        elif major:
            _bump(bg, "kickboxing", MT_KB_TIERS["major_org"], "major_org", source=source)
        else:
            _bump(bg, "kickboxing", MT_KB_TIERS["regional"], "regional", source=source)


def _infer_sambo_judo_other(text: str, bg: SportBackground, *, source: str) -> None:
    t = _norm(text)
    if re.search(r"\bsambo\b|combat sambo", t):
        if re.search(r"world|olympic|international|european champion", t):
            _bump(bg, "sambo", OTHER_TIERS["international"], "international", source=source)
        elif re.search(r"national", t):
            _bump(bg, "sambo", OTHER_TIERS["national"], "national", source=source)
        else:
            _bump(bg, "sambo", OTHER_TIERS["amateur"], "amateur", source=source)
    if re.search(r"\bjudo\b|judoka", t):
        if re.search(r"olympi(c|an)|world (judo )?champion|ijf", t):
            _bump(bg, "judo", OTHER_TIERS["international"], "international", source=source)
        elif re.search(r"national", t):
            _bump(bg, "judo", OTHER_TIERS["national"], "national", source=source)
        else:
            _bump(bg, "judo", OTHER_TIERS["amateur"], "amateur", source=source)
    # Catch-all athletic bases
    if re.search(r"collegiate athlete|college football|olympic.{0,20}(track|swim|gymnast)", t):
        if re.search(r"olympi(c|an)|international", t):
            _bump(bg, "other", OTHER_TIERS["international"], "international", source=source)
        elif re.search(r"national|ncaa|division", t):
            _bump(bg, "other", OTHER_TIERS["national"], "national", source=source)
        else:
            _bump(bg, "other", OTHER_TIERS["amateur"], "amateur", source=source)


def parse_prior_sport_text(*chunks: str, name: str = "", source: str = "text") -> SportBackground:
    """Infer sport tiers from free-text bio / notes / gym strengths."""
    bg = SportBackground(name=name)
    blob = " | ".join(str(c) for c in chunks if c and str(c).strip())
    if not blob.strip():
        return bg
    bg.notes = blob[:500]
    _infer_wrestling(blob, bg, source=source)
    _infer_bjj(blob, bg, source=source)
    _infer_boxing(blob, bg, source=source)
    _infer_muay_thai_kickboxing(blob, bg, source=source)
    _infer_sambo_judo_other(blob, bg, source=source)
    return bg


def _merge_backgrounds(*parts: SportBackground) -> SportBackground:
    out = SportBackground(name=next((p.name for p in parts if p.name), ""))
    notes: list[str] = []
    for part in parts:
        if not part:
            continue
        for sport in BASE_SPORTS:
            _bump(out, sport, float(part.tiers.get(sport, 0.0)), str(part.labels.get(sport, "unknown")))
        for s in part.sources:
            if s not in out.sources:
                out.sources.append(s)
        if part.notes:
            notes.append(part.notes)
        if part.name and not out.name:
            out.name = part.name
    out.notes = " || ".join(notes)[:500]
    return out


def load_prior_sport_profiles() -> pd.DataFrame:
    ensure_data_dirs()
    if not PRIOR_SPORT_CACHE.is_file():
        return pd.DataFrame(columns=_CACHE_COLS)
    try:
        df = pd.read_csv(PRIOR_SPORT_CACHE)
        for col in _CACHE_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df["name"] = df["name"].map(lambda x: clean_fighter_name(str(x)) if pd.notna(x) else "")
        return df
    except Exception as exc:
        logger.warning("Prior-sport cache unreadable: %s", exc)
        return pd.DataFrame(columns=_CACHE_COLS)


def save_prior_sport_profiles(df: pd.DataFrame) -> None:
    ensure_data_dirs()
    out = df.copy()
    for col in _CACHE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    out[_CACHE_COLS].drop_duplicates(subset=["name"], keep="last").to_csv(
        PRIOR_SPORT_CACHE, index=False
    )


def _profile_from_cache_row(row: pd.Series | dict[str, Any]) -> SportBackground:
    bg = SportBackground(name=str(row.get("name") or ""))
    for sport in BASE_SPORTS:
        key = f"{sport}_tier"
        val = row.get(key, 0.0)
        try:
            tier = float(val) if pd.notna(val) else 0.0
        except (TypeError, ValueError):
            tier = 0.0
        bg.tiers[sport] = tier
        bg.labels[sport] = "cached" if tier > 0 else "unknown"
    # Prefer stored primary if present and consistent
    primary = str(row.get("primary_base") or "").strip().lower()
    if primary in BASE_SPORTS and float(bg.tiers.get(primary, 0.0)) < 0.15:
        bg.tiers[primary] = max(float(row.get("base_level_tier") or 0.15), 0.15)
        bg.labels[primary] = str(row.get("base_level_label") or "cached")
    src = str(row.get("sources") or "")
    bg.sources = [s for s in src.split("|") if s]
    bg.notes = str(row.get("notes") or "")
    return bg


def match_prior_sport_row(name: str) -> dict[str, Any] | None:
    clean = clean_fighter_name(name)
    if not clean:
        return None
    df = load_prior_sport_profiles()
    if df.empty:
        return None
    for _, row in df.iterrows():
        if _fighters_same_person(clean, str(row.get("name") or "")):
            return row.to_dict()
    return None


def _collect_source_texts(name: str) -> list[tuple[str, str]]:
    """Gather (source, text) blobs from Wiki / Sherdog / gyms (fail-soft)."""
    chunks: list[tuple[str, str]] = []
    clean = clean_fighter_name(name)

    try:
        from src.wikipedia_fighters import match_wikipedia_row

        wiki = match_wikipedia_row(clean)
        if wiki:
            parts = [
                str(wiki.get("stance") or ""),
                str(wiki.get("team") or ""),
                str(wiki.get("nickname") or ""),
                str(wiki.get("nationality") or ""),
                str(wiki.get("weight_class") or ""),
            ]
            # Wikipedia pages often leave style in stance/team; also use title
            parts.append(str(wiki.get("title") or ""))
            blob = " | ".join(p for p in parts if p and p.strip())
            if blob.strip():
                chunks.append(("wikipedia", blob))
    except Exception as exc:
        logger.debug("Wiki prior-sport lookup failed: %s", exc)

    try:
        from src.sherdog import match_sherdog_row

        sh = match_sherdog_row(clean)
        if sh:
            parts = [
                str(sh.get("team") or ""),
                str(sh.get("nickname") or ""),
                str(sh.get("nationality") or ""),
                str(sh.get("weight_class") or ""),
            ]
            blob = " | ".join(p for p in parts if p and p.strip())
            if blob.strip():
                chunks.append(("sherdog", blob))
    except Exception as exc:
        logger.debug("Sherdog prior-sport lookup failed: %s", exc)

    try:
        from src.gym_data import load_gym_profiles

        gyms = load_gym_profiles()
        if not gyms.empty:
            for _, row in gyms.iterrows():
                if _fighters_same_person(clean, str(row.get("fighter_name") or row.get("fighter_key") or "")):
                    parts = [
                        str(row.get("strengths") or ""),
                        str(row.get("notes") or ""),
                        str(row.get("gym") or ""),
                    ]
                    blob = " | ".join(p for p in parts if p and p.strip())
                    if blob.strip():
                        chunks.append(("gyms", blob))
                    break
    except Exception as exc:
        logger.debug("Gym prior-sport lookup failed: %s", exc)

    return chunks


def resolve_prior_sport(name: str, *, persist: bool = True) -> SportBackground:
    """Resolve a fighter's prior-sport profile from cache + available bios."""
    clean = clean_fighter_name(name)
    bg = SportBackground(name=clean)
    if not clean:
        return bg

    cached = match_prior_sport_row(clean)
    parts: list[SportBackground] = []
    if cached:
        parts.append(_profile_from_cache_row(cached))

    for source, text in _collect_source_texts(clean):
        parts.append(parse_prior_sport_text(text, name=clean, source=source))

    if parts:
        bg = _merge_backgrounds(*parts)
        bg.name = clean
    else:
        bg.name = clean

    if persist and (bg.base_level_tier > 0 or bg.primary_base != "other" or any(v > 0 for v in bg.tiers.values())):
        try:
            df = load_prior_sport_profiles()
            df = pd.concat([df, pd.DataFrame([bg.to_row()])], ignore_index=True)
            save_prior_sport_profiles(df)
        except Exception as exc:
            logger.debug("Could not persist prior-sport profile: %s", exc)
    return bg


@lru_cache(maxsize=1)
def _gym_text_index() -> dict[str, str]:
    try:
        from src.gym_data import load_gym_profiles

        gyms = load_gym_profiles()
    except Exception:
        return {}
    out: dict[str, str] = {}
    if gyms.empty:
        return out
    for _, row in gyms.iterrows():
        key = clean_fighter_name(str(row.get("fighter_name") or "")).lower()
        if not key:
            continue
        blob = " | ".join(
            str(row.get(c) or "") for c in ("strengths", "notes", "gym") if str(row.get(c) or "").strip()
        )
        if blob.strip():
            out[key] = blob
    return out


def build_prior_sport_cache_from_available(*, persist: bool = True) -> pd.DataFrame:
    """Rebuild profiles for all names seen in Wiki/Sherdog/gym caches (no live scrape)."""
    names: set[str] = set()
    try:
        from src.wikipedia_fighters import load_wikipedia_fighters

        w = load_wikipedia_fighters()
        if not w.empty:
            names.update(clean_fighter_name(str(n)) for n in w["name"].dropna())
    except Exception:
        pass
    try:
        from src.sherdog import load_sherdog_fighters

        s = load_sherdog_fighters()
        if not s.empty:
            names.update(clean_fighter_name(str(n)) for n in s["name"].dropna())
    except Exception:
        pass
    try:
        from src.gym_data import load_gym_profiles

        g = load_gym_profiles()
        if not g.empty:
            names.update(clean_fighter_name(str(n)) for n in g["fighter_name"].dropna())
    except Exception:
        pass

    rows = []
    for name in sorted(n for n in names if n):
        bg = resolve_prior_sport(name, persist=False)
        rows.append(bg.to_row())
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_CACHE_COLS)
    if persist and not df.empty:
        # Collapse case-variant duplicates by cleaned lower key (keep highest tier).
        df["_key"] = df["name"].map(lambda n: clean_fighter_name(str(n)).lower())
        df = df.sort_values("base_level_tier", ascending=False).drop_duplicates("_key", keep="first")
        df = df.drop(columns=["_key"])
        save_prior_sport_profiles(df)
        logger.info(
            "Prior-sport cache: %s fighters (%s with tier>0)",
            len(df),
            int((df["base_level_tier"] > 0).sum()),
        )
    return df.reset_index(drop=True)


def fill_history_from_prior_sport(history: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-sport features onto long history (static pre-fight attributes)."""
    if history.empty or "fighter" not in history.columns:
        return history

    out = history.copy()
    feat_cols = [
        "primary_base",
        "base_level_tier",
        "multi_base",
        "base_grappling",
        "base_striking",
        *[f"base_{s}" for s in BASE_SPORTS],
        *[f"{s}_tier" for s in BASE_SPORTS],
    ]
    for col in feat_cols:
        if col not in out.columns:
            out[col] = np.nan if col != "primary_base" else ""

    # Ensure cache has gym-derived profiles at least once per process.
    try:
        cached = load_prior_sport_profiles()
        if cached.empty:
            build_prior_sport_cache_from_available(persist=True)
            cached = load_prior_sport_profiles()
    except Exception as exc:
        logger.warning("Prior-sport cache build skipped: %s", exc)
        cached = load_prior_sport_profiles()

    by_name: dict[str, SportBackground] = {}
    if not cached.empty:
        for _, row in cached.iterrows():
            key = clean_fighter_name(str(row.get("name") or "")).lower()
            if key:
                by_name[key] = _profile_from_cache_row(row)

    # Also parse gym strengths for fighters not yet cached (common path).
    gym_idx = _gym_text_index()

    filled = 0
    resolved: dict[str, SportBackground] = {}
    for idx in out.index:
        raw = str(out.at[idx, "fighter"] or "")
        key = clean_fighter_name(raw).lower()
        if not key:
            continue
        if key not in resolved:
            if key in by_name:
                bg = by_name[key]
            elif key in gym_idx:
                bg = parse_prior_sport_text(gym_idx[key], name=raw, source="gyms")
            else:
                bg = SportBackground(name=raw)
            # Fuzzy cache match
            if bg.base_level_tier <= 0 and by_name:
                for ck, cb in by_name.items():
                    if _fighters_same_person(key, ck):
                        bg = _merge_backgrounds(bg, cb)
                        break
            resolved[key] = bg
        bg = resolved[key]
        feats = bg.feature_dict()
        out.at[idx, "primary_base"] = bg.primary_base
        for col, val in feats.items():
            if col in out.columns:
                if pd.isna(out.at[idx, col]) or (col.startswith("base_") and col != "base_level_tier"):
                    out.at[idx, col] = val
                    filled += 1
                elif col.endswith("_tier") and (pd.isna(out.at[idx, col]) or float(out.at[idx, col] or 0) == 0):
                    out.at[idx, col] = val
                    filled += 1
        if pd.isna(out.at[idx, "base_level_tier"]) or float(out.at[idx, "base_level_tier"] or 0) == 0:
            out.at[idx, "base_level_tier"] = feats["base_level_tier"]
            filled += 1
        if pd.isna(out.at[idx, "multi_base"]):
            out.at[idx, "multi_base"] = feats["multi_base"]
            filled += 1

    if filled:
        logger.info("Prior-sport fill: %s cells (%s fighters resolved)", filled, len(resolved))
    return out


def prior_sport_coverage(fighter_names: list[str] | pd.Series) -> dict[str, float]:
    names = [clean_fighter_name(str(n)) for n in fighter_names if clean_fighter_name(str(n))]
    uniq = sorted(set(names))
    if not uniq:
        return {"n": 0.0, "pct_known": 0.0, "n_known": 0.0}
    df = load_prior_sport_profiles()
    known = 0
    for n in uniq:
        row = match_prior_sport_row(n) if not df.empty else None
        if row and float(row.get("base_level_tier") or 0) > 0:
            known += 1
            continue
        # Gym-only inference counts as known if tier>0
        key = n.lower()
        if key in _gym_text_index():
            bg = parse_prior_sport_text(_gym_text_index()[key], name=n, source="gyms")
            if bg.base_level_tier > 0:
                known += 1
    return {"n": float(len(uniq)), "pct_known": known / len(uniq), "n_known": float(known)}
