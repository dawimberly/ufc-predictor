"""Wikipedia fighter bios as a thin-data supplement/fallback.

Caches to data/cache/wikipedia_fighters.csv.
Uses the MediaWiki API (fail-soft). Prefer Greco/UFCStats when present.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

import config
from src.data_loader import _fighters_same_person, clean_fighter_name, ensure_data_dirs

logger = logging.getLogger(__name__)

WIKI_CACHE = config.CACHE_DIR / "wikipedia_fighters.csv"
WIKI_API = "https://en.wikipedia.org/w/api.php"

_COLS = [
    "name",
    "title",
    "nickname",
    "height_in",
    "reach_in",
    "stance",
    "birth_date",
    "nationality",
    "team",
    "weight_class",
    "source",
    "fetched_at",
]

_USER_AGENT = (
    "UFC-Predictor/1.0 (research bot; local offline-first; contact: local-dev)"
)


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": _USER_AGENT})
    return sess


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def load_wikipedia_fighters() -> pd.DataFrame:
    ensure_data_dirs()
    if not WIKI_CACHE.is_file():
        return _empty()
    try:
        df = pd.read_csv(WIKI_CACHE)
        for col in _COLS:
            if col not in df.columns:
                df[col] = np.nan
        if "name" in df.columns:
            df["name"] = df["name"].map(lambda x: clean_fighter_name(str(x)) if pd.notna(x) else "")
        return df
    except Exception as exc:
        logger.warning("Wikipedia cache unreadable: %s", exc)
        return _empty()


def _save(df: pd.DataFrame) -> None:
    ensure_data_dirs()
    out = df.copy()
    for col in _COLS:
        if col not in out.columns:
            out[col] = np.nan
    out[_COLS].drop_duplicates(subset=["name"], keep="last").to_csv(WIKI_CACHE, index=False)
    _WIKI_INDEX_CACHE["mtime"] = None
    _WIKI_INDEX_CACHE["by_key"] = {}
    _WIKI_INDEX_CACHE["rows"] = []


def _parse_height_inches(text: str) -> float:
    if not text:
        return np.nan
    # {{convert|5|ft|10|in|...}}
    m = re.search(
        r"\{\{convert\|(\d+)\|ft\|(\d+)\|in",
        text,
        re.I,
    )
    if m:
        return float(m.group(1)) * 12.0 + float(m.group(2))
    m = re.search(r"(\d+)\s*ft\.?\s*(\d+)\s*in", text, re.I)
    if m:
        return float(m.group(1)) * 12.0 + float(m.group(2))
    m = re.search(r"(\d+)\s*['′]\s*(\d+)", text)
    if m:
        return float(m.group(1)) * 12.0 + float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", text, re.I)
    if m:
        return float(m.group(1)) / 2.54
    return np.nan


def _parse_reach_inches(text: str) -> float:
    if not text:
        return np.nan
    m = re.search(r"\{\{convert\|(\d+(?:\.\d+)?)\|in", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|\"|″)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", text, re.I)
    if m:
        return float(m.group(1)) / 2.54
    return np.nan


def _parse_birth_date(text: str) -> str:
    if not text:
        return ""
    m = re.search(
        r"\{\{[Bb]irth[_ ]date(?:[_ ]and[_ ]age)?\|(\d{4})\|(\d{1,2})\|(\d{1,2})",
        text,
    )
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    dt = pd.to_datetime(re.sub(r"[\[\]]", "", text), errors="coerce")
    if pd.notna(dt):
        return dt.date().isoformat()
    return ""


def _infobox_field(wikitext: str, *keys: str) -> str:
    for key in keys:
        # Values may contain nested {{convert|...|...}} with pipes — take to end of line.
        pat = rf"\|\s*{re.escape(key)}\s*=\s*([^\n]+)"
        m = re.search(pat, wikitext, re.I)
        if m:
            raw = m.group(1).strip()
            # Trim trailing template closers / wiki noise but keep convert markup for parsers.
            raw = re.sub(r"<[^>]+>", "", raw)
            raw = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", raw)
            return raw.strip()
    return ""


def _extract_infobox(wikitext: str) -> str:
    m = re.search(r"\{\{[Ii]nfobox[\s\S]*?\n\}\}", wikitext)
    if m:
        return m.group(0)
    # Fallback: first 8k of page (infobox usually near top)
    return wikitext[:8000]


def parse_wikipedia_wikitext(wikitext: str, *, title: str = "", query_name: str = "") -> dict[str, Any]:
    box = _extract_infobox(wikitext)
    height = _parse_height_inches(_infobox_field(box, "height", "Height"))
    reach = _parse_reach_inches(_infobox_field(box, "reach", "Reach", "arm_reach"))
    stance_raw = _infobox_field(box, "style", "stance", "fighting_style")
    stance = ""
    low = stance_raw.lower()
    if "southpaw" in low:
        stance = "Southpaw"
    elif "switch" in low:
        stance = "Switch"
    elif "orthodox" in low:
        stance = "Orthodox"
    birth = _parse_birth_date(_infobox_field(box, "birth_date", "born"))
    nickname = _infobox_field(box, "nickname", "other_names")
    nickname = re.sub(r'[\"“”]', "", nickname).strip()
    nationality = _infobox_field(box, "nationality", "citizenship", "country")
    team = _infobox_field(box, "team", "fighting_out_of", "gym", "trainer")
    weight_class = _infobox_field(box, "weight_class", "class", "division")
    name = clean_fighter_name(query_name or title)
    return {
        "name": name,
        "title": title,
        "nickname": nickname,
        "height_in": height,
        "reach_in": reach,
        "stance": stance,
        "birth_date": birth,
        "nationality": nationality,
        "team": team,
        "weight_class": weight_class,
        "source": "wikipedia",
        "fetched_at": _utc_now(),
    }


def _candidate_titles(name: str) -> list[str]:
    """Prefer a small set of high-yield Wikipedia titles to reduce rate limits."""
    from src.fighter_aliases import alias_lookup_names

    titles: list[str] = []
    # Cap aliases — each title is an API hit.
    for alias in alias_lookup_names(name)[:3]:
        for suffix in (" (fighter)", " (mixed martial artist)", ""):
            title = f"{alias}{suffix}" if suffix else alias
            if title not in titles:
                titles.append(title)
        if len(titles) >= 6:
            break
    return titles[:6]


def wikipedia_bio_quality(row: dict[str, Any] | pd.Series) -> dict[str, bool]:
    """Which useful bio fields are present on a Wikipedia cache row."""
    def _ok(key: str) -> bool:
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        if key in ("height_in", "reach_in"):
            try:
                return float(v) > 0
            except (TypeError, ValueError):
                return False
        return bool(str(v).strip()) and str(v).strip().lower() not in {"nan", "none"}

    return {
        "height": _ok("height_in"),
        "reach": _ok("reach_in"),
        "stance": _ok("stance"),
        "birth_date": _ok("birth_date"),
        "nationality": _ok("nationality"),
        "team": _ok("team"),
        "nickname": _ok("nickname"),
    }


def wikipedia_coverage(fighter_names: list[str] | pd.Series) -> dict[str, float]:
    names = [clean_fighter_name(str(n)) for n in fighter_names if clean_fighter_name(str(n))]
    uniq = sorted(set(n for n in names if n))
    if not uniq:
        return {
            "n": 0.0,
            "pct_fighters": 0.0,
            "n_matched": 0.0,
            "pct_with_bio_fields": 0.0,
            "pct_height": 0.0,
            "pct_reach": 0.0,
            "pct_stance": 0.0,
            "pct_team": 0.0,
        }
    matched = 0
    bio_ok = 0
    field_hits = {"height": 0, "reach": 0, "stance": 0, "team": 0, "birth_date": 0, "nationality": 0}
    for n in uniq:
        row = match_wikipedia_row(n)
        if row is None:
            continue
        matched += 1
        q = wikipedia_bio_quality(row)
        if sum(1 for k in ("height", "reach", "stance", "team", "birth_date") if q.get(k)) >= 1:
            bio_ok += 1
        for k in field_hits:
            if q.get(k):
                field_hits[k] += 1
    n = len(uniq)
    return {
        "n": float(n),
        "pct_fighters": matched / n,
        "n_matched": float(matched),
        "pct_with_bio_fields": bio_ok / n,
        "n_with_bio_fields": float(bio_ok),
        "pct_height": field_hits["height"] / n,
        "pct_reach": field_hits["reach"] / n,
        "pct_stance": field_hits["stance"] / n,
        "pct_team": field_hits["team"] / n,
        "pct_birth_date": field_hits["birth_date"] / n,
        "pct_nationality": field_hits["nationality"] / n,
    }


def refresh_wikipedia_for_names(
    names: list[str],
    *,
    max_fetch: int = 40,
    force: bool = False,
    batch_size: int = 5,
    batch_delay_sec: float = 8.0,
    per_fighter_delay_sec: float = 5.0,
) -> dict[str, Any]:
    """
    Best-effort Wikipedia refresh with slow batched pacing.

    Hard-caches successes so we never re-hit pages already in wikipedia_fighters.csv.
    Default pacing: ~5s between fighters, ~8s between batches of 5.
    """
    import time as _time

    stats: dict[str, Any] = {
        "requested": 0,
        "fetched": 0,
        "cached": 0,
        "failed": 0,
        "skipped_hard_cache": 0,
        "failure_reasons": {},
        "failures": [],
        "batch_size": batch_size,
        "batch_delay_sec": batch_delay_sec,
        "per_fighter_delay_sec": per_fighter_delay_sec,
    }
    reasons: dict[str, int] = stats["failure_reasons"]
    sess = _session()
    seen: set[str] = set()
    attempted = 0
    batch_count = 0

    for raw in names:
        clean = clean_fighter_name(str(raw))
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        stats["requested"] += 1

        # Hard cache: never re-hit successful pages unless force=True.
        existing = match_wikipedia_row(clean)
        if existing and not force:
            q = wikipedia_bio_quality(existing)
            if any(q.values()):
                stats["cached"] += 1
                stats["skipped_hard_cache"] += 1
                continue

        if attempted >= max_fetch:
            break

        # Inter-fighter delay (skip before first attempt)
        if attempted > 0:
            _time.sleep(max(0.0, per_fighter_delay_sec))
            if batch_count > 0 and batch_count % batch_size == 0:
                logger.info(
                    "Wikipedia batch pause %.1fs after %s fetches…",
                    batch_delay_sec,
                    batch_count,
                )
                _time.sleep(max(0.0, batch_delay_sec))

        attempted += 1
        batch_count += 1
        try:
            profile, reason = fetch_wikipedia_fighter_detailed(clean, force=force, session=sess)
            if profile:
                stats["fetched"] += 1
            else:
                stats["failed"] += 1
                key = reason or "unknown"
                reasons[key] = int(reasons.get(key, 0)) + 1
                stats["failures"].append({"name": clean, "reason": key})
                logger.info("Wikipedia miss [%s]: %s", key, clean)
                # Extra backoff when rate-limited
                if key in {"rate_limited", "blocked"}:
                    _time.sleep(batch_delay_sec)
        except Exception as exc:
            stats["failed"] += 1
            key = "timeout" if "timeout" in str(exc).lower() else "exception"
            reasons[key] = int(reasons.get(key, 0)) + 1
            stats["failures"].append({"name": clean, "reason": key, "detail": str(exc)[:120]})
            logger.info("Wikipedia refresh error for %s: %s", clean, exc)
    return stats


def fetch_wikipedia_fighter(
    name: str,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    profile, _reason = fetch_wikipedia_fighter_detailed(name, force=force, session=session)
    return profile


def fetch_wikipedia_fighter_detailed(
    name: str,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Fetch Wikipedia bio; return (profile|None, failure_reason)."""
    import time as _time

    from src.fighter_aliases import names_match_aliased

    clean = clean_fighter_name(name)
    if not clean:
        return None, "empty_name"

    # Hard cache: never re-hit successful pages unless force=True.
    cached = load_wikipedia_fighters()
    if not force and not cached.empty:
        for _, row in cached.iterrows():
            if names_match_aliased(clean, str(row.get("name") or "")):
                q = wikipedia_bio_quality(row)
                if any(q.values()):
                    return row.to_dict(), ""

    sess = session or _session()
    saw_page = False
    saw_empty_infobox = False
    saw_non_mma = False
    last_err = ""
    for title in _candidate_titles(clean):
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": title,
                "redirects": 1,
            }
            # Be polite — MediaWiki rate limits burst traffic.
            _time.sleep(max(1.0, float(getattr(config, "REQUEST_DELAY_SEC", 0.5))))
            resp = sess.get(WIKI_API, params=params, timeout=config.REQUEST_TIMEOUT_SEC)
            if resp.status_code in (429, 503):
                _time.sleep(8.0)
                resp = sess.get(WIKI_API, params=params, timeout=config.REQUEST_TIMEOUT_SEC)
            if resp.status_code == 403:
                return None, "blocked"
            if resp.status_code == 429:
                return None, "rate_limited"
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return None, "timeout"
        except Exception as exc:
            last_err = str(exc)[:80]
            logger.info("Wikipedia fetch failed for %s: %s", title, exc)
            continue

        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            if page.get("missing") is not None:
                continue
            saw_page = True
            revs = page.get("revisions") or []
            if not revs:
                continue
            slots = revs[0].get("slots") or {}
            wikitext = (slots.get("main") or {}).get("*") or revs[0].get("*") or ""
            if not wikitext:
                continue
            low = wikitext.lower()
            if "mixed martial" not in low and "ufc" not in low and "infobox martial artist" not in low:
                if title == clean and "fighter" not in (page.get("title") or "").lower():
                    saw_non_mma = True
                    continue
            profile = parse_wikipedia_wikitext(
                wikitext,
                title=str(page.get("title") or title),
                query_name=clean,
            )
            useful = any(
                pd.notna(profile.get(k)) and str(profile.get(k) or "").strip()
                for k in ("height_in", "reach_in", "stance", "birth_date", "nationality", "team")
            )
            if not useful:
                saw_empty_infobox = True
                continue
            cached = load_wikipedia_fighters()
            cached = pd.concat([cached, pd.DataFrame([profile])], ignore_index=True)
            _save(cached)
            return profile, ""
    if saw_empty_infobox:
        return None, "empty_infobox"
    if saw_non_mma:
        return None, "non_mma_page"
    if saw_page:
        return None, "name_mismatch"
    if last_err:
        low = last_err.lower()
        if "403" in low:
            return None, "blocked"
        if "429" in low:
            return None, "rate_limited"
        return None, "request_error"
    return None, "page_not_found"


_WIKI_INDEX_CACHE: dict[str, Any] = {"mtime": None, "by_key": {}, "rows": []}


def _wikipedia_alias_index() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build / reuse alias-key index over the Wikipedia fighter cache."""
    from src.fighter_aliases import alias_lookup_names, normalize_alias_key

    path = WIKI_CACHE
    mtime = path.stat().st_mtime if path.exists() else None
    if _WIKI_INDEX_CACHE["mtime"] == mtime and _WIKI_INDEX_CACHE["by_key"]:
        return _WIKI_INDEX_CACHE["by_key"], _WIKI_INDEX_CACHE["rows"]

    df = load_wikipedia_fighters()
    by_key: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    if not df.empty:
        for _, row in df.iterrows():
            rowd = row.to_dict()
            rows.append(rowd)
            n = clean_fighter_name(str(rowd.get("name") or ""))
            if not n:
                continue
            for alias in alias_lookup_names(n):
                by_key.setdefault(normalize_alias_key(alias), rowd)
    _WIKI_INDEX_CACHE["mtime"] = mtime
    _WIKI_INDEX_CACHE["by_key"] = by_key
    _WIKI_INDEX_CACHE["rows"] = rows
    return by_key, rows


def match_wikipedia_row(name: str) -> dict[str, Any] | None:
    """Match against Wikipedia cache via alias keys (O(aliases), not O(cache))."""
    from src.fighter_aliases import alias_lookup_names, normalize_alias_key

    clean = clean_fighter_name(name)
    if not clean:
        return None
    by_key, rows = _wikipedia_alias_index()
    if not rows:
        return None

    for alias in alias_lookup_names(clean):
        hit = by_key.get(normalize_alias_key(alias))
        if hit is not None:
            return hit

    for rowd in rows:
        if _fighters_same_person(clean, str(rowd.get("name") or "")):
            return rowd
    return None


def fill_history_from_wikipedia(history: pd.DataFrame) -> pd.DataFrame:
    """Fill missing physical / stance fields from Wikipedia cache only (no network)."""
    if history.empty or "fighter" not in history.columns:
        return history
    wiki = load_wikipedia_fighters()
    if wiki.empty:
        return history

    by_name: dict[str, dict[str, Any]] = {}
    for _, row in wiki.iterrows():
        key = clean_fighter_name(str(row.get("name") or "")).lower()
        if key:
            by_name[key] = row.to_dict()

    out = history.copy()
    filled = 0
    resolved: dict[str, dict[str, Any] | None] = {}
    for idx in out.index:
        raw = str(out.at[idx, "fighter"] or "")
        key = clean_fighter_name(raw).lower()
        if not key:
            continue
        if key not in resolved:
            prof = by_name.get(key)
            if prof is None:
                for nk, nrow in by_name.items():
                    if _fighters_same_person(key, nk):
                        prof = nrow
                        break
            resolved[key] = prof
        prof = resolved[key]
        if not prof:
            continue
        if "height_in" in out.columns and pd.isna(out.at[idx, "height_in"]):
            v = prof.get("height_in")
            if v is not None and pd.notna(v) and float(v) > 0:
                out.at[idx, "height_in"] = float(v)
                filled += 1
        if "reach_in" in out.columns and pd.isna(out.at[idx, "reach_in"]):
            v = prof.get("reach_in")
            if v is not None and pd.notna(v) and float(v) > 0:
                out.at[idx, "reach_in"] = float(v)
                filled += 1
        if "stance" in out.columns:
            cur = str(out.at[idx, "stance"] or "").strip()
            wiki_stance = str(prof.get("stance") or "").strip()
            if (not cur or cur.lower() in {"nan", "none"}) and wiki_stance:
                out.at[idx, "stance"] = wiki_stance
                filled += 1
        if "age" in out.columns and pd.isna(out.at[idx, "age"]) and prof.get("birth_date"):
            dob = pd.to_datetime(prof.get("birth_date"), errors="coerce")
            fight_dt = pd.to_datetime(out.at[idx, config.DATE_COLUMN] if config.DATE_COLUMN in out.columns else None, errors="coerce")
            if pd.notna(dob) and pd.notna(fight_dt):
                out.at[idx, "age"] = float((fight_dt - dob).days) / 365.25
                filled += 1
        for col, key_field in (
            ("nickname", "nickname"),
            ("nationality", "nationality"),
            ("team", "team"),
        ):
            if col in out.columns and (pd.isna(out.at[idx, col]) or str(out.at[idx, col]).strip() == ""):
                val = str(prof.get(key_field) or "").strip()
                if val:
                    out.at[idx, col] = val
                    filled += 1
    if filled:
        logger.info("Wikipedia fill: %s cells", filled)
    return out
