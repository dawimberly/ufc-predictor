"""Sherdog fighter profiles + fight history (fail-soft, leakage-safe).

Caches:
  data/cache/sherdog_fighters.csv
  data/cache/sherdog_fights.csv
  data/cache/sherdog_name_index.json

Live scrapes are best-effort; blocked/missing sources leave UFCStats/Greco intact.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from src.data_loader import _fighters_same_person, clean_fighter_name, ensure_data_dirs

logger = logging.getLogger(__name__)

SHERDOG_BASE = "https://www.sherdog.com"
SHERDOG_FIGHTERS_CACHE = config.CACHE_DIR / "sherdog_fighters.csv"
SHERDOG_FIGHTS_CACHE = config.CACHE_DIR / "sherdog_fights.csv"
SHERDOG_INDEX_CACHE = config.CACHE_DIR / "sherdog_name_index.json"

_FIGHTER_COLS = [
    "name",
    "sherdog_id",
    "url",
    "nickname",
    "weight_class",
    "height_in",
    "reach_in",
    "birth_date",
    "nationality",
    "team",
    "wins",
    "losses",
    "draws",
    "source",
    "fetched_at",
]
_FIGHT_COLS = [
    "sherdog_id",
    "fighter",
    "opponent",
    "result",
    "method",
    "event",
    "bout_date",
    "weight_class",
    "source",
]

_USER_AGENT = (
    "Mozilla/5.0 (compatible; UFC-Predictor/1.0; +https://github.com/local/ufc-predictor)"
)


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return sess


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_fighters() -> pd.DataFrame:
    return pd.DataFrame(columns=_FIGHTER_COLS)


def _empty_fights() -> pd.DataFrame:
    return pd.DataFrame(columns=_FIGHT_COLS)


def load_sherdog_fighters() -> pd.DataFrame:
    ensure_data_dirs()
    if not SHERDOG_FIGHTERS_CACHE.is_file():
        return _empty_fighters()
    try:
        df = pd.read_csv(SHERDOG_FIGHTERS_CACHE)
        for col in _FIGHTER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        if "name" in df.columns:
            df["name"] = df["name"].map(lambda x: clean_fighter_name(str(x)) if pd.notna(x) else "")
        return df
    except Exception as exc:
        logger.warning("Sherdog fighters cache unreadable: %s", exc)
        return _empty_fighters()


def load_sherdog_fights() -> pd.DataFrame:
    ensure_data_dirs()
    if not SHERDOG_FIGHTS_CACHE.is_file():
        return _empty_fights()
    try:
        df = pd.read_csv(SHERDOG_FIGHTS_CACHE)
        for col in _FIGHT_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df["bout_date"] = pd.to_datetime(df["bout_date"], errors="coerce")
        if "fighter" in df.columns:
            df["fighter"] = df["fighter"].map(
                lambda x: clean_fighter_name(str(x)) if pd.notna(x) else ""
            )
        return df
    except Exception as exc:
        logger.warning("Sherdog fights cache unreadable: %s", exc)
        return _empty_fights()


def _save_fighters(df: pd.DataFrame) -> None:
    ensure_data_dirs()
    out = df.copy()
    for col in _FIGHTER_COLS:
        if col not in out.columns:
            out[col] = np.nan
    out[_FIGHTER_COLS].drop_duplicates(subset=["sherdog_id"], keep="last").to_csv(
        SHERDOG_FIGHTERS_CACHE, index=False
    )


def _save_fights(df: pd.DataFrame) -> None:
    ensure_data_dirs()
    out = df.copy()
    for col in _FIGHT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    key = ["sherdog_id", "opponent", "bout_date", "result"]
    out[_FIGHT_COLS].drop_duplicates(subset=key, keep="last").to_csv(
        SHERDOG_FIGHTS_CACHE, index=False
    )


def _load_index() -> dict[str, str]:
    if not SHERDOG_INDEX_CACHE.is_file():
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(SHERDOG_INDEX_CACHE.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def _save_index(index: dict[str, str]) -> None:
    ensure_data_dirs()
    SHERDOG_INDEX_CACHE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _parse_height_inches(text: str) -> float:
    if not text:
        return np.nan
    m = re.search(r"(\d+)\s*['′]\s*(\d+)", text)
    if m:
        return float(m.group(1)) * 12.0 + float(m.group(2))
    m = re.search(r"(\d+)\s*ft\.?\s*(\d+)", text, re.I)
    if m:
        return float(m.group(1)) * 12.0 + float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", text, re.I)
    if m:
        return float(m.group(1)) / 2.54
    return np.nan


def _parse_reach_inches(text: str) -> float:
    if not text:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|\"|″)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", text, re.I)
    if m:
        return float(m.group(1)) / 2.54
    return _parse_height_inches(text)


def _parse_record_triplet(text: str) -> tuple[float, float, float]:
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)(?:\s*[-–]\s*(\d+))?", text or "")
    if not m:
        return np.nan, np.nan, np.nan
    return float(m.group(1)), float(m.group(2)), float(m.group(3) or 0)


def _extract_fighter_id(url: str) -> str:
    m = re.search(r"/fighter/[^/]+-(\d+)", url or "")
    return m.group(1) if m else ""


def search_sherdog_fighter(
    name: str,
    *,
    weight_class: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """Best-effort Sherdog search. Returns {name, url, sherdog_id} or None."""
    from src.fighter_aliases import alias_lookup_names

    clean = clean_fighter_name(name)
    if not clean:
        return None
    index = _load_index()
    # Try all alias keys against the local index first.
    for alias in alias_lookup_names(clean):
        key = alias.lower()
        if key in index:
            sid = index[key]
            fighters = load_sherdog_fighters()
            hit = fighters[fighters["sherdog_id"].astype(str) == str(sid)]
            if not hit.empty:
                row = hit.iloc[0].to_dict()
                return {
                    "name": row.get("name") or clean,
                    "url": row.get("url") or f"{SHERDOG_BASE}/fighter/{sid}",
                    "sherdog_id": str(sid),
                }

    sess = session or _session()
    # Try each alias spelling against Sherdog search.
    for alias in alias_lookup_names(clean):
        url = f"{SHERDOG_BASE}/stats/fightfinder?SearchTxt={quote_plus(alias)}"
        try:
            resp = sess.get(url, timeout=config.REQUEST_TIMEOUT_SEC)
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            logger.info("Sherdog search failed for %s: %s", alias, exc)
            continue

        if "just a moment" in html.lower() or "cf-browser-verification" in html.lower():
            logger.info("Sherdog search blocked (bot challenge) for %s", alias)
            return None

        soup = BeautifulSoup(html, "html.parser")
        candidates: list[dict[str, Any]] = []
        alias_clean = clean_fighter_name(alias)
        from src.fighter_aliases import names_match_aliased, normalize_alias_key

        for a in soup.select("a[href*='/fighter/']"):
            href = a.get("href") or ""
            if "/fighter/" not in href:
                continue
            full = urljoin(SHERDOG_BASE, href)
            sid = _extract_fighter_id(full)
            label = clean_fighter_name(a.get_text(" ", strip=True))
            if not sid or not label:
                continue
            if (
                names_match_aliased(clean, label)
                or names_match_aliased(alias_clean, label)
                or normalize_alias_key(clean) == normalize_alias_key(label)
                or normalize_alias_key(alias_clean) == normalize_alias_key(label)
            ):
                candidates.append({"name": label, "url": full, "sherdog_id": sid})

        if not candidates:
            continue

        exact = [
            c
            for c in candidates
            if c["name"].lower() == clean.lower() or c["name"].lower() == alias_clean.lower()
        ]
        chosen = exact[0] if exact else candidates[0]
        if weight_class:
            pass
        index = _load_index()
        index[clean.lower()] = str(chosen["sherdog_id"])
        for alias_name in alias_lookup_names(clean):
            index[alias_name.lower()] = str(chosen["sherdog_id"])
        _save_index(index)
        return chosen

    # Last-resort known IDs when search ranking/HTML misses common aliases.
    from src.fighter_aliases import SHERDOG_ID_OVERRIDES, normalize_alias_key

    sid = SHERDOG_ID_OVERRIDES.get(normalize_alias_key(clean))
    if sid:
        return {
            "name": clean,
            "url": f"{SHERDOG_BASE}/fighter/{sid}",
            "sherdog_id": str(sid),
        }
    return None


def parse_sherdog_fighter_html(html: str, *, url: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse fighter bio + fight history from a Sherdog fighter page."""
    soup = BeautifulSoup(html, "html.parser")
    name_el = soup.select_one("span.fn") or soup.select_one("h1")
    name = clean_fighter_name(name_el.get_text(" ", strip=True) if name_el else "")
    sid = _extract_fighter_id(url) or ""

    nickname = ""
    nick_el = soup.select_one("span.nickname")
    if nick_el:
        nickname = re.sub(r'[\"“”]', "", nick_el.get_text(" ", strip=True)).strip()

    bio: dict[str, str] = {}
    for row in soup.select(".bio-holder tr, .fighter-info tr, table tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        key = cells[0].get_text(" ", strip=True).lower()
        val = cells[1].get_text(" ", strip=True)
        bio[key] = val

    wins = losses = draws = np.nan
    record_el = soup.select_one(".record, .fighter-data .winsloses")
    if record_el:
        wins, losses, draws = _parse_record_triplet(record_el.get_text(" ", strip=True))
    for key, val in bio.items():
        if "record" in key or "wins" in key:
            w, l, d = _parse_record_triplet(val)
            if not np.isnan(w):
                wins, losses, draws = w, l, d

    height = _parse_height_inches(bio.get("height", "") or bio.get("ht", ""))
    reach = _parse_reach_inches(bio.get("reach", "") or bio.get("arm reach", ""))
    birth = bio.get("born", "") or bio.get("birth date", "") or bio.get("date of birth", "")
    birth_date = pd.to_datetime(birth, errors="coerce")
    nationality = bio.get("nationality", "") or bio.get("country", "")
    team = bio.get("association", "") or bio.get("team", "") or bio.get("gym", "")
    weight_class = bio.get("class", "") or bio.get("weight class", "") or bio.get("division", "")

    profile = {
        "name": name,
        "sherdog_id": sid,
        "url": url or (f"{SHERDOG_BASE}/fighter/{sid}" if sid else ""),
        "nickname": nickname,
        "weight_class": weight_class,
        "height_in": height,
        "reach_in": reach,
        "birth_date": birth_date.date().isoformat() if pd.notna(birth_date) else "",
        "nationality": nationality,
        "team": team,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "source": "sherdog",
        "fetched_at": _utc_now(),
    }

    fights: list[dict[str, Any]] = []
    table = soup.select_one("table.fight_history, .module.fight_history table, table.new_fight_history")
    if table is None:
        for t in soup.find_all("table"):
            headers = " ".join(th.get_text(" ", strip=True).lower() for th in t.find_all("th"))
            if "result" in headers and "opponent" in headers:
                table = t
                break
    if table is not None:
        for tr in table.select("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            result = tds[0].get_text(" ", strip=True).upper()
            if result not in {"WIN", "LOSS", "DRAW", "NC", "N/C"}:
                continue
            opp_a = tds[1].find("a")
            opponent = clean_fighter_name(
                opp_a.get_text(" ", strip=True) if opp_a else tds[1].get_text(" ", strip=True)
            )
            event = tds[2].get_text(" ", strip=True) if len(tds) > 2 else ""
            method = tds[3].get_text(" ", strip=True) if len(tds) > 3 else ""
            date_txt = ""
            for td in tds:
                txt = td.get_text(" ", strip=True)
                if re.search(r"\d{4}", txt) and re.search(r"[A-Za-z]", txt):
                    # Prefer dedicated date cell patterns like "Jul / 25 / 2025"
                    if re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", txt, re.I):
                        date_txt = txt
            bout_date = pd.to_datetime(date_txt.replace("/", " "), errors="coerce")
            fights.append(
                {
                    "sherdog_id": sid,
                    "fighter": name,
                    "opponent": opponent,
                    "result": result,
                    "method": method,
                    "event": event,
                    "bout_date": bout_date,
                    "weight_class": weight_class,
                    "source": "sherdog",
                }
            )
    return profile, fights


def fetch_sherdog_fighter(
    name: str,
    *,
    weight_class: str | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    profile, _reason = fetch_sherdog_fighter_detailed(
        name, weight_class=weight_class, force=force, session=session
    )
    return profile


def fetch_sherdog_fighter_detailed(
    name: str,
    *,
    weight_class: str | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Fetch and cache one fighter. Returns (profile|None, failure_reason)."""
    clean = clean_fighter_name(name)
    if not clean:
        return None, "empty_name"

    fighters = load_sherdog_fighters()
    if not force and not fighters.empty:
        from src.fighter_aliases import names_match_aliased

        for _, row in fighters.iterrows():
            if names_match_aliased(clean, str(row.get("name") or "")):
                if weight_class:
                    wc = str(row.get("weight_class") or "").lower()
                    if wc and weight_class.lower()[:4] not in wc and wc[:4] not in weight_class.lower():
                        continue
                return row.to_dict(), ""

    hit = search_sherdog_fighter(clean, weight_class=weight_class, session=session)
    if not hit:
        return None, "name_mismatch"

    sess = session or _session()
    try:
        time.sleep(max(0.25, float(getattr(config, "REQUEST_DELAY_SEC", 0.5))))
        resp = sess.get(hit["url"], timeout=config.REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        html = resp.text
    except requests.Timeout:
        return None, "timeout"
    except Exception as exc:
        logger.info("Sherdog fetch failed for %s: %s", clean, exc)
        return None, "request_error"

    if "just a moment" in html.lower() or "cf-browser-verification" in html.lower():
        return None, "blocked"
    if len(html) < 2000:
        return None, "blocked"

    profile, fights = parse_sherdog_fighter_html(html, url=hit["url"])
    if not profile.get("sherdog_id"):
        profile["sherdog_id"] = hit["sherdog_id"]
    if not profile.get("name"):
        profile["name"] = clean
    if not profile.get("name") and not fights:
        return None, "empty_infobox"

    fighters = load_sherdog_fighters()
    fighters = pd.concat([fighters, pd.DataFrame([profile])], ignore_index=True)
    _save_fighters(fighters)

    if fights:
        fight_df = load_sherdog_fights()
        fight_df = pd.concat([fight_df, pd.DataFrame(fights)], ignore_index=True)
        _save_fights(fight_df)

    return profile, ""


_SHERDOG_INDEX_MEM: dict[str, Any] = {"mtime": None, "by_key": {}, "rows": []}


def match_sherdog_row(
    name: str,
    *,
    weight_class: str | None = None,
) -> dict[str, Any] | None:
    """Match a fighter against the local Sherdog cache (no network)."""
    from src.fighter_aliases import alias_lookup_names, normalize_alias_key

    clean = clean_fighter_name(name)
    if not clean:
        return None

    path = SHERDOG_FIGHTERS_CACHE
    mtime = path.stat().st_mtime if path.exists() else None
    if _SHERDOG_INDEX_MEM["mtime"] != mtime or not _SHERDOG_INDEX_MEM["by_key"]:
        fighters = load_sherdog_fighters()
        by_key: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        if not fighters.empty:
            for _, row in fighters.iterrows():
                rowd = row.to_dict()
                rows.append(rowd)
                n = clean_fighter_name(str(rowd.get("name") or ""))
                if not n:
                    continue
                for alias in alias_lookup_names(n):
                    by_key.setdefault(normalize_alias_key(alias), rowd)
        _SHERDOG_INDEX_MEM["mtime"] = mtime
        _SHERDOG_INDEX_MEM["by_key"] = by_key
        _SHERDOG_INDEX_MEM["rows"] = rows

    by_key = _SHERDOG_INDEX_MEM["by_key"]
    rows = _SHERDOG_INDEX_MEM["rows"]
    if not rows:
        return None

    matches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for alias in alias_lookup_names(clean):
        hit = by_key.get(normalize_alias_key(alias))
        if hit is None:
            continue
        sid = str(hit.get("sherdog_id") or hit.get("name") or id(hit))
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        matches.append(hit)
    if not matches:
        for rowd in rows:
            if _fighters_same_person(clean, str(rowd.get("name") or "")):
                matches.append(rowd)
                break
    if not matches:
        return None
    if weight_class:
        wc_l = weight_class.lower()
        wc_hits = [
            m
            for m in matches
            if str(m.get("weight_class") or "").lower()
            and (
                wc_l[:5] in str(m.get("weight_class") or "").lower()
                or str(m.get("weight_class") or "").lower()[:5] in wc_l
            )
        ]
        if wc_hits:
            return wc_hits[0]
    return matches[0]


def sherdog_record_as_of(
    name: str,
    as_of: pd.Timestamp | str | None,
    *,
    weight_class: str | None = None,
) -> dict[str, float]:
    """Leakage-safe career record using only Sherdog fights strictly before ``as_of``."""
    profile = match_sherdog_row(name, weight_class=weight_class)
    fights = load_sherdog_fights()
    clean = clean_fighter_name(name)
    ts = pd.to_datetime(as_of, errors="coerce")

    out = {
        "sherdog_wins": np.nan,
        "sherdog_losses": np.nan,
        "sherdog_draws": np.nan,
        "sherdog_fight_count": np.nan,
        "sherdog_win_rate": np.nan,
        "sherdog_finish_rate": np.nan,
    }
    if fights.empty and profile:
        # Do NOT use scraped career W-L here — those are current totals and
        # leak post-as_of results into historical rows. Leave NaN instead.
        return out

    if fights.empty:
        return out

    sid = str(profile.get("sherdog_id") or "") if profile else ""
    mask = fights["fighter"].map(
        lambda x: bool(_fighters_same_person(clean, str(x)))
    ).astype(bool)
    if sid:
        mask = mask | (fights["sherdog_id"].astype(str) == sid).fillna(False)
    sub = fights.loc[mask].copy()
    if pd.isna(ts):
        # Without an as-of date we cannot safely filter — refuse full-career totals.
        return out
    sub = sub[sub["bout_date"].notna() & (sub["bout_date"] < ts.normalize())]
    if sub.empty:
        return out

    res = sub["result"].astype(str).str.upper()
    wins = float((res == "WIN").sum())
    losses = float((res == "LOSS").sum())
    draws = float(res.isin(["DRAW", "NC", "N/C"]).sum())
    decided = wins + losses
    methods = sub.loc[res == "WIN", "method"].astype(str).str.lower()
    finishes = float(methods.str.contains(r"ko|tko|sub", regex=True, na=False).sum())
    out.update(
        {
            "sherdog_wins": wins,
            "sherdog_losses": losses,
            "sherdog_draws": draws,
            "sherdog_fight_count": wins + losses + draws,
            "sherdog_win_rate": wins / decided if decided > 0 else np.nan,
            "sherdog_finish_rate": finishes / wins if wins > 0 else np.nan,
        }
    )
    return out


def refresh_sherdog_for_names(
    names: list[str],
    *,
    max_fetch: int = 40,
    force: bool = False,
) -> dict[str, Any]:
    """Best-effort refresh for a name list. Never raises on scrape failure."""
    stats: dict[str, Any] = {
        "requested": 0,
        "fetched": 0,
        "cached": 0,
        "failed": 0,
        "failure_reasons": {},
        "failures": [],
    }
    reasons: dict[str, int] = stats["failure_reasons"]
    sess = _session()
    seen: set[str] = set()
    attempted = 0
    for raw in names:
        clean = clean_fighter_name(str(raw))
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        stats["requested"] += 1
        existing = match_sherdog_row(clean)
        if existing and not force:
            stats["cached"] += 1
            continue
        if attempted >= max_fetch:
            break
        attempted += 1
        try:
            profile, reason = fetch_sherdog_fighter_detailed(clean, force=force, session=sess)
            if profile:
                stats["fetched"] += 1
            else:
                stats["failed"] += 1
                key = reason or "unknown"
                reasons[key] = int(reasons.get(key, 0)) + 1
                stats["failures"].append({"name": clean, "reason": key})
                logger.info("Sherdog miss [%s]: %s", key, clean)
        except Exception as exc:
            stats["failed"] += 1
            key = "timeout" if "timeout" in str(exc).lower() else "exception"
            reasons[key] = int(reasons.get(key, 0)) + 1
            stats["failures"].append({"name": clean, "reason": key, "detail": str(exc)[:120]})
            logger.info("Sherdog refresh error for %s: %s", clean, exc)
    return stats


def fill_history_from_sherdog(history: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe Sherdog career features onto long history rows."""
    if history.empty or "fighter" not in history.columns:
        return history
    out = history.copy()
    date_col = config.DATE_COLUMN
    for col in (
        "sherdog_win_rate",
        "sherdog_fight_count",
        "sherdog_finish_rate",
        "sherdog_wins",
        "sherdog_losses",
    ):
        if col not in out.columns:
            out[col] = np.nan

    fighters = load_sherdog_fighters()
    fights = load_sherdog_fights()
    if fighters.empty and fights.empty:
        return out

    # Pre-index profiles by cleaned name for O(1) physical fills.
    prof_by_name: dict[str, dict[str, Any]] = {}
    for _, row in fighters.iterrows():
        key = clean_fighter_name(str(row.get("name") or "")).lower()
        if key:
            prof_by_name[key] = row.to_dict()

    # Map history fighter names → Sherdog profile (exact key, then fuzzy once).
    hist_names = {
        clean_fighter_name(str(n)).lower()
        for n in out["fighter"].dropna().astype(str)
        if clean_fighter_name(str(n))
    }
    name_to_prof: dict[str, dict[str, Any]] = {}
    for hn in hist_names:
        if hn in prof_by_name:
            name_to_prof[hn] = prof_by_name[hn]
            continue
        for pk, prof in prof_by_name.items():
            if _fighters_same_person(hn, pk):
                name_to_prof[hn] = prof
                break

    if not name_to_prof:
        return out

    # Pre-index fight history by sherdog_id / fighter for fast as-of rolls.
    fights_by_sid: dict[str, pd.DataFrame] = {}
    fights_by_name: dict[str, pd.DataFrame] = {}
    if not fights.empty:
        if "sherdog_id" in fights.columns:
            for sid, grp in fights.groupby(fights["sherdog_id"].astype(str), sort=False):
                if sid and sid != "nan":
                    fights_by_sid[str(sid)] = grp
        if "fighter" in fights.columns:
            for fname, grp in fights.groupby(
                fights["fighter"].map(lambda x: clean_fighter_name(str(x)).lower()),
                sort=False,
            ):
                if fname:
                    fights_by_name[str(fname)] = grp

    def _record_from_group(grp: pd.DataFrame, as_of: Any) -> dict[str, float]:
        out_rec = {
            "sherdog_wins": np.nan,
            "sherdog_losses": np.nan,
            "sherdog_draws": np.nan,
            "sherdog_fight_count": np.nan,
            "sherdog_win_rate": np.nan,
            "sherdog_finish_rate": np.nan,
        }
        if grp is None or grp.empty:
            return out_rec
        ts = pd.to_datetime(as_of, errors="coerce")
        # Require a fight date — never stamp current career totals onto rows.
        if pd.isna(ts) or "bout_date" not in grp.columns:
            return out_rec
        sub = grp[grp["bout_date"].notna() & (grp["bout_date"] < ts.normalize())]
        if sub.empty:
            return out_rec
        res = sub["result"].astype(str).str.upper()
        wins = float((res == "WIN").sum())
        losses = float((res == "LOSS").sum())
        draws = float(res.isin(["DRAW", "NC", "N/C"]).sum())
        decided = wins + losses
        methods = sub.loc[res == "WIN", "method"].astype(str).str.lower()
        finishes = float(methods.str.contains(r"ko|tko|sub", regex=True, na=False).sum())
        out_rec.update(
            {
                "sherdog_wins": wins,
                "sherdog_losses": losses,
                "sherdog_draws": draws,
                "sherdog_fight_count": wins + losses + draws,
                "sherdog_win_rate": wins / decided if decided > 0 else np.nan,
                "sherdog_finish_rate": finishes / wins if wins > 0 else np.nan,
            }
        )
        return out_rec

    filled = 0
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for idx in out.index:
        name = clean_fighter_name(str(out.at[idx, "fighter"] or ""))
        key = name.lower()
        prof = name_to_prof.get(key)
        if not prof:
            continue
        ts = pd.to_datetime(out.at[idx, date_col] if date_col in out.columns else None, errors="coerce")
        day = str(ts.date()) if pd.notna(ts) else ""
        cache_key = (key, day)
        if cache_key not in cache:
            sid = str(prof.get("sherdog_id") or "")
            grp = fights_by_sid.get(sid) if sid else None
            if grp is None:
                grp = fights_by_name.get(key)
            # Dated fight log only — no scraped career W-L fallback (leakage).
            rec = _record_from_group(grp if grp is not None else pd.DataFrame(), ts)
            cache[cache_key] = rec
        rec = cache[cache_key]
        for col, val in rec.items():
            if col in out.columns and pd.isna(out.at[idx, col]) and pd.notna(val):
                out.at[idx, col] = val
                filled += 1
        for hist_col, src_col in (("height_in", "height_in"), ("reach_in", "reach_in")):
            if hist_col in out.columns and pd.isna(out.at[idx, hist_col]):
                v = prof.get(src_col)
                if v is not None and pd.notna(v) and float(v) > 0:
                    out.at[idx, hist_col] = float(v)
                    filled += 1
    if filled:
        logger.info("Sherdog fill: %s cells (%s fighters)", filled, len(name_to_prof))
    return out


def sherdog_coverage(fighter_names: list[str] | pd.Series) -> dict[str, float]:
    names = [clean_fighter_name(str(n)) for n in fighter_names if clean_fighter_name(str(n))]
    uniq = sorted(set(n for n in names if n))
    if not uniq:
        return {
            "n": 0.0,
            "pct_fighters": 0.0,
            "n_matched": 0.0,
            "pct_with_history": 0.0,
            "n_with_history": 0.0,
        }
    fights = load_sherdog_fights()
    fight_fighters: set[str] = set()
    if not fights.empty and "fighter" in fights.columns:
        fight_fighters = {
            clean_fighter_name(str(n)).lower()
            for n in fights["fighter"].dropna()
            if clean_fighter_name(str(n))
        }
    matched = 0
    with_hist = 0
    for n in uniq:
        row = match_sherdog_row(n)
        if row is None:
            continue
        matched += 1
        key = clean_fighter_name(n).lower()
        sid = str(row.get("sherdog_id") or "")
        has_hist = key in fight_fighters
        if not has_hist and sid and not fights.empty and "sherdog_id" in fights.columns:
            has_hist = bool((fights["sherdog_id"].astype(str) == sid).any())
        if has_hist:
            with_hist += 1
    n = len(uniq)
    return {
        "n": float(n),
        "pct_fighters": matched / n,
        "n_matched": float(matched),
        "pct_with_history": with_hist / n,
        "n_with_history": float(with_hist),
    }
