"""Weigh-in context: athlete photos + missed-weight notes (display / research).

Not a LightGBM feature. Photos are cached under ``data/cache/weigh_in/``.
Missed-weight / scale notes live in ``data/weigh_in_notes.json`` (manual or
future scrape). Fail-soft on network / missing images.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import config
from src.data_loader import clean_fighter_name

logger = logging.getLogger(__name__)

CACHE_DIR = Path(config.CACHE_DIR) / "weigh_in"
IMAGE_DIR = CACHE_DIR / "images"
NOTES_PATH = Path(config.DATA_DIR) / "weigh_in_notes.json"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class WeighInNote:
    fighter: str
    event: str = ""
    date: str = ""
    missed_weight: bool = False
    weighed_lb: float | None = None
    limit_lb: float | None = None
    note: str = ""

    def line(self) -> str:
        bits: list[str] = []
        if self.missed_weight:
            bits.append("MISSED WEIGHT")
        if self.weighed_lb is not None and self.limit_lb is not None:
            bits.append(f"{self.weighed_lb:g} lb / {self.limit_lb:g} limit")
        elif self.weighed_lb is not None:
            bits.append(f"{self.weighed_lb:g} lb")
        if self.note:
            bits.append(self.note)
        body = " · ".join(bits) if bits else "weigh-in noted"
        return f"Weigh-in ({self.fighter}): {body}"


def athlete_slug(name: Any) -> str:
    """UFC.com-style athlete slug."""
    text = clean_fighter_name(str(name or ""))
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", text)
    ascii_name = "".join(c for c in norm if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.casefold()).strip("-")
    return slug


def athlete_page_url(name: Any) -> str:
    slug = athlete_slug(name)
    if not slug:
        return ""
    return f"https://www.ufc.com/athlete/{quote(slug)}"


@lru_cache(maxsize=2)
def _load_notes(path_str: str, mtime_ns: int) -> tuple[WeighInNote, ...]:
    path = Path(path_str)
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("weigh_in_notes load failed: %s", exc)
        return ()
    rows = raw.get("notes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return ()
    out: list[WeighInNote] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("fighter") or "").strip()
        if not name:
            continue
        weighed = row.get("weighed_lb")
        limit = row.get("limit_lb")
        try:
            weighed_f = float(weighed) if weighed is not None and str(weighed) != "" else None
        except (TypeError, ValueError):
            weighed_f = None
        try:
            limit_f = float(limit) if limit is not None and str(limit) != "" else None
        except (TypeError, ValueError):
            limit_f = None
        missed = bool(row.get("missed_weight"))
        if not missed and weighed_f is not None and limit_f is not None:
            missed = weighed_f > limit_f + 0.05
        out.append(
            WeighInNote(
                fighter=name,
                event=str(row.get("event") or ""),
                date=str(row.get("date") or ""),
                missed_weight=missed,
                weighed_lb=weighed_f,
                limit_lb=limit_f,
                note=str(row.get("note") or ""),
            )
        )
    return tuple(out)


def reload_weigh_in_notes() -> None:
    _load_notes.cache_clear()


def list_weigh_in_notes() -> list[WeighInNote]:
    p = NOTES_PATH
    mtime = int(p.stat().st_mtime_ns) if p.is_file() else 0
    return list(_load_notes(str(p), mtime))


def lookup_weigh_in_note(
    fighter: Any,
    *,
    event: str | None = None,
    date: str | None = None,
) -> WeighInNote | None:
    key = clean_fighter_name(fighter).casefold()
    if not key:
        return None
    hits = [
        n
        for n in list_weigh_in_notes()
        if clean_fighter_name(n.fighter).casefold() == key
    ]
    if not hits:
        return None
    if date:
        d = str(date)[:10]
        dated = [n for n in hits if n.date[:10] == d]
        if dated:
            hits = dated
    if event:
        ev = str(event).casefold()
        ev_hits = [n for n in hits if ev and ev in n.event.casefold()]
        if ev_hits:
            hits = ev_hits
    # Prefer most recently dated note
    hits = sorted(hits, key=lambda n: n.date or "", reverse=True)
    return hits[0]


def format_weigh_in_line(
    fighter_1: Any,
    fighter_2: Any,
    *,
    event: str | None = None,
    date: str | None = None,
) -> str | None:
    parts: list[str] = []
    for name in (fighter_1, fighter_2):
        note = lookup_weigh_in_note(name, event=event, date=date)
        if note is not None:
            parts.append(note.line())
    if not parts:
        return None
    return " | ".join(parts)


def _cached_image_path(name: Any) -> Path:
    slug = athlete_slug(name) or "unknown"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_DIR / f"{slug}.jpg"


def resolve_athlete_image_url(name: Any, *, timeout: float = 8.0) -> str:
    """Best-effort UFC.com athlete og:image URL."""
    url = athlete_page_url(name)
    if not url:
        return ""
    try:
        import requests

        resp = requests.get(
            url,
            headers={"User-Agent": UA, "Accept": "text/html"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return ""
        html = resp.text
        m = re.search(
            r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
            html,
            flags=re.IGNORECASE,
        )
        if not m:
            m = re.search(
                r'content=["\']([^"\']+)["\']\s+property=["\']og:image["\']',
                html,
                flags=re.IGNORECASE,
            )
        if not m:
            return ""
        return str(m.group(1)).strip()
    except Exception as exc:
        logger.debug("athlete image url failed for %s: %s", name, exc)
        return ""


def ensure_athlete_image(
    name: Any,
    *,
    force: bool = False,
    timeout: float = 8.0,
) -> Path | None:
    """Download/cache athlete image; return local path or None."""
    path = _cached_image_path(name)
    if path.is_file() and path.stat().st_size > 500 and not force:
        return path
    img_url = resolve_athlete_image_url(name, timeout=timeout)
    if not img_url:
        return path if path.is_file() else None
    try:
        import requests

        resp = requests.get(
            img_url,
            headers={"User-Agent": UA, "Accept": "image/*"},
            timeout=timeout,
        )
        if resp.status_code != 200 or not resp.content:
            return path if path.is_file() else None
        path.write_bytes(resp.content)
        return path
    except Exception as exc:
        logger.debug("athlete image download failed for %s: %s", name, exc)
        return path if path.is_file() else None


def pair_image_paths(
    fighter_1: Any,
    fighter_2: Any,
    *,
    fetch: bool = True,
) -> tuple[Path | None, Path | None]:
    if fetch:
        return ensure_athlete_image(fighter_1), ensure_athlete_image(fighter_2)
    p1 = _cached_image_path(fighter_1)
    p2 = _cached_image_path(fighter_2)
    return (p1 if p1.is_file() else None, p2 if p2.is_file() else None)


def ensure_notes_file() -> Path:
    """Create empty notes scaffold if missing."""
    if NOTES_PATH.is_file():
        return NOTES_PATH
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "notes": [
            {
                "fighter": "Example Fighter",
                "event": "UFC Fight Night",
                "date": "2026-01-01",
                "missed_weight": False,
                "weighed_lb": None,
                "limit_lb": None,
                "note": "Delete this example; add real weigh-in rows as needed.",
            }
        ],
        "docs": (
            "Manual / research weigh-in notes. Displayed in fight context. "
            "Not added to FEATURE_COLUMNS unless a keep-rule A/B passes."
        ),
    }
    NOTES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return NOTES_PATH
