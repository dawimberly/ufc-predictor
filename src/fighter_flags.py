"""Manual fighter integrity / skip flags (user-curated).

Loaded from ``data/fighter_flags.json``. Active ``skip`` flags block moneyline
and prop candidates involving that fighter. Display-only context also surfaces
the flag. Not a model feature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import config
from src.data_loader import clean_fighter_name

logger = logging.getLogger(__name__)

FLAGS_PATH = Path(config.DATA_DIR) / "fighter_flags.json"


@dataclass(frozen=True)
class FighterFlag:
    name: str
    reason: str
    note: str
    action: str
    fight: str = ""
    event: str = ""
    date: str = ""

    def label(self) -> str:
        bit = self.note or self.reason or "flagged"
        return f"FLAG: {self.name} — {bit}"


def _norm(name: Any) -> str:
    return clean_fighter_name(str(name or "")).casefold().strip()


@lru_cache(maxsize=4)
def _load_raw(path_str: str, mtime_ns: int) -> tuple[FighterFlag, ...]:
    path = Path(path_str)
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("fighter_flags load failed: %s", exc)
        return ()
    rows = raw.get("fighters") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return ()
    out: list[FighterFlag] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("active", True):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append(
            FighterFlag(
                name=name,
                reason=str(row.get("reason") or "flagged"),
                note=str(row.get("note") or ""),
                action=str(row.get("action") or "skip").strip().lower(),
                fight=str(row.get("fight") or ""),
                event=str(row.get("event") or ""),
                date=str(row.get("date") or ""),
            )
        )
    return tuple(out)


def reload_fighter_flags() -> None:
    """Clear cache after editing fighter_flags.json."""
    _load_raw.cache_clear()
    _alias_index.cache_clear()


def list_active_flags(*, path: Path | None = None) -> list[FighterFlag]:
    p = path or FLAGS_PATH
    mtime = int(p.stat().st_mtime_ns) if p.is_file() else 0
    return list(_load_raw(str(p), mtime))


@lru_cache(maxsize=4)
def _alias_index(path_str: str, mtime_ns: int) -> dict[str, FighterFlag]:
    p = Path(path_str)
    raw_fighters: list[dict[str, Any]] = []
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows = data.get("fighters") if isinstance(data, dict) else data
            if isinstance(rows, list):
                raw_fighters = [r for r in rows if isinstance(r, dict)]
        except Exception:
            raw_fighters = []
    idx: dict[str, FighterFlag] = {}
    for row in raw_fighters:
        if not row.get("active", True):
            continue
        flag = FighterFlag(
            name=str(row.get("name") or "").strip(),
            reason=str(row.get("reason") or "flagged"),
            note=str(row.get("note") or ""),
            action=str(row.get("action") or "skip").strip().lower(),
            fight=str(row.get("fight") or ""),
            event=str(row.get("event") or ""),
            date=str(row.get("date") or ""),
        )
        if not flag.name:
            continue
        keys = {_norm(flag.name)}
        for a in row.get("aliases") or []:
            k = _norm(a)
            if k:
                keys.add(k)
        for k in keys:
            idx[k] = flag
    return idx


def lookup_fighter_flag(name: Any, *, path: Path | None = None) -> FighterFlag | None:
    p = path or FLAGS_PATH
    mtime = int(p.stat().st_mtime_ns) if p.is_file() else 0
    return _alias_index(str(p), mtime).get(_norm(name))


def fight_integrity_flags(
    fighter_1: Any,
    fighter_2: Any = None,
    *,
    path: Path | None = None,
) -> list[FighterFlag]:
    """Flags hitting either corner (deduped)."""
    seen: set[str] = set()
    out: list[FighterFlag] = []
    for name in (fighter_1, fighter_2):
        flag = lookup_fighter_flag(name, path=path)
        if flag is None:
            continue
        key = _norm(flag.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(flag)
    return out


def should_skip_fight(
    fighter_1: Any,
    fighter_2: Any = None,
    *,
    path: Path | None = None,
) -> tuple[bool, str]:
    """True if either fighter has an active skip flag."""
    hits = [
        f
        for f in fight_integrity_flags(fighter_1, fighter_2, path=path)
        if f.action == "skip"
    ]
    if not hits:
        return False, ""
    names = ", ".join(f.name for f in hits)
    note = hits[0].note or hits[0].reason
    return True, f"fighter_flag:{names}|{note}"


def format_flag_badge(fighter_1: Any, fighter_2: Any = None) -> str | None:
    hits = fight_integrity_flags(fighter_1, fighter_2)
    if not hits:
        return None
    names = " + ".join(f.name for f in hits)
    return f"Integrity FLAG: {names} (skip)"
