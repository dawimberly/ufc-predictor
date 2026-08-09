"""Judge panel geography display helpers (research / UI notes only).

Not a model feature module — do not add to FEATURE_COLUMNS.
"""

from __future__ import annotations

import re
from typing import Any

# Seed: commission / known home base for frequent UFC judges (manual research).
JUDGE_COUNTRY_SEED: dict[str, str] = {
    "sal d'amato": "usa",
    "sal damato": "usa",
    "derek cleary": "usa",
    "chris lee": "usa",
    "michael bell": "usa",
    "mike bell": "usa",
    "adalaide byrd": "usa",
    "adelaide byrd": "usa",
    "tony weeks": "usa",
    "ron mccarthy": "usa",
    "dave hagen": "usa",
    "glenn trowbridge": "usa",
    "marcos rosales": "usa",
    "doug crosby": "usa",
    "chris leben": "usa",
    "brian puccillo": "usa",
    "will fisher": "usa",
    "eric colon": "usa",
    "eric colón": "usa",
    "junichiro kamijo": "japan",
    "ben cartlidge": "uk",
    "david lethaby": "uk",
    "mark collett": "uk",
    "clemens werner": "germany",
    "anders ohlsson": "sweden",
    "vito paolillo": "italy",
    "darryl ransom": "usa",
    "guilherme bravo": "brazil",
    "dave tirelli": "usa",
}


def _norm_judge(name: str) -> str:
    s = str(name or "").lower().replace("\xa0", " ")
    s = (
        s.replace("ó", "o")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ú", "u")
    )
    s = re.sub(r"[^a-z0-9'\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def judge_country(name: str, seed: dict[str, str] | None = None) -> str:
    seed = seed or JUDGE_COUNTRY_SEED
    key = _norm_judge(name)
    if key in seed:
        return seed[key]
    last = key.split()[-1] if key else ""
    hits = [v for k, v in seed.items() if k.endswith(last) and last]
    if len(set(hits)) == 1:
        return hits[0]
    return ""


def format_panel_geography_note(
    judge_names: list[str] | str,
    *,
    panel_event_country_share: float | None,
    event_country: str | None = None,
) -> str:
    """Display strip: judge names + majority event-country / mixed/neutral."""
    if isinstance(judge_names, str):
        names = judge_names.strip()
    else:
        names = "; ".join(str(n) for n in (judge_names or []) if n)
    if not names:
        return ""
    ec = (event_country or "").strip()
    try:
        share = float(panel_event_country_share) if panel_event_country_share is not None else None
    except (TypeError, ValueError):
        share = None
    if not ec or share is None:
        label = "panel geography n/a"
    elif share >= 0.67:
        label = "panel majority event-country"
    elif share <= 0.0:
        label = "panel non-local / mixed"
    else:
        label = "mixed/neutral"
    return f"Judges: {names} | {label}"
