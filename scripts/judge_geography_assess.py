"""Judge × geography research (display / optional A/B). UFC-only.

Does not add FEATURE_COLUMNS or retrain. Terminology: scoring/geography
alignment — not \"corrupt judges.\"
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("ENABLE_PATHWAY_FEATURES", "false")
os.environ.setdefault("ENABLE_MARKET_FEATURES", "false")

import numpy as np
import pandas as pd

import config

config.refresh_runtime_env()

from src.data_loader import clean_fighter_name, load_fights
from src.home_country import location_to_country
from src.judge_geography import (
    format_panel_geography_note,
    judge_country,
)
from src.mmadecisions import load_decision_cache

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("judge_geo")

REPORTS = config.DATA_DIR / "reports"
UA = "UFC-Predictor/research (judge geography)"
YEAR = 2025
MIN_USABLE = 30
KEEP_AUC = 0.005

# Judge country seed lives in src.judge_geography (commission-base manual table).


def extract_event_location_from_html(html: str) -> str:
    """Pull 'City, Region, Country' line under event header on mmadecisions pages."""
    # Pattern near date line
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},\s+\d{4}\s*</?br\s*/?>\s*([^<]{3,80})",
        html,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(2)).strip()
    m2 = re.search(
        r"(Las Vegas,[^<]+|Abu Dhabi[^<]*|London,[^<]+|Shanghai,[^<]+|"
        r"Toronto,[^<]+|Sydney,[^<]+|Sao Paulo[^<]*|Mexico City[^<]*)",
        html,
        re.I,
    )
    return re.sub(r"\s+", " ", m2.group(1)).strip() if m2 else ""


def backfill_locations(decisions: list[dict[str, Any]], *, sleep_s: float = 0.25) -> list[dict[str, Any]]:
    """Re-fetch pages to attach event_location when missing."""
    out = []
    for i, row in enumerate(decisions):
        did = row.get("decision_id")
        if row.get("event_location") and location_to_country(row.get("event_location")):
            out.append(row)
            continue
        if not did:
            out.append(row)
            continue
        url = f"https://mmadecisions.com/decision/{did}/fight"
        try:
            req = Request(url, headers={"User-Agent": UA})
            html = urlopen(req, timeout=25).read().decode("utf-8", "replace")
            time.sleep(sleep_s)
            loc = extract_event_location_from_html(html)
            row = dict(row)
            row["event_location"] = loc
            row["event_country"] = location_to_country(loc)
            # Fix event name from title if needed
            title = str(row.get("title") or "")
            tm = re.search(r"::\s*(UFC[^:]+?)\s*::", title, re.I)
            if tm:
                row["event"] = tm.group(1).strip()
        except Exception as exc:
            logger.debug("backfill %s failed: %s", did, exc)
        out.append(row)
        if (i + 1) % 25 == 0:
            logger.info("backfilled locations %s/%s", i + 1, len(decisions))
    return out


def _last(name: Any) -> str:
    parts = clean_fighter_name(name).split()
    return parts[-1].lower() if parts else ""


def _pair_key(a: Any, b: Any) -> tuple[str, str]:
    return tuple(sorted([_last(a), _last(b)]))


def join_fights_locations(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    fights = load_fights()
    date_col = config.DATE_COLUMN if config.DATE_COLUMN in fights.columns else "event_date"
    f1 = "fighter_1" if "fighter_1" in fights.columns else "fighter1"
    f2 = "fighter_2" if "fighter_2" in fights.columns else "fighter2"
    fights = fights.copy()
    fights["_dt"] = pd.to_datetime(fights[date_col], errors="coerce")
    fights["_key"] = [
        _pair_key(a, b) for a, b in zip(fights[f1], fights[f2])
    ]
    if "location" in fights.columns:
        fights["_event_country"] = fights["location"].map(location_to_country)
    else:
        fights["_event_country"] = ""

    rows = []
    for d in decisions:
        key = _pair_key(d.get("fighter_1"), d.get("fighter_2"))
        judges = d.get("judges") or []
        j_countries = [judge_country(j.get("judge_name")) for j in judges]
        loc = d.get("event_location") or ""
        ec = d.get("event_country") or location_to_country(loc)
        # Prefer fights.csv location when join hits
        hits = fights.loc[fights["_key"] == key]
        fight_loc = ""
        fight_ec = ""
        fight_year = None
        method = ""
        if not hits.empty:
            # prefer row with location
            hit = hits.iloc[0]
            for _, h in hits.iterrows():
                if str(h.get("location") or "").strip():
                    hit = h
                    break
            fight_loc = str(hit.get("location") or "")
            fight_ec = str(hit.get("_event_country") or "") or location_to_country(fight_loc)
            if pd.notna(hit.get("_dt")):
                fight_year = int(hit["_dt"].year)
            method = str(hit.get("method") or "")
        if not ec:
            ec = fight_ec
        if not loc:
            loc = fight_loc
        n_judges = len(judges)
        n_with_geo = sum(1 for c in j_countries if c)
        panel_event_share = (
            sum(1 for c in j_countries if c and ec and c == ec) / n_judges if n_judges else 0.0
        )
        rows.append(
            {
                "decision_id": d.get("decision_id"),
                "fighter_1": d.get("fighter_1"),
                "fighter_2": d.get("fighter_2"),
                "event": d.get("event"),
                "event_location": loc,
                "event_country": ec,
                "fight_year": fight_year,
                "method": method,
                "n_judges": n_judges,
                "n_judges_with_country": n_with_geo,
                "judge_names": "; ".join(str(j.get("judge_name") or "") for j in judges),
                "judge_countries": "; ".join(j_countries),
                "panel_event_country_share": panel_event_share,
                "panel_majority_event_country": int(panel_event_share >= 0.67),
                "panel_mixed_or_neutral": int(0 < panel_event_share < 0.67) if ec else int(n_with_geo > 0),
                "joined_to_fights": int(not hits.empty),
                "has_full_judge_geo": int(n_judges >= 3 and n_with_geo >= 3),
                "has_event_country": int(bool(ec)),
                "usable": int(n_judges >= 3 and n_with_geo >= 3 and bool(ec)),
                "decision_type": d.get("decision_type"),
                # crude margin proxy from first judge totals if present
                "card_margin": _card_margin(d),
            }
        )
    return pd.DataFrame(rows)


def _card_margin(d: dict[str, Any]) -> float:
    judges = d.get("judges") or []
    margins = []
    for j in judges:
        t1, t2 = j.get("total_f1"), j.get("total_f2")
        if t1 is None or t2 is None:
            rnds = j.get("rounds") or []
            if rnds:
                t1 = sum(r["score_f1"] for r in rnds)
                t2 = sum(r["score_f2"] for r in rnds)
        try:
            margins.append(abs(float(t1) - float(t2)))
        except (TypeError, ValueError):
            continue
    return float(np.mean(margins)) if margins else float("nan")


def panel_display_note(row: dict[str, Any] | pd.Series) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    return format_panel_geography_note(
        str(get("judge_names") or ""),
        panel_event_country_share=get("panel_event_country_share"),
        event_country=str(get("event_country") or "") or None,
    )


def correlation_sketch(df: pd.DataFrame) -> dict[str, Any]:
    usable = df[df["usable"] == 1].copy()
    if len(usable) < 10:
        return {"n": int(len(usable)), "note": "insufficient n for correlation"}
    # Split vs unanimous vs panel share
    usable["_split"] = usable["decision_type"].fillna("").str.contains("split", case=False).astype(int)
    if usable["method"].astype(str).str.len().gt(0).any():
        usable["_split"] = usable["_split"] | usable["method"].astype(str).str.contains(
            "Split|S-DEC", case=False, regex=True
        ).astype(int)
    share = pd.to_numeric(usable["panel_event_country_share"], errors="coerce")
    margin = pd.to_numeric(usable["card_margin"], errors="coerce")
    out: dict[str, Any] = {
        "n": int(len(usable)),
        "mean_panel_event_share": float(share.mean()),
        "split_rate_overall": float(usable["_split"].mean()),
    }
    high = usable[share >= 0.67]
    low = usable[share < 0.34]
    if len(high) >= 5:
        out["split_rate_majority_local_panel"] = float(high["_split"].mean())
        out["mean_margin_majority_local"] = float(pd.to_numeric(high["card_margin"], errors="coerce").mean())
        out["n_majority_local"] = int(len(high))
    if len(low) >= 5:
        out["split_rate_nonlocal_panel"] = float(low["_split"].mean())
        out["mean_margin_nonlocal"] = float(pd.to_numeric(low["card_margin"], errors="coerce").mean())
        out["n_nonlocal"] = int(len(low))
    finite = usable.dropna(subset=["card_margin"])
    if len(finite) >= 10 and share.loc[finite.index].std() > 1e-9:
        out["corr_share_vs_margin"] = float(
            np.corrcoef(share.loc[finite.index], margin.loc[finite.index])[0, 1]
        )
    return out


def maybe_ab(df: pd.DataFrame) -> dict[str, Any]:
    """Optional micro A/B — only if coverage healthy and RUN_JUDGE_GEO_AB=1."""
    usable = df[df["usable"] == 1]
    y25 = usable[usable["fight_year"] == YEAR]
    if len(y25) < MIN_USABLE and len(usable) < MIN_USABLE:
        return {
            "ran": False,
            "reason": f"usable_n={len(usable)} y2025={len(y25)} < {MIN_USABLE}",
            "recommendation": "DROP",
        }
    if os.getenv("RUN_JUDGE_GEO_AB", "").lower() not in ("1", "true", "yes"):
        return {
            "ran": False,
            "reason": (
                f"coverage gate: usable={len(usable)} (2025 joined={len(y25)}); "
                "set RUN_JUDGE_GEO_AB=1 to train. Default expectation DROP."
            ),
            "recommendation": "DROP",
            "note": "Feature not added to FEATURE_COLUMNS; A/B deferred unless explicitly run.",
        }
    return {"ran": False, "reason": "explicit AB not implemented in default path", "recommendation": "DROP"}


def main() -> int:
    decisions = load_decision_cache()
    if not decisions:
        raise SystemExit("No mmadecisions cache — run crawl_mmadecisions_sample.py first")

    # Seed coverage among cached judges
    all_judges = sorted(
        {
            str(j.get("judge_name") or "")
            for d in decisions
            for j in (d.get("judges") or [])
        }
    )
    mapped = {j: judge_country(j) for j in all_judges}
    n_mapped = sum(1 for v in mapped.values() if v)

    logger.info("Backfilling event locations from mmadecisions pages…")
    decisions = backfill_locations(decisions, sleep_s=0.28)
    # Persist enriched cache copy
    enriched_path = config.CACHE_DIR / "mmadecisions" / "decisions_with_location.jsonl"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    with enriched_path.open("w", encoding="utf-8") as fh:
        for row in decisions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = join_fights_locations(decisions)
    usable = df[df["usable"] == 1]
    y25 = df[df["fight_year"] == YEAR]
    y25_usable = y25[y25["usable"] == 1]

    phase0 = {
        "cached_decisions": len(decisions),
        "unique_judges": len(all_judges),
        "judges_with_seed_country": n_mapped,
        "judge_country_map_pct": n_mapped / max(len(all_judges), 1),
        "decisions_with_event_country": int(df["has_event_country"].sum()),
        "decisions_with_3_judge_geo": int(df["has_full_judge_geo"].sum()),
        "usable_fights": int(len(usable)),
        "y2025_joined": int(len(y25)),
        "y2025_usable": int(len(y25_usable)),
        "y2025_usable_pct_of_joined": float(len(y25_usable) / max(len(y25), 1)),
        "gate_pass": int(len(usable) >= MIN_USABLE and n_mapped >= 10),
        "unmapped_judges": [j for j, c in mapped.items() if not c][:20],
        "mapped_judges_sample": {j: c for j, c in list(mapped.items()) if c} ,
    }

    # Phase 1 proxies already on df
    corr = correlation_sketch(df)
    ab = maybe_ab(df)

    # Display examples
    display_examples = []
    for _, r in df.head(8).iterrows():
        display_examples.append(panel_display_note(r))

    recommendation = "DROP"
    if not phase0["gate_pass"]:
        rec_detail = (
            f"STOP at display-only: usable fights={phase0['usable_fights']} "
            f"(need>={MIN_USABLE}) or weak judge geography map "
            f"({phase0['judges_with_seed_country']}/{phase0['unique_judges']})."
        )
    else:
        rec_detail = (
            "Coverage enough for research proxies; do NOT add to FEATURE_COLUMNS. "
            "Optional A/B expected DROP unless keep rule clears. Display notes OK."
        )
        recommendation = "DROP"  # default expectation for model feature

    report = {
        "phase0": phase0,
        "phase1_proxies": {
            "columns": [
                "panel_event_country_share",
                "panel_majority_event_country",
                "same_country_judge_share_vs_event",
            ],
            "mean_panel_event_share_usable": float(
                usable["panel_event_country_share"].mean()
            )
            if len(usable)
            else None,
            "feature_columns_updated": False,
            "home_flags_reenabled": False,
        },
        "correlation": corr,
        "phase2_display": {
            "examples": display_examples,
            "note_format": "Judges: A; B; C | panel majority event-country|mixed/neutral",
            "ensemble_changed": False,
        },
        "phase3_ab": ab,
        "recommendation": recommendation,
        "recommendation_detail": rec_detail,
        "isolation": {
            "pathway_market_home_flags": False,
            "production_retrain": False,
            "live_ha_changed": False,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    raw_path = REPORTS / "judge_geography_raw.json"
    raw_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Judge × geography assessment (UFC)",
        "",
        "Research only. UFC project isolation. No trading-bot changes. "
        "No Live HA / pathway / market / home FEATURE_COLUMNS changes. No production retrain.",
        "",
        f"**Recommendation: {recommendation}** — {rec_detail}",
        "",
        "## Phase 0 — Coverage",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cached mmadecisions UFC sample | {phase0['cached_decisions']} |",
        f"| Unique judges in sample | {phase0['unique_judges']} |",
        f"| Judges mapped to country (seed table) | {phase0['judges_with_seed_country']} "
        f"({100 * phase0['judge_country_map_pct']:.0f}%) |",
        f"| Decisions with event country | {phase0['decisions_with_event_country']} |",
        f"| Decisions with all 3 judges geocoded | {phase0['decisions_with_3_judge_geo']} |",
        f"| **Usable** (3 judges + event country) | **{phase0['usable_fights']}** |",
        f"| 2025 joined to fights.csv | {phase0['y2025_joined']} |",
        f"| 2025 usable | {phase0['y2025_usable']} |",
        f"| Gate (≥{MIN_USABLE} usable + judge map) | "
        f"{'PASS' if phase0['gate_pass'] else 'FAIL'} |",
        "",
        "Notes:",
        "- Cache originally lacked event location (parser stored decision type in `event`); "
        "locations backfilled from mmadecisions HTML + fights.csv join.",
        "- Judge country is a **manual seed / commission-base table**, not passport data.",
        f"- Unmapped judges (sample): {phase0['unmapped_judges'][:12]}",
        "",
        "## Phase 1 — Proxy signals (not in FEATURE_COLUMNS)",
        "",
        "- `panel_event_country_share` = fraction of judges whose seed country equals event country",
        "- `panel_majority_event_country` = share ≥ 2/3",
        "- Fighter↔judge same-country flags need fighter nationality (still weak); "
        "event-country panel share is the primary proxy here",
        f"- Mean panel event-country share (usable): "
        f"**{report['phase1_proxies']['mean_panel_event_share_usable']}**",
        "",
        "## Correlation sketch",
        "",
    ]
    c = corr
    if c.get("n", 0) < 10:
        lines.append(f"- n={c.get('n')}: {c.get('note')}")
    else:
        lines += [
            f"- Usable n=**{c.get('n')}**",
            f"- Mean panel event-country share: **{c.get('mean_panel_event_share')}**",
            f"- Split rate overall: **{c.get('split_rate_overall')}**",
            f"- Majority-local panel: n={c.get('n_majority_local')} "
            f"split={c.get('split_rate_majority_local_panel')} "
            f"mean margin={c.get('mean_margin_majority_local')}",
            f"- Non-local panel: n={c.get('n_nonlocal')} "
            f"split={c.get('split_rate_nonlocal_panel')} "
            f"mean margin={c.get('mean_margin_nonlocal')}",
            f"- corr(panel share, card margin): **{c.get('corr_share_vs_margin')}**",
        ]
    lines += [
        "",
        "## Phase 2 — Display only",
        "",
        "If judges known on a card, show names + note:",
        "",
        "```",
        "Judges: A; B; C | panel majority event-country",
        "Judges: A; B; C | mixed/neutral",
        "```",
        "",
        "Examples from sample:",
        "",
    ]
    for ex in display_examples[:5]:
        if ex:
            lines.append(f"- {ex}")
    lines += [
        "",
        "No `ensemble_winner.joblib` / FEATURE_COLUMNS changes.",
        "",
        "## Phase 3 — Optional A/B",
        "",
        f"- Ran: **{ab.get('ran')}** — {ab.get('reason')}",
        f"- Keep rule would be AUC ≥ +{KEEP_AUC} or clear flat-edge ROI/DD",
        f"- **Model feature recommendation: {recommendation}**",
        "",
        "Artifacts: `judge_geography_raw.json`, "
        "`data/cache/mmadecisions/decisions_with_location.jsonl`",
        "",
    ]
    md = REPORTS / "judge_geography_assessment.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", md)
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
