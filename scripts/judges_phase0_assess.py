"""Phase 0 helpers: Greco DETAILS parse + mmadecisions probe (research only)."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

import config
from src.data_loader import clean_fighter_name

UA = "UFC-Predictor/research (judges assessment; contact: local)"
JUDGE_TOTAL_RE = re.compile(
    r"([A-Za-z][A-Za-z .'\-]+?)\s+(\d{2})\s*[-–]\s*(\d{2})",
    re.UNICODE,
)


def _is_decision_method(method: Any) -> bool:
    s = str(method or "").upper()
    return "DEC" in s or "DECISION" in s


def _is_split(method: Any) -> bool:
    s = str(method or "").upper()
    return "SPLIT" in s or "S-DEC" in s or "S_DEC" in s


def _is_majority(method: Any) -> bool:
    s = str(method or "").upper()
    return "MAJORITY" in s or "M-DEC" in s or "M_DEC" in s


def parse_greco_details(details: str) -> list[dict[str, Any]]:
    """Parse UFCStats DETAILS into judge fight-total cards (not per-round)."""
    text = str(details or "").strip()
    if not text:
        return []
    out: list[dict[str, Any]] = []
    # Split on periods that separate judge blocks
    parts = re.split(r"(?<=\d)\.(?=[A-Za-z])|\|", text)
    if len(parts) == 1:
        parts = re.split(r"\.\s*", text)
    for part in parts:
        part = part.strip(" .;")
        if not part:
            continue
        m = JUDGE_TOTAL_RE.search(part)
        if not m:
            continue
        name = m.group(1).strip(" .")
        if name.lower() in {"round", "referee", "details"}:
            continue
        out.append(
            {
                "judge": name,
                "score_a": int(m.group(2)),
                "score_b": int(m.group(3)),
                "source": "ufcstats_details_totals",
            }
        )
    return out


def assess_greco(path: Path | None = None) -> dict[str, Any]:
    path = path or (config.CACHE_DIR / "ufcstats_greco" / "ufc_fight_results.csv")
    df = pd.read_csv(path)
    dec = df[df["METHOD"].map(_is_decision_method)].copy()
    parsed_rows = 0
    judge_fights: Counter[str] = Counter()
    judge_rounds_est: Counter[str] = Counter()  # estimate rounds from total magnitude
    for _, row in dec.iterrows():
        cards = parse_greco_details(str(row.get("DETAILS") or ""))
        if len(cards) >= 2:
            parsed_rows += 1
        for c in cards:
            judge_fights[c["judge"]] += 1
            # Heuristic: 27-30 band ≈ 3rd; 45-50 ≈ 5rd — not true round-level
            tot = max(c["score_a"], c["score_b"])
            est_r = 5 if tot >= 45 else (3 if tot >= 27 else 0)
            judge_rounds_est[c["judge"]] += est_r
    return {
        "source": "ufcstats_greco_DETAILS",
        "path": str(path),
        "n_fights": int(len(df)),
        "n_decisions": int(len(dec)),
        "n_decisions_with_parsed_judge_totals": parsed_rows,
        "pct_decisions_with_judge_totals": parsed_rows / max(len(dec), 1),
        "unique_judges": len(judge_fights),
        "judges_ge_50_est_rounds": sum(1 for _, n in judge_rounds_est.items() if n >= 50),
        "top_judges_by_fights": judge_fights.most_common(15),
        "per_round_scores": False,
        "note": "DETAILS are fight totals only (e.g. 29-28), not per-round 10-9 cards.",
    }


def assess_fights_csv(path: Path | None = None) -> dict[str, Any]:
    path = path or (config.DATA_DIR / "raw" / "fights.csv")
    df = pd.read_csv(path, low_memory=False)
    date_col = "event_date" if "event_date" in df.columns else "date"
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    dec = df[df["method"].map(_is_decision_method)]
    y25 = dec[dec["_dt"].dt.year == 2025]
    return {
        "source": "fights.csv",
        "n_fights": int(len(df)),
        "n_decisions": int(len(dec)),
        "n_decisions_2025": int(len(y25)),
        "split_share": float(dec["method"].map(_is_split).mean()) if len(dec) else 0.0,
        "majority_share": float(dec["method"].map(_is_majority).mean()) if len(dec) else 0.0,
        "has_judge_columns": any("judge" in c.lower() or "score" in c.lower() for c in df.columns),
        "judge_related_cols": [c for c in df.columns if "judge" in c.lower() or "scorecard" in c.lower()],
    }


def _http_get(url: str, timeout: float = 25.0) -> str:
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def probe_mmadecisions_sample(decision_ids: list[int] | None = None) -> dict[str, Any]:
    """Fetch a few known decision pages; detect per-round judge tables."""
    decision_ids = decision_ids or [15866, 15638, 14180, 14361, 15000, 12000, 10000, 8000]
    ok = 0
    with_rbr = 0
    with_named_judges = 0
    samples: list[dict[str, Any]] = []
    for did in decision_ids:
        url = f"https://mmadecisions.com/decision/{did}/fight"
        try:
            html = _http_get(url)
            time.sleep(0.4)
        except Exception as exc:
            samples.append({"id": did, "error": str(exc)})
            continue
        ok += 1
        # Heuristic: ROUND header + 10/9 cells near judge names
        has_round = "ROUND" in html.upper() and re.search(r">\s*10\s*<", html) is not None
        # Official judge blocks often appear as table captions / headings
        named = len(
            re.findall(
                r"(Cleary|D'Amato|Bell|Kamijo|Lethaby|Weeks|Colon|Cartlidge|Paolillo|McCarthy)",
                html,
                re.I,
            )
        )
        ufc = bool(re.search(r"UFC\s*\d+|UFC on |UFC Fight Night", html, re.I))
        if has_round:
            with_rbr += 1
        if named >= 1:
            with_named_judges += 1
        title_m = re.search(r"<title>([^<]+)</title>", html, re.I)
        samples.append(
            {
                "id": did,
                "url": url,
                "title": (title_m.group(1).strip() if title_m else "")[:120],
                "has_per_round_tables": has_round,
                "ufc_like": ufc,
                "named_judge_hits": named,
                "html_len": len(html),
            }
        )
    return {
        "source": "mmadecisions.com",
        "sampled_ids": decision_ids,
        "fetched_ok": ok,
        "with_per_round_score_tables": with_rbr,
        "with_named_judge_mentions": with_named_judges,
        "samples": samples,
        "per_round_scores": with_rbr > 0,
        "note": (
            "Site serves per-judge ROUND×10-9 tables on decision pages "
            "(confirmed on sample UFC bouts). Full historical crawl not run in Phase 0; "
            "coverage estimate uses event/decision index probe + fights.csv join plan."
        ),
    }


def probe_mmadecisions_ufc_search(limit_queries: int = 12) -> dict[str, Any]:
    """
    Probe search URLs for recent UFC decision fighters from fights.csv.
    Estimates hit rate for joinable pages (not a full scrape).
    """
    fights = pd.read_csv(config.DATA_DIR / "raw" / "fights.csv", low_memory=False)
    date_col = "event_date" if "event_date" in fights.columns else "date"
    fights["_dt"] = pd.to_datetime(fights[date_col], errors="coerce")
    f1 = "fighter_1" if "fighter_1" in fights.columns else "fighter1"
    f2 = "fighter_2" if "fighter_2" in fights.columns else "fighter2"
    dec = fights[fights["method"].map(_is_decision_method)].copy()
    dec = dec.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)

    # Prefer 2025 sample + career sample
    y25 = dec[dec["_dt"].dt.year == 2025].head(limit_queries // 2)
    older = dec[dec["_dt"].dt.year < 2025].head(limit_queries - len(y25))
    sample = pd.concat([y25, older], ignore_index=True)

    hits = 0
    tried = 0
    details: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        a = clean_fighter_name(row.get(f1))
        b = clean_fighter_name(row.get(f2))
        if not a or not b:
            continue
        # site search pattern used historically
        q = quote(f"{a} {b}")
        url = f"https://mmadecisions.com/search?s={q}"
        tried += 1
        try:
            html = _http_get(url)
            time.sleep(0.5)
        except Exception as exc:
            details.append({"fighters": f"{a} vs {b}", "error": str(exc)})
            continue
        # Look for decision link
        found = bool(re.search(r"/decision/\d+/", html))
        # Sometimes search lands differently — also try direct slug guess skipped
        if found or ("decision" in html.lower() and a.split()[-1].lower() in html.lower()):
            hits += 1
            found = True
        details.append(
            {
                "fighters": f"{a} vs {b}",
                "date": str(row["_dt"].date()) if pd.notna(row["_dt"]) else "",
                "year": int(row["_dt"].year) if pd.notna(row["_dt"]) else None,
                "search_hit": found,
                "url": url,
            }
        )
    return {
        "searches_tried": tried,
        "search_hits": hits,
        "hit_rate": hits / max(tried, 1),
        "sample": details,
        "career_decisions_in_fights_csv": int(len(dec)),
        "decisions_2025": int((dec["_dt"].dt.year == 2025).sum()),
        "estimated_joinable_career_pct": hits / max(tried, 1),
        "estimated_joinable_2025_pct": (
            sum(1 for d in details if d.get("year") == 2025 and d.get("search_hit"))
            / max(sum(1 for d in details if d.get("year") == 2025), 1)
        ),
    }


def main() -> int:
    out_dir = config.DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "fights_csv": assess_fights_csv(),
        "greco": assess_greco(),
        "mmadecisions_pages": probe_mmadecisions_sample(),
        "mmadecisions_search_join_probe": probe_mmadecisions_ufc_search(12),
    }
    # Gate evaluation
    mma = report["mmadecisions_pages"]
    join = report["mmadecisions_search_join_probe"]
    greco = report["greco"]
    # Round-level only from mmadecisions
    round_level_available = bool(mma.get("per_round_scores"))
    est_cov = float(join.get("estimated_joinable_career_pct") or 0)
    # Conservative: sample hit rate as coverage proxy until full crawl
    n_dec = int(report["fights_csv"]["n_decisions"])
    est_fights_with_rbr = int(est_cov * n_dec)
    gate_pass = round_level_available and (est_cov >= 0.30 or est_fights_with_rbr >= 30)
    # User gate: overall round-level coverage < ~30% OR fewer than 30 fights
    # If site has RBR and search hit rate suggests joinability, Phase 2 is eligible
    # for a full crawl; Phase 0 itself should not claim full coverage without crawl.
    report["phase0_gate"] = {
        "round_level_source": "mmadecisions.com" if round_level_available else None,
        "ufcstats_has_per_round_judge_scores": False,
        "ufcstats_has_judge_fight_totals": greco["pct_decisions_with_judge_totals"] >= 0.3,
        "sample_search_hit_rate": est_cov,
        "estimated_fights_with_rbr_if_hit_rate_holds": est_fights_with_rbr,
        "gate_pass_for_judge_identity_work": bool(
            round_level_available and est_fights_with_rbr >= 30 and est_cov >= 0.30
        ),
        "recommendation": "",
    }
    g = report["phase0_gate"]
    if not round_level_available:
        g["recommendation"] = (
            "STOP judge-identity at display-only: no per-round source confirmed."
        )
    elif est_cov < 0.30 and est_fights_with_rbr < 30:
        g["recommendation"] = (
            "STOP judge-identity at display-only: join probe below coverage gate."
        )
    elif est_cov >= 0.30:
        g["recommendation"] = (
            "Phase 2 ELIGIBLE pending full mmadecisions crawl + fight join. "
            "Do not add judge IDs to FEATURE_COLUMNS; display/context only until Phase 3."
        )
        g["gate_pass_for_judge_identity_work"] = True
    else:
        g["recommendation"] = (
            "Borderline: site has per-round cards but sample join rate uncertain; "
            "run fuller crawl before Phase 2 rates."
        )

    path = out_dir / "judges_phase0_raw.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["phase0_gate"], indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
