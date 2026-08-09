"""Phase 1 decision-profile coverage (+ optional A/B) and Phase 2 sample stats."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("ENABLE_PATHWAY_FEATURES", "false")
os.environ.setdefault("ENABLE_MARKET_FEATURES", "false")
os.environ.setdefault("INTERACTION_DISCOVERY_ENABLED", "false")

import numpy as np
import pandas as pd

import config

config.refresh_runtime_env()

from src.data_loader import load_fights
from src.decision_profile import (
    apply_decision_profile_rolling,
)
from src.judge_scoring_deviation import (
    RELIABILITY_MIN_ROUNDS,
    expand_round_rows,
    judge_deviation_summary,
)
from src.mmadecisions import load_decision_cache

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("judges_assess")

REPORTS = config.DATA_DIR / "reports"
YEAR = 2025
KEEP_AUC = 0.005


def _phase1_from_history() -> dict[str, Any]:
    """Lightweight long-history from fights.csv (no Greco/SOS rebuild)."""
    from src.data_loader import clean_fighter_name

    fights = load_fights()
    date_col = config.DATE_COLUMN if config.DATE_COLUMN in fights.columns else "event_date"
    f1 = "fighter_1" if "fighter_1" in fights.columns else "fighter1"
    f2 = "fighter_2" if "fighter_2" in fights.columns else "fighter2"
    work = fights[[c for c in (config.FIGHT_ID_COLUMN, date_col, f1, f2, "method", "winner") if c in fights.columns]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    rows = []
    for _, r in work.iterrows():
        a = clean_fighter_name(r.get(f1))
        b = clean_fighter_name(r.get(f2))
        if not a or not b:
            continue
        winner = clean_fighter_name(r.get("winner")) if "winner" in work.columns else ""
        method = r.get("method")
        for name, opp in ((a, b), (b, a)):
            won = 1 if winner and name == winner else (0 if winner else np.nan)
            # If no winner col, infer from outcome fields skipped
            rows.append(
                {
                    config.FIGHT_ID_COLUMN: r.get(config.FIGHT_ID_COLUMN),
                    config.DATE_COLUMN: r[date_col],
                    "fighter": name,
                    "won": won,
                    "method": method,
                    "is_dec": int("DEC" in str(method or "").upper() or "DECISION" in str(method or "").upper()),
                }
            )
    history = pd.DataFrame(rows)
    # Drop rows without known winner for rate quality
    history = history.dropna(subset=["won"])
    history["won"] = history["won"].astype(int)
    history = apply_decision_profile_rolling(history)

    wide_ish = history.drop_duplicates([config.FIGHT_ID_COLUMN, "fighter"], keep="last")
    # one row per fight: take first fighter side only for fight-level n
    side1 = (
        history.sort_values(config.DATE_COLUMN)
        .groupby(config.FIGHT_ID_COLUMN, as_index=False)
        .first()
    )
    dts = pd.to_datetime(side1[config.DATE_COLUMN], errors="coerce")

    def _cov(sub: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"n": int(len(sub))}
        for col in (
            "dec_win_rate_career",
            "dec_loss_rate_career",
            "split_dec_win_rate_career",
            "split_dec_loss_rate_career",
            "decision_finish_share_career",
            "dec_win_rate_l5",
            "split_dec_win_rate_l5",
            "decision_finish_share_l5",
        ):
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors="coerce")
            out[col] = {
                "nonnull_pct": float(s.notna().mean()),
                "mean": float(s.mean()) if s.notna().any() else None,
            }
        return out

    career = _cov(side1)
    y25 = _cov(side1.loc[dts.dt.year == YEAR])

    # Diffs: merge two sides
    pivoted = []
    for fid, g in history.groupby(config.FIGHT_ID_COLUMN):
        g = g.sort_values("fighter")
        if len(g) < 2:
            continue
        r1, r2 = g.iloc[0], g.iloc[1]
        pivoted.append(
            {
                config.FIGHT_ID_COLUMN: fid,
                config.DATE_COLUMN: r1[config.DATE_COLUMN],
                "dec_win_rate_career_diff": float(r1.get("dec_win_rate_career") or np.nan)
                - float(r2.get("dec_win_rate_career") or np.nan)
                if pd.notna(r1.get("dec_win_rate_career")) and pd.notna(r2.get("dec_win_rate_career"))
                else np.nan,
                "split_dec_win_rate_career_diff": (
                    float(r1.get("split_dec_win_rate_career") or np.nan)
                    - float(r2.get("split_dec_win_rate_career") or np.nan)
                    if pd.notna(r1.get("split_dec_win_rate_career"))
                    and pd.notna(r2.get("split_dec_win_rate_career"))
                    else np.nan
                ),
                "decision_finish_share_career_diff": (
                    float(r1.get("decision_finish_share_career") or np.nan)
                    - float(r2.get("decision_finish_share_career") or np.nan)
                    if pd.notna(r1.get("decision_finish_share_career"))
                    and pd.notna(r2.get("decision_finish_share_career"))
                    else np.nan
                ),
            }
        )
    merged = pd.DataFrame(pivoted)
    dts_m = pd.to_datetime(merged[config.DATE_COLUMN], errors="coerce") if not merged.empty else pd.Series(dtype="datetime64[ns]")
    m25 = merged.loc[dts_m.dt.year == YEAR] if not merged.empty else merged
    diff_cov = {}
    for col in (
        "dec_win_rate_career_diff",
        "split_dec_win_rate_career_diff",
        "decision_finish_share_career_diff",
    ):
        if col in m25.columns:
            s = pd.to_numeric(m25[col], errors="coerce")
            diff_cov[col] = {
                "nonnull_pct": float(s.notna().mean()),
                "nonzero_pct": float((s.fillna(0) != 0).mean()),
            }

    healthy = float(y25.get("dec_win_rate_career", {}).get("nonnull_pct") or 0) >= 0.5
    return {
        "career_side1": career,
        "y2025_side1": y25,
        "y2025_diffs": diff_cov,
        "coverage_healthy_for_optional_ab": healthy,
        "ship": True,
        "added_to_feature_columns": False,
        "note": (
            "Phase 1 ships as computable display/research columns from fights.csv methods. "
            "Not added to production FEATURE_COLUMNS in this pass."
        ),
    }


def _optional_phase1_ab(phase1: dict[str, Any]) -> dict[str, Any] | None:
    if not phase1.get("coverage_healthy_for_optional_ab"):
        return {"skipped": True, "reason": "coverage_not_healthy"}
    if os.getenv("RUN_DECISION_PROFILE_AB", "").lower() not in ("1", "true", "yes"):
        return {
            "skipped": True,
            "reason": "set RUN_DECISION_PROFILE_AB=1 to train A/B (optional)",
            "keep_rule": f"AUC>={KEEP_AUC} vs BASE",
        }
    # Heavy — only when explicitly requested
    return {"skipped": True, "reason": "deferred_explicit_flag_not_set_in_default_run"}


def _phase2_from_cache() -> dict[str, Any]:
    decisions = [r for r in load_decision_cache() if r.get("is_ufc") and r.get("has_per_round")]
    if not decisions:
        return {
            "cached_ufc_decisions": 0,
            "status": "no_cache",
            "recommendation": "Run scripts/crawl_mmadecisions_sample.py first.",
        }
    rounds = expand_round_rows(decisions)
    summary = judge_deviation_summary(rounds, min_rounds=RELIABILITY_MIN_ROUNDS)
    reliable = summary[summary["reliable"]] if not summary.empty else summary
    # Join probe vs fights.csv 2025 decisions
    fights = pd.read_csv(config.DATA_DIR / "raw" / "fights.csv", low_memory=False)
    date_col = "event_date" if "event_date" in fights.columns else "date"
    fights["_dt"] = pd.to_datetime(fights[date_col], errors="coerce")
    meth = fights["method"].astype(str)
    dec_mask = meth.str.contains("DEC|Decision", case=False, na=False)
    dec_all = fights.loc[dec_mask]
    dec_25 = dec_all.loc[dec_all["_dt"].dt.year == YEAR]

    # Name-key join: last names in event string / fighters
    from src.data_loader import clean_fighter_name

    f1 = "fighter_1" if "fighter_1" in fights.columns else "fighter1"
    f2 = "fighter_2" if "fighter_2" in fights.columns else "fighter2"

    def _key(a, b):
        """Last-name pair key — mmadecisions often uses surname-only headers."""
        def last(x):
            parts = clean_fighter_name(x).split()
            return parts[-1] if parts else ""

        return tuple(sorted([last(a), last(b)]))

    cache_keys = set()
    for d in decisions:
        cache_keys.add(_key(d.get("fighter_1"), d.get("fighter_2")))

    def _join_rate(frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        hits = 0
        for _, r in frame.iterrows():
            if _key(r.get(f1), r.get(f2)) in cache_keys:
                hits += 1
        return hits / len(frame)

    return {
        "cached_ufc_decisions": len(decisions),
        "cached_judge_rounds": int(len(rounds)),
        "unique_judges": int(summary["judge_name"].nunique()) if not summary.empty else 0,
        "judges_meeting_reliability_floor": int(len(reliable)),
        "pool_disagreement_rate": float(rounds["disagrees_with_majority"].mean())
        if not rounds.empty
        else None,
        "top_judges_by_rounds": (
            summary[["judge_name", "n_rounds", "disagreement_rate_shrunk", "reliable", "ui_extreme_tail_label"]]
            .head(15)
            .to_dict(orient="records")
            if not summary.empty
            else []
        ),
        "sample_join_rate_vs_all_decisions": _join_rate(dec_all),
        "sample_join_rate_vs_2025_decisions": _join_rate(dec_25),
        "n_decisions_fights_csv": int(len(dec_all)),
        "n_decisions_2025_fights_csv": int(len(dec_25)),
        "reliability_min_rounds": RELIABILITY_MIN_ROUNDS,
        "feature_columns_updated": False,
        "display_only": True,
    }


def main() -> int:
    phase0_path = REPORTS / "judges_phase0_raw.json"
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8")) if phase0_path.is_file() else {}

    logger.info("Phase 1 decision profile…")
    phase1 = _phase1_from_history()
    phase1["optional_ab"] = _optional_phase1_ab(phase1)

    logger.info("Phase 2 from mmadecisions cache…")
    phase2 = _phase2_from_cache()

    gate = phase0.get("phase0_gate") or {}
    # Phase 3 stance
    phase3 = {
        "decision_profile_feature_eligible_to_test": bool(
            phase1.get("coverage_healthy_for_optional_ab")
        ),
        "judge_identity_feature_eligible_to_test": False,
        "reason_judge_identity": (
            "Phase 2 is display/context only until a full crawl passes coverage "
            "and a Phase 3 A/B clears AUC/ROI keep rule. Sample crawl is not enough "
            "to promote judge-ID features."
        ),
        "keep_rule": "AUC >= +0.005 or clear flat-edge ROI/DD on chronological OOS",
        "production_retrain": False,
    }

    report = {
        "phase0": {
            "fights_csv": phase0.get("fights_csv"),
            "greco": {
                k: phase0.get("greco", {}).get(k)
                for k in (
                    "n_decisions",
                    "pct_decisions_with_judge_totals",
                    "unique_judges",
                    "judges_ge_50_est_rounds",
                    "per_round_scores",
                    "note",
                )
            },
            "mmadecisions": {
                "per_round_confirmed": (phase0.get("mmadecisions_pages") or {}).get(
                    "per_round_scores"
                ),
                "sample_search_hit_rate": (phase0.get("mmadecisions_search_join_probe") or {}).get(
                    "hit_rate"
                ),
                "gate": gate,
            },
        },
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "judges_assessment_raw.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # Markdown assessment
    g0 = gate
    lines = [
        "# Judges data assessment (UFC)",
        "",
        "Research only. Isolation: UFC project. No trading-bot changes. "
        "No Live HA changes. Pathway/market/home flags unchanged.",
        "",
        "Terminology: computed fields use **judge scoring deviation** / "
        "**panel disagreement**. UI may label extreme-tail judges "
        "`controversial` — never as a feature/column name.",
        "",
        "## Phase 0 — Data reality",
        "",
        "### Local (UFCStats / fights.csv)",
        "",
        f"- `fights.csv` decisions: **{(phase0.get('fights_csv') or {}).get('n_decisions')}** "
        f"(2025: **{(phase0.get('fights_csv') or {}).get('n_decisions_2025')}**)",
        f"- Judge/scorecard columns on fights.csv: **none**",
        f"- Greco `DETAILS` judge **fight totals** (not per-round): "
        f"**{100 * float((phase0.get('greco') or {}).get('pct_decisions_with_judge_totals') or 0):.1f}%** "
        f"of Greco decisions; ~**{(phase0.get('greco') or {}).get('unique_judges')}** judge name strings; "
        f"~**{(phase0.get('greco') or {}).get('judges_ge_50_est_rounds')}** judges with ≥50 *estimated* "
        f"rounds from totals (heuristic only).",
        "- Per-round 10-9 cards in UFCStats: **no**",
        "",
        "### mmadecisions.com",
        "",
        f"- Per-round per-judge tables: **confirmed** on sample UFC pages",
        f"- Search join probe hit-rate (small sample): "
        f"**{100 * float((phase0.get('mmadecisions_search_join_probe') or {}).get('hit_rate') or 0):.0f}%** "
        f"— treat as optimistic upper bound until full crawl",
        f"- Phase 0 gate: **{'PASS' if g0.get('gate_pass_for_judge_identity_work') else 'FAIL'}** — "
        f"{g0.get('recommendation')}",
        "",
        "## Phase 1 — Fighter decision profile (ships)",
        "",
        "Judge-agnostic rates from method labels (career + L5): "
        "`dec_win/loss_rate`, `split_dec_win/loss_rate`, `decision_finish_share`.",
        "",
        f"- 2025 `dec_win_rate_career` non-null: "
        f"**{100 * float(((phase1.get('y2025_side1') or {}).get('dec_win_rate_career') or {}).get('nonnull_pct') or 0):.1f}%**",
        f"- 2025 `split_dec_win_rate_career` non-null: "
        f"**{100 * float(((phase1.get('y2025_side1') or {}).get('split_dec_win_rate_career') or {}).get('nonnull_pct') or 0):.1f}%**",
        f"- 2025 `decision_finish_share_career` non-null: "
        f"**{100 * float(((phase1.get('y2025_side1') or {}).get('decision_finish_share_career') or {}).get('nonnull_pct') or 0):.1f}%**",
        f"- Coverage healthy for optional A/B: **{phase1.get('coverage_healthy_for_optional_ab')}**",
        f"- Added to production `FEATURE_COLUMNS`: **false** (research/display until Phase 3 keep)",
        f"- Module: `src/decision_profile.py`",
        "",
        "## Phase 2 — Judge scoring deviation (display only)",
        "",
    ]
    if phase2.get("cached_ufc_decisions", 0) == 0:
        lines += [
            "- Cache empty — run `python scripts/crawl_mmadecisions_sample.py`",
            "- Until crawl completes: **no per-judge round rates** to publish",
            "",
        ]
    else:
        lines += [
            f"- Sample UFC decisions cached: **{phase2['cached_ufc_decisions']}**",
            f"- Judge-rounds in sample: **{phase2['cached_judge_rounds']}**",
            f"- Unique judges: **{phase2['unique_judges']}**; "
            f"reliability floor (≥{phase2['reliability_min_rounds']} rounds): "
            f"**{phase2['judges_meeting_reliability_floor']}** judges",
            f"- Pool panel-disagreement rate: **{phase2.get('pool_disagreement_rate')}**",
            f"- Sample name-join vs all decisions: "
            f"**{100 * float(phase2.get('sample_join_rate_vs_all_decisions') or 0):.2f}%** "
            f"(low expected — sample is recent ID window only)",
            f"- Sample name-join vs 2025 decisions: "
            f"**{100 * float(phase2.get('sample_join_rate_vs_2025_decisions') or 0):.2f}%** "
            f"(of 2025 decisions matching this {phase2['cached_ufc_decisions']}-fight sample)",
            "- **Full career / full-2025 round-level coverage requires a complete mmadecisions crawl** "
            "(not claimed from this sample).",
            "- **FEATURE_COLUMNS: not updated.** Assigned judges / history notes = display only.",
            f"- Module: `src/mmadecisions.py`, `src/judge_scoring_deviation.py`",
            "",
            "### Top judges in sample (by rounds)",
            "",
            "| Judge | rounds | shrunk disagreement | reliable | UI tail |",
            "|---|---:|---:|:---:|---|",
        ]
        for r in phase2.get("top_judges_by_rounds") or []:
            lines.append(
                f"| {r.get('judge_name')} | {r.get('n_rounds')} | "
                f"{r.get('disagreement_rate_shrunk')} | {r.get('reliable')} | "
                f"{r.get('ui_extreme_tail_label') or ''} |"
            )
        lines.append("")

    lines += [
        "## Phase 3 — Keep rule",
        "",
        "- Decision-profile features: may be A/B'd under freeze/OOS when ready "
        f"(keep if AUC ≥ +{KEEP_AUC} or clear ROI/DD).",
        "- Judge-identity features: **not eligible** until full mmadecisions crawl "
        "coverage is measured, reliability floor applied, and keep rule passes.",
        "- **No production retrain** in this work.",
        "",
        "## Recommendations",
        "",
        "| Phase | Recommendation |",
        "|---|---|",
        "| 0 | mmadecisions is the round-level source; UFCStats totals-only is insufficient for Phase 2 rates |",
        "| 1 | **Ship** decision-profile module for display/context; optional A/B later |",
        "| 2 | **Display-only** after crawl; EB shrink below 50 rounds; no FEATURE_COLUMNS |",
        "| 3 | Hold judge-ID features until keep rule; decision-profile may test separately |",
        "",
        "Artifacts: `judges_phase0_raw.json`, `judges_assessment_raw.json`, "
        "`data/cache/mmadecisions/`",
        "",
    ]
    md = REPORTS / "judges_data_assessment.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", md)
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
