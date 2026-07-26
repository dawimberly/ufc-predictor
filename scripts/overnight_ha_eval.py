"""Overnight HA evaluation: fixed-stake, live, autopsy, sleeves, tighter path, summary.

Fail-soft per job. Prefer ROI-on-stake + max DD over bankroll multiple.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.ha_backtest import (
    analyze_ticket_gate_mix,
    format_ha_backtest_summary,
    run_ha_walkforward_backtest,
    save_ha_backtest_reports,
)

STAMP = datetime.now().strftime("%Y%m%d")
AS_OF = datetime(2026, 7, 26)
BASELINE = Path("reports/ha_walkforward_leakage_fixed_20260726_summary.json")
BASELINE_TICKETS = Path("reports/ha_walkforward_leakage_fixed_20260726_tickets.csv")
REPORTS = Path("reports")
LOG: list[str] = []

logger = logging.getLogger("overnight_eval")


def _log(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)
    logger.info(msg)


def _safe(name: str, fn) -> Any:
    try:
        _log(f"=== START {name} ===")
        out = fn()
        _log(f"=== DONE {name} ===")
        return out
    except Exception as exc:
        tb = traceback.format_exc()
        _log(f"=== FAIL {name}: {exc} ===\n{tb}")
        return {"error": str(exc), "traceback": tb}


def _load_baseline_tickets() -> pd.DataFrame:
    if not BASELINE_TICKETS.is_file():
        raise FileNotFoundError(f"Missing baseline tickets: {BASELINE_TICKETS}")
    return pd.read_csv(BASELINE_TICKETS)


def _odds_bucket(odds: float) -> str:
    if not np.isfinite(odds):
        return "missing"
    if odds < 1.40:
        return "heavy_fav_<1.40"
    if odds < 1.80:
        return "fav_1.40_1.80"
    if odds < 2.20:
        return "near_even_1.80_2.20"
    return "dog_>=2.20"


def _edge_bucket(edge: float) -> str:
    if not np.isfinite(edge):
        return "missing"
    if edge < 0.10:
        return "edge_<0.10"
    if edge < 0.15:
        return "edge_0.10_0.15"
    if edge < 0.25:
        return "edge_0.15_0.25"
    return "edge_>=0.25"


def _prob_bucket(prob: float) -> str:
    if not np.isfinite(prob):
        return "missing"
    if prob < 0.70:
        return "prob_<0.70"
    if prob < 0.85:
        return "prob_0.70_0.85"
    if prob < 0.95:
        return "prob_0.85_0.95"
    return "prob_>=0.95"


def job_fixed_stake() -> dict[str, Any]:
    """Primary $5 fixed stake; derive $3 by stake-linear scaling (same tickets)."""
    report5 = run_ha_walkforward_backtest(
        bankroll_start=100.0,
        last_year=True,
        use_dynamic_thresholds=True,
        profile="paper",
        as_of=AS_OF,
        fixed_stake_usd=5.0,
    )
    paths5 = save_ha_backtest_reports(
        report5,
        stamp=STAMP,
        prefix="ha_wf_fixed_stake_5",
        baseline_path=BASELINE if BASELINE.is_file() else None,
        baseline_label="vs leakage-fixed compounding Paper (reference)",
        baseline_tickets_path=BASELINE_TICKETS if BASELINE_TICKETS.is_file() else None,
    )
    _log(format_ha_backtest_summary(report5))

    # $3 secondary: same selection, linear scale of stakes/pnl from $5 run
    scale = 3.0 / 5.0
    s5 = report5.get("summary") or {}
    tickets3 = []
    for t in report5.get("tickets") or []:
        t3 = dict(t)
        for k in ("stake", "pnl"):
            if t3.get(k) is not None:
                try:
                    t3[k] = float(t3[k]) * scale
                except (TypeError, ValueError):
                    pass
        tickets3.append(t3)
    report3 = {
        "summary": {
            **s5,
            "fixed_stake_usd": 3.0,
            "staking": "fixed_stake",
            "bankroll_start": 100.0,
            "bankroll_final": 100.0 + float(s5.get("total_pnl") or 0) * scale,
            "total_pnl": float(s5.get("total_pnl") or 0) * scale,
            "total_stake": float(s5.get("total_stake") or 0) * scale,
            "roi_pct": float(s5.get("total_pnl") or 0) * scale,
            "roi_on_stake_pct": s5.get("roi_on_stake_pct"),  # unchanged under linear scale
            "max_drawdown_usd": float(s5.get("max_drawdown_usd") or 0) * scale,
            "notes": list(s5.get("notes") or [])
            + ["$3 run derived by linear stake scale from $5 fixed-stake WF (identical tickets)."],
        },
        "monthly": report5.get("monthly"),
        "per_event": report5.get("per_event"),
        "tickets": tickets3,
        "generated_at": report5.get("generated_at"),
        "wf_train_meta": report5.get("wf_train_meta"),
    }
    # Recalc bankroll_final properly: start 100 + scaled pnl
    report3["summary"]["bankroll_final"] = 100.0 + float(report3["summary"]["total_pnl"])
    report3["summary"]["roi_pct"] = float(report3["summary"]["total_pnl"])  # $ start 100 → % = pnl
    paths3 = save_ha_backtest_reports(
        report3,
        stamp=STAMP,
        prefix="ha_wf_fixed_stake_3",
    )
    return {
        "fixed_5": report5.get("summary"),
        "fixed_3": report3.get("summary"),
        "paths_5": {k: str(v) for k, v in paths5.items()},
        "paths_3": {k: str(v) for k, v in paths3.items()},
    }


def job_live_profile() -> dict[str, Any]:
    report = run_ha_walkforward_backtest(
        bankroll_start=100.0,
        last_year=True,
        use_dynamic_thresholds=True,
        profile="live",
        as_of=AS_OF,
    )
    paths = save_ha_backtest_reports(
        report,
        stamp=STAMP,
        prefix="ha_wf_live_profile",
        baseline_path=BASELINE if BASELINE.is_file() else None,
        baseline_label="vs Paper leakage-fixed baseline",
        baseline_tickets_path=BASELINE_TICKETS if BASELINE_TICKETS.is_file() else None,
    )
    _log(format_ha_backtest_summary(report))
    return {"summary": report.get("summary"), "paths": {k: str(v) for k, v in paths.items()}}


def job_loss_autopsy() -> dict[str, Any]:
    tickets = _load_baseline_tickets()
    settled = tickets[tickets["status"].astype(str) == "settled"].copy()
    losses = settled[settled["won"].fillna(0).astype(int) == 0].copy()

    # Enrich weight class from features when fight_id available
    wc_map: dict[str, str] = {}
    try:
        from src.replay import load_replay_features

        feats = load_replay_features()
        fid = config.FIGHT_ID_COLUMN
        wc_col = "weight_class" if "weight_class" in feats.columns else None
        if wc_col and fid in feats.columns:
            for _, r in feats[[fid, wc_col]].dropna().iterrows():
                try:
                    key = str(int(float(r[fid])))
                except (TypeError, ValueError):
                    key = str(r[fid])
                wc_map[key] = str(r[wc_col])
    except Exception as exc:
        _log(f"weight-class enrich skipped: {exc}")

    rows = []
    for _, t in losses.iterrows():
        fid = t.get("fight_id")
        try:
            fid_key = str(int(float(fid))) if pd.notna(fid) else ""
        except (TypeError, ValueError):
            fid_key = str(fid or "")
        odds = float(t["odds"]) if pd.notna(t.get("odds")) else np.nan
        edge = float(t["edge"]) if pd.notna(t.get("edge")) else np.nan
        prob = float(t["prob"]) if pd.notna(t.get("prob")) else np.nan
        rows.append(
            {
                "date": t.get("event_date"),
                "event": t.get("event"),
                "bet_type": t.get("bet_type"),
                "legs": t.get("picks") or t.get("pick") or "",
                "n_legs": t.get("n_legs"),
                "model_prob": prob,
                "edge": edge,
                "odds": odds,
                "stake": t.get("stake"),
                "pnl": t.get("pnl"),
                "fight": t.get("fight"),
                "fight_id": fid_key,
                "weight_class": wc_map.get(fid_key, ""),
                "odds_bucket": _odds_bucket(odds),
                "edge_bucket": _edge_bucket(edge),
                "prob_bucket": _prob_bucket(prob),
                "uncertainty": "",  # not persisted on tickets CSV
                "skip_gate_notes": "",
            }
        )
    loss_df = pd.DataFrame(rows)
    csv_path = REPORTS / f"ha_loss_autopsy_{STAMP}.csv"
    loss_df.to_csv(csv_path, index=False)

    # Pattern summary
    patterns: list[str] = []
    if not loss_df.empty:
        by_type = loss_df["bet_type"].value_counts().to_dict()
        patterns.append(f"By type: {by_type}")
        patterns.append(
            f"Odds buckets: {loss_df['odds_bucket'].value_counts().to_dict()}"
        )
        patterns.append(
            f"Edge buckets: {loss_df['edge_bucket'].value_counts().to_dict()}"
        )
        patterns.append(
            f"Prob buckets: {loss_df['prob_bucket'].value_counts().to_dict()}"
        )
        if loss_df["weight_class"].astype(str).str.len().gt(0).any():
            patterns.append(
                f"Top weight classes: {loss_df['weight_class'].value_counts().head(8).to_dict()}"
            )
        n_parlay = int((loss_df["bet_type"].astype(str).str.contains("parlay")).sum())
        n_single = int((loss_df["bet_type"].astype(str) == "single").sum())
        patterns.append(f"Loss mix: singles={n_single}, parlays={n_parlay}")

    md_path = REPORTS / f"ha_loss_autopsy_{STAMP}.md"
    md = [
        f"# HA Loss Autopsy ({STAMP})",
        "",
        f"Source: `{BASELINE_TICKETS}`",
        f"Settled losses: **{len(loss_df)}** / {len(settled)} settled tickets",
        "",
        "## Patterns",
        *[f"- {p}" for p in patterns],
        "",
        "## Notes",
        "- Uncertainty / skip-gate fields were not stored on the WF tickets CSV; left blank.",
        "- Weight class joined from fight_features when fight_id matched.",
        "",
        f"CSV: `{csv_path}`",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {
        "n_losses": len(loss_df),
        "n_settled": len(settled),
        "patterns": patterns,
        "csv": str(csv_path),
        "md": str(md_path),
    }


def job_sleeve_eval() -> dict[str, Any]:
    tickets = _load_baseline_tickets()
    settled = tickets[tickets["status"].astype(str) == "settled"].copy()

    wc_map: dict[str, str] = {}
    try:
        from src.replay import load_replay_features

        feats = load_replay_features()
        fid = config.FIGHT_ID_COLUMN
        if "weight_class" in feats.columns and fid in feats.columns:
            for _, r in feats[[fid, "weight_class"]].dropna().iterrows():
                try:
                    key = str(int(float(r[fid])))
                except (TypeError, ValueError):
                    key = str(r[fid])
                wc_map[key] = str(r["weight_class"])
    except Exception as exc:
        _log(f"sleeve wc enrich skipped: {exc}")

    def _fid_key(v: Any) -> str:
        try:
            return str(int(float(v))) if pd.notna(v) else ""
        except (TypeError, ValueError):
            return str(v or "")

    settled["weight_class"] = settled["fight_id"].map(lambda x: wc_map.get(_fid_key(x), "unknown"))
    settled["odds_f"] = pd.to_numeric(settled["odds"], errors="coerce")
    settled["edge_f"] = pd.to_numeric(settled["edge"], errors="coerce")
    settled["prob_f"] = pd.to_numeric(settled["prob"], errors="coerce")
    settled["stake_f"] = pd.to_numeric(settled["stake"], errors="coerce")
    settled["pnl_f"] = pd.to_numeric(settled["pnl"], errors="coerce")
    settled["won_i"] = settled["won"].fillna(0).astype(int)
    settled["odds_bucket"] = settled["odds_f"].map(_odds_bucket)
    settled["edge_bucket"] = settled["edge_f"].map(_edge_bucket)
    settled["prob_bucket"] = settled["prob_f"].map(_prob_bucket)
    # Parlays lack per-leg prob/edge/odds often — mark
    settled.loc[settled["bet_type"].astype(str).str.contains("parlay"), "prob_bucket"] = (
        settled.loc[settled["bet_type"].astype(str).str.contains("parlay"), "prob_bucket"]
        .replace("missing", "parlay_no_leg_prob")
    )

    segments: list[dict[str, Any]] = []

    def _add_segment(dim: str, value: str, g: pd.DataFrame) -> None:
        n = len(g)
        if n == 0:
            return
        stake = float(g["stake_f"].sum())
        pnl = float(g["pnl_f"].sum())
        hit = float(g["won_i"].mean()) if n else None
        roi = (pnl / stake) if stake > 0 else None
        weak = bool(n >= 5 and ((hit is not None and hit < 0.55) or (roi is not None and roi < 0)))
        segments.append(
            {
                "dimension": dim,
                "segment": value,
                "n": n,
                "wins": int(g["won_i"].sum()),
                "hit_rate": hit,
                "stake": round(stake, 2),
                "pnl": round(pnl, 2),
                "roi_on_stake": roi,
                "weak_n5": weak,
            }
        )

    for bt, g in settled.groupby(settled["bet_type"].astype(str)):
        _add_segment("bet_type", bt, g)
    for col, dim in (
        ("prob_bucket", "model_prob"),
        ("edge_bucket", "edge"),
        ("odds_bucket", "odds"),
        ("weight_class", "weight_class"),
    ):
        for val, g in settled.groupby(settled[col].astype(str)):
            _add_segment(dim, val, g)

    seg_df = pd.DataFrame(segments).sort_values(["dimension", "n"], ascending=[True, False])
    out = REPORTS / f"ha_sleeve_eval_{STAMP}.csv"
    seg_df.to_csv(out, index=False)
    weak = seg_df[seg_df["weak_n5"] == True]  # noqa: E712
    return {
        "csv": str(out),
        "n_segments": len(seg_df),
        "weak_segments": weak.to_dict("records"),
    }


def job_tighter_path() -> dict[str, Any]:
    """Paper WF with tighter path-risk: parlay share ≤25%, flatter stakes, lower card risk."""
    import src.high_accuracy_strategy as ha

    paper_profile = getattr(config, "_PROFILE_PAPER", None)
    if not isinstance(paper_profile, dict):
        raise RuntimeError("config._PROFILE_PAPER missing; cannot tighten card risk")

    orig_paper = dict(ha._PAPER)
    orig_card_risk = float(paper_profile.get("max_card_risk_fraction") or 0.55)

    try:
        ha._PAPER["max_parlay_share"] = 0.25
        ha._PAPER["stake_power"] = 0.75  # flatter
        ha._PAPER["drawdown_soft_pct"] = 0.20
        ha._PAPER["drawdown_hard_pct"] = 0.35
        ha._PAPER["drawdown_soft_mult"] = 0.60
        ha._PAPER["drawdown_hard_mult"] = 0.35
        paper_profile["max_card_risk_fraction"] = min(orig_card_risk, 0.35)

        report = run_ha_walkforward_backtest(
            bankroll_start=100.0,
            last_year=True,
            use_dynamic_thresholds=True,
            profile="paper",
            as_of=AS_OF,
        )
        notes = list((report.get("summary") or {}).get("notes") or [])
        notes.append(
            "Tighter path-risk: max_parlay_share=25%, stake_power=0.75, "
            "max_card_risk_fraction≤35%, earlier DD soft/hard cuts."
        )
        if report.get("summary") is not None:
            report["summary"]["notes"] = notes
            report["summary"]["path_risk_mode"] = "tighter"

        paths = save_ha_backtest_reports(
            report,
            stamp=STAMP,
            prefix="ha_wf_tighter_path",
            baseline_path=BASELINE if BASELINE.is_file() else None,
            baseline_label="vs Paper leakage-fixed baseline",
            baseline_tickets_path=BASELINE_TICKETS if BASELINE_TICKETS.is_file() else None,
        )
        _log(format_ha_backtest_summary(report))
        return {"summary": report.get("summary"), "paths": {k: str(v) for k, v in paths.items()}}
    finally:
        ha._PAPER.clear()
        ha._PAPER.update(orig_paper)
        paper_profile["max_card_risk_fraction"] = orig_card_risk


def job_executive_summary(results: dict[str, Any]) -> str:
    base = {}
    if BASELINE.is_file():
        base = (json.loads(BASELINE.read_text(encoding="utf-8")).get("summary") or {})

    fixed = (results.get("fixed_stake") or {}).get("fixed_5") or {}
    fixed3 = (results.get("fixed_stake") or {}).get("fixed_3") or {}
    live = (results.get("live") or {}).get("summary") or {}
    tight = (results.get("tighter") or {}).get("summary") or {}
    autopsy = results.get("autopsy") or {}
    sleeve = results.get("sleeve") or {}

    def _m(s: dict[str, Any], key: str, fmt: str = ".1f") -> str:
        v = s.get(key)
        if v is None:
            return "n/a"
        if key == "hit_rate":
            return f"{100 * float(v):.1f}%"
        if "roi" in key or "drawdown_pct" in key or key == "roi_pct":
            return f"{float(v):{fmt}}%"
        if "bankroll" in key or "pnl" in key or key.endswith("_usd"):
            return f"${float(v):.2f}"
        return str(v)

    weak = sleeve.get("weak_segments") or []
    weak_lines = [
        f"- `{w.get('dimension')}={w.get('segment')}` n={w.get('n')} "
        f"hit={100 * float(w.get('hit_rate') or 0):.0f}% "
        f"ROI-stake={100 * float(w.get('roi_on_stake') or 0):.1f}%"
        for w in weak[:12]
    ] or ["- None flagged (n≥5 and hit<55% or ROI-stake<0)"]

    # Go / no-go heuristics
    trust_roi = float(fixed.get("roi_on_stake_pct") or base.get("roi_on_stake_pct") or 0)
    trust_hit = float(fixed.get("hit_rate") or base.get("hit_rate") or 0)
    live_roi = float(live.get("roi_on_stake_pct") or 0) if live else None
    go_notes = []
    if trust_roi >= 10 and trust_hit >= 0.70:
        go_notes.append(
            "**Conditional GO for Paper** — fixed-stake / leakage-fixed ROI-on-stake "
            "supports real edge, but max DD ~55%+ is too high for full bankroll compounding."
        )
    else:
        go_notes.append("**NO-GO for meaningful size** until ROI-on-stake stays ≥10% out of sample.")
    go_notes.append(
        "**NO-GO for Live full size** until Live-profile DD and ticket volume are acceptable; "
        "prefer Live gates with tiny fixed stakes if trading live."
    )
    go_notes.append(
        "**Parlays:** treat as optional sauce only — singles carry the edge; cap share ≤25%."
    )

    md = f"""# HA Overnight Evaluation ({STAMP})

## Trustworthy results

| Run | Final $ / PnL | ROI on stake | Hit | Max DD | Tickets |
|---|---:|---:|---:|---:|---:|
| **Leakage-fixed Paper (compounding)** | {_m(base,'bankroll_final')} | {_m(base,'roi_on_stake_pct')} | {_m(base,'hit_rate')} | {_m(base,'max_drawdown_pct')} | {base.get('n_tickets','n/a')} |
| **Fixed $5 / ticket** | PnL {_m(fixed,'total_pnl')} (end {_m(fixed,'bankroll_final')}) | {_m(fixed,'roi_on_stake_pct')} | {_m(fixed,'hit_rate')} | {_m(fixed,'max_drawdown_pct')} / {_m(fixed,'max_drawdown_usd')} | {fixed.get('n_tickets','n/a')} |
| Fixed $3 (scaled) | PnL {_m(fixed3,'total_pnl')} | {_m(fixed3,'roi_on_stake_pct')} | {_m(fixed3,'hit_rate')} | {_m(fixed3,'max_drawdown_usd')} | {fixed3.get('n_tickets','n/a')} |
| **Live profile** | {_m(live,'bankroll_final')} | {_m(live,'roi_on_stake_pct')} | {_m(live,'hit_rate')} | {_m(live,'max_drawdown_pct')} | {live.get('n_tickets','n/a')} |
| **Tighter path Paper** | {_m(tight,'bankroll_final')} | {_m(tight,'roi_on_stake_pct')} | {_m(tight,'hit_rate')} | {_m(tight,'max_drawdown_pct')} | {tight.get('n_tickets','n/a')} |

**What to trust:** ROI on stake + hit rate from **fixed-stake** and **leakage-fixed** walks.  
**What not to over-read:** Bankroll multiples under % compounding (path-dependent, stake scales with luck).

Window: `2025-07-26 → 2026-06-06`. Features: leakage-fixed schema (as-of Elo/SOS/Sherdog/Greco).

## Real edge estimate

- Prefer **ROI on stake ≈ {_m(fixed,'roi_on_stake_pct') if fixed else _m(base,'roi_on_stake_pct')}** (fixed $5) / **{_m(base,'roi_on_stake_pct')}** (compounding baseline) as the edge signal.
- Hit rate **{_m(fixed,'hit_rate') if fixed else _m(base,'hit_rate')}** with ~50–60 tickets is informative but still a modest sample — expect regression.
- Compounding bankroll (~{_m(base,'bankroll_final')} from $100) mixes edge with **stake inflation after wins**; do not treat +1800% bankroll as the edge.

## Main failure modes (loss autopsy)

- Losses: **{autopsy.get('n_losses', 'n/a')}** / {autopsy.get('n_settled', 'n/a')} settled
{chr(10).join(f"- {p}" for p in (autopsy.get('patterns') or [])[:8])}
- Detail: `{autopsy.get('csv', 'n/a')}` · `{autopsy.get('md', 'n/a')}`

## Weak sleeves (n≥5)

{chr(10).join(weak_lines)}

Full table: `{sleeve.get('csv', 'n/a')}`

## Recommended Paper vs Live

| Setting | Recommendation |
|---|---|
| **Paper** | Keep HA singles floors; use **fixed stake or capped card risk** (≤35%); **parlay share ≤25%**; flatter allocation |
| **Live** | Stricter gates already; only if Live WF ROI-on-stake stays healthy at tiny $ size — Live end bankroll {_m(live,'bankroll_final')}, ROI-stake {_m(live,'roi_on_stake_pct')}, DD {_m(live,'max_drawdown_pct')} |
| **Parlays** | Optional; never primary. Prefer high-confidence 2-legs only under the share cap |
| **Compounding %** | Avoid full Paper compounding on real money until max DD is controlled |

## Go / no-go (real cards)

{chr(10).join(f"- {g}" for g in go_notes)}

## Next 3 highest-value improvements

1. **Persist uncertainty + skip reasons on WF tickets** so autopsy/sleeves can use disagreement / interval width (currently blank).
2. **Fixed-stake or hard $ card caps in Paper production** — separates edge from compounding; target max DD well below 40% of bankroll.
3. **Parlay share hard-cap ≤25% + drop low-EV 2-legs** — singles (~90%+ hit in trusted run) carry the edge; parlays drive variance.

## Job log

{chr(10).join(f"- {line}" for line in LOG if line.startswith('=== '))}

## Artifacts

- Fixed $5: `reports/ha_wf_fixed_stake_5_{STAMP}.html`
- Fixed $3: `reports/ha_wf_fixed_stake_3_{STAMP}.html`
- Live: `reports/ha_wf_live_profile_{STAMP}.html`
- Tighter path: `reports/ha_wf_tighter_path_{STAMP}.html`
- Loss autopsy: `reports/ha_loss_autopsy_{STAMP}.csv` / `.md`
- Sleeves: `reports/ha_sleeve_eval_{STAMP}.csv`
"""
    path = REPORTS / f"ha_overnight_eval_{STAMP}.md"
    path.write_text(md, encoding="utf-8")
    _log(f"Executive summary -> {path}")
    return str(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    REPORTS.mkdir(parents=True, exist_ok=True)
    _log(f"Overnight HA eval stamp={STAMP} as_of={AS_OF.date()}")

    results: dict[str, Any] = {}
    # Fast jobs first (no WF retrain)
    results["autopsy"] = _safe("loss_autopsy", job_loss_autopsy)
    results["sleeve"] = _safe("sleeve_eval", job_sleeve_eval)
    # WF jobs (slow)
    results["fixed_stake"] = _safe("fixed_stake_wf", job_fixed_stake)
    results["live"] = _safe("live_profile_wf", job_live_profile)
    results["tighter"] = _safe("tighter_path_wf", job_tighter_path)
    summary_path = _safe("executive_summary", lambda: job_executive_summary(results))

    meta = {
        "stamp": STAMP,
        "as_of": str(AS_OF.date()),
        "results": {
            k: (v if not isinstance(v, dict) or "summary" not in v else {"summary": v.get("summary"), "paths": v.get("paths"), "error": v.get("error")})
            for k, v in results.items()
        },
        "executive_summary": summary_path,
        "log": LOG,
    }
    (REPORTS / f"ha_overnight_eval_{STAMP}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    _log("OVERNIGHT EVAL COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
