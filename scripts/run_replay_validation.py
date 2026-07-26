"""Replay last 5 finished UFC cards with odds; write validation CSV + summary."""

from __future__ import annotations

import sys
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.project_paths import bootstrap
from src.replay import (
    _event_col,
    format_replay_summary,
    list_past_events,
    load_replay_features,
    replay_event,
    run_replay,
    write_replay_csv,
)
from src.predictor import FightPredictor


def events_with_odds(
    features: pd.DataFrame,
    *,
    min_cov: float = 0.4,
    min_odds_fights: int = 4,
    last_scan: int = 80,
) -> list[dict]:
    """Newest-last list of past events that have usable opening odds."""
    catalog = list_past_events(features, last=last_scan)
    ev = _event_col(features)
    out: list[dict] = []
    for e in catalog:
        sub = features[features[ev].astype(str) == str(e["event"])]
        if "f1_odds" not in sub.columns:
            cov, n_odds = 0.0, 0
        else:
            odds = pd.to_numeric(sub["f1_odds"], errors="coerce")
            n_odds = int(odds.notna().sum())
            cov = float(odds.notna().mean()) if len(sub) else 0.0
        if cov >= min_cov and n_odds >= min_odds_fights:
            out.append({**e, "odds_cov": cov, "n_odds": n_odds})
    return out


def main() -> int:
    bootstrap(entry_file=ROOT / "main.py")
    config.UFC_PROFILE = "paper"
    config.apply_profile_overrides()
    config.DYNAMIC_THRESHOLDS_ENABLED = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.log_paper_uncertainty_threshold_delta(force=True)

    print("Loading features…")
    features = load_replay_features()
    with_odds = events_with_odds(features, min_cov=0.5, min_odds_fights=5)
    if len(with_odds) < 5:
        # Relax slightly if history is sparse
        with_odds = events_with_odds(features, min_cov=0.4, min_odds_fights=4)
    if len(with_odds) < 5:
        with_odds = events_with_odds(features, min_cov=0.25, min_odds_fights=3)
    if not with_odds:
        print("No finished cards with odds found.")
        return 1

    selected = with_odds[-5:]
    print(f"Selected {len(selected)} finished cards with odds:")
    for e in selected:
        print(
            f"  {e['event_date']}  {e['event']}  "
            f"({e['fights']} fights, odds {e['odds_cov']:.0%} / {e['n_odds']})"
        )

    predictor = FightPredictor()
    per_event: list[dict] = []
    all_rows: list[dict] = []
    gate_notes: list[str] = []

    # Historical validation only — do not pull live book lines into past cards.
    import src.replay as replay_mod

    _orig_replay_event = replay_mod.replay_event

    def _replay_hist(event_name, **kwargs):
        kwargs.setdefault("narrative", False)
        result = _orig_replay_event(event_name, **kwargs)
        return result

    # Monkeypatch live odds merge inside replay_event path via predictor merge
    from src import predictor as pred_mod

    _orig_merge = pred_mod.merge_predictions_with_odds

    def _no_live_merge(predictions, odds=None, *, fetch_if_missing=True, force_refresh=False):
        return _orig_merge(
            predictions, odds=odds, fetch_if_missing=False, force_refresh=False
        )

    pred_mod.merge_predictions_with_odds = _no_live_merge

    try:
        for e in selected:
            name = e["event"]
            print(f"\nReplaying {name} …")
            result = _replay_hist(
                name,
                features=features,
                predictor=predictor,
                use_dynamic_thresholds=True,
                explain=False,
                narrative=False,
            )
            summary = result["summary"]
            rows = result["rows"]
            alerts = result.get("alerts") or {}

            # Strategy-rating / uncertainty effects from alerts
            singles = alerts.get("singles") or []
            skipped = alerts.get("skipped") or []
            sr_mults = [
                float(s.get("strategy_rating_mult") or 1.0)
                for s in singles
                if s.get("strategy_rating_mult") not in (None, "", 1, 1.0)
            ]
            unc_actions = Counter(
                str(s.get("uncertainty_action") or s.get("uncertainty_reason") or "")
                for s in (singles + skipped)
                if s.get("uncertainty_action") or s.get("uncertainty_reason")
            )
            unc_skips = sum(
                1
                for s in skipped
                if "uncertain" in str(s.get("skip_reason") or "").lower()
                or "disagreement" in str(s.get("skip_reason") or "").lower()
                or "interval" in str(s.get("skip_reason") or "").lower()
                or str(s.get("uncertainty_action") or "").lower() in ("skip", "block")
            )
            note_parts = []
            if sr_mults:
                note_parts.append(
                    f"strat_rating_mult avg={sum(sr_mults)/len(sr_mults):.2f} (n={len(sr_mults)})"
                )
            if unc_skips:
                note_parts.append(f"uncertainty_skips={unc_skips}")
            if unc_actions:
                top_u = ", ".join(f"{k}:{v}" for k, v in unc_actions.most_common(3) if k)
                if top_u:
                    note_parts.append(f"unc=[{top_u}]")
            gate_notes.append("; ".join(note_parts) if note_parts else "—")

            for r in rows:
                r["odds_cov"] = e["odds_cov"]
                r["gate_effects"] = "; ".join(note_parts) if note_parts else ""
            per_event.append(
                {**summary, "odds_cov": e["odds_cov"], "gate_effects": gate_notes[-1]}
            )
            all_rows.extend(rows)
    finally:
        pred_mod.merge_predictions_with_odds = _orig_merge

    # Aggregate
    scored = [r for r in all_rows if r.get("correct") is not None]
    n = len(scored)
    correct_n = sum(1 for r in scored if r.get("correct") == 1)
    bets = [r for r in all_rows if r.get("bet_taken")]
    pnls = [float(r["pnl"]) for r in bets if r.get("pnl") not in ("", None)]
    stakes = [float(r["stake"]) for r in bets if r.get("stake") not in ("", None)]
    skip_counts: Counter[str] = Counter()
    for r in all_rows:
        code = str(r.get("skip_reason") or "").strip()
        if code:
            skip_counts[code] += 1

    total_pnl = sum(pnls) if pnls else None
    total_stake = sum(stakes) if stakes else 0.0

    # Concise table
    print("\n" + "=" * 100)
    print("REPLAY VALIDATION — last 5 finished cards with odds")
    print("=" * 100)
    hdr = (
        f"{'Event':<42} {'Acc':>7} {'Bets':>5} {'PnL':>10} {'Top skips':<28} Effects"
    )
    print(hdr)
    print("-" * 100)
    for pe, note in zip(per_event, gate_notes):
        acc = pe.get("accuracy")
        acc_s = f"{acc:.0%}" if acc is not None else "n/a"
        pnl = pe.get("pnl")
        pnl_s = f"${pnl:+.1f}" if pnl is not None else "n/a"
        skips = pe.get("top_skips") or list((pe.get("skip_counts") or {}).items())[:2]
        if isinstance(skips, list) and skips and isinstance(skips[0], (list, tuple)):
            skip_s = ", ".join(f"{k}:{v}" for k, v in skips[:2])
        else:
            skip_s = "—"
        ev = str(pe.get("event") or "")[:42]
        print(
            f"{ev:<42} {acc_s:>7} {pe.get('bets_taken', 0):>5} {pnl_s:>10} "
            f"{skip_s:<28} {note}"
        )
    print("-" * 100)
    acc_all = (correct_n / n) if n else None
    acc_all_s = f"{acc_all:.1%}" if acc_all is not None else "n/a"
    pnl_all_s = f"${total_pnl:+.2f}" if total_pnl is not None else "n/a"
    roi = (total_pnl / total_stake) if total_pnl is not None and total_stake > 0 else None
    roi_s = f" | ROI {roi:.1%}" if roi is not None else ""
    print(
        f"{'TOTAL (' + str(len(per_event)) + ' cards)':<42} {acc_all_s:>7} "
        f"{len(bets):>5} {pnl_all_s:>10}  "
        f"({correct_n}/{n} picks){roi_s}"
    )
    if skip_counts:
        top = ", ".join(f"{k} x{v}" for k, v in skip_counts.most_common(6))
        print(f"Overall skips: {top}")
    print("=" * 100)

    stamp = datetime.now().strftime("%Y%m%d")
    out = ROOT / "reports" / f"replay_validation_{stamp}.csv"
    write_replay_csv(all_rows, out)

    summary_rows = []
    for pe, note in zip(per_event, gate_notes):
        summary_rows.append(
            {
                "event": pe.get("event"),
                "fights": pe.get("fights"),
                "accuracy": pe.get("accuracy"),
                "correct": pe.get("correct"),
                "scored": pe.get("scored"),
                "bets_taken": pe.get("bets_taken"),
                "bets_won": pe.get("bets_won"),
                "bets_lost": pe.get("bets_lost"),
                "pnl": pe.get("pnl"),
                "roi": pe.get("roi"),
                "odds_cov": pe.get("odds_cov"),
                "top_skips": "; ".join(
                    f"{k}:{v}" for k, v in (pe.get("top_skips") or [])[:5]
                ),
                "gate_effects": note,
            }
        )
    summary_path = ROOT / "reports" / f"replay_validation_{stamp}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nDetails CSV: {out}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
