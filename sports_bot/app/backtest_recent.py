"""
Recent-card backtest against the UFC-Predictor model + history.

Uses C:\\UFC-Predictor (or UFC_PREDICTOR_ROOT) for fights/features/model.
Writes HTML under <ufc-root>/reports/backtest_YYYYMMDD.html
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_ufc_root() -> Path:
    import os

    env = os.getenv("UFC_PREDICTOR_ROOT", "").strip()
    candidates = [
        Path(env) if env else None,
        Path(r"C:\UFC-Predictor"),
        Path(__file__).resolve().parents[4] / "UFC-Predictor",
    ]
    for path in candidates:
        if path and path.is_dir() and (path / "config.py").is_file():
            return path.resolve()
    raise FileNotFoundError(
        "UFC-Predictor root not found. Set UFC_PREDICTOR_ROOT or install at C:\\UFC-Predictor"
    )


def _ensure_ufc_on_path(root: Path) -> None:
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _last_n_events(features: pd.DataFrame, n: int) -> list[str]:
    import config

    date_col = config.DATE_COLUMN
    event_col = "event_name" if "event_name" in features.columns else "event"
    work = features.dropna(subset=[date_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    if event_col not in work.columns:
        # synthesize event buckets by date
        work["_event_key"] = work[date_col].dt.strftime("%Y-%m-%d")
        event_col = "_event_key"
    # chronological unique events
    ordered = (
        work.groupby(event_col, sort=False)[date_col]
        .max()
        .sort_values()
        .index.tolist()
    )
    if n <= 0:
        return ordered
    return ordered[-n:]


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not mask.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "avg_pred": None, "emp_rate": None})
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "avg_pred": float(p[mask].mean()),
                "emp_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_recent_card_backtest(*, last: int = 5, ufc_root: Path | None = None) -> dict[str, Any]:
    """Score the last N completed event cards with the current model."""
    root = ufc_root or resolve_ufc_root()
    _ensure_ufc_on_path(root)

    from src.project_paths import bootstrap

    bootstrap(entry_file=root / "main.py")
    import config
    from src.backtester import evaluate_classification, simulate_value_bets
    from src.data_loader import load_fights, load_processed_features
    from src.feature_engineering import build_feature_matrix
    from src.predictor import FightPredictor

    try:
        features = load_processed_features()
    except Exception:
        features = pd.DataFrame()
    if features is None or features.empty:
        fights = load_fights()
        features = build_feature_matrix(fights, keep_unlabeled=False)

    if config.TARGET_COLUMN not in features.columns:
        raise ValueError(f"Missing target {config.TARGET_COLUMN}")

    labeled = features.dropna(subset=[config.TARGET_COLUMN]).copy()
    events = _last_n_events(labeled, last)
    if not events:
        raise ValueError("No dated events found in feature matrix.")

    event_col = "event_name" if "event_name" in labeled.columns else (
        "event" if "event" in labeled.columns else None
    )
    if event_col is None:
        labeled["_event_key"] = pd.to_datetime(labeled[config.DATE_COLUMN]).dt.strftime("%Y-%m-%d")
        event_col = "_event_key"

    subset = labeled[labeled[event_col].isin(events)].copy()
    predictor = FightPredictor()
    preds = predictor.predict_batch(subset, apply_style_bonus=False)

    y = preds[config.TARGET_COLUMN].astype(int).to_numpy()
    p = preds["prob_f1_win"].astype(float).to_numpy()
    classification = evaluate_classification(y, p)
    # Pick accuracy: predicted winner vs actual
    f1 = preds.get("fighter_1", preds.get("fighter1"))
    f2 = preds.get("fighter_2", preds.get("fighter2"))
    pick = preds.get("predicted_winner")
    if pick is None:
        pick = np.where(p >= 0.5, f1, f2)
    actual = np.where(y == 1, f1, f2)
    pick_acc = float((pd.Series(pick).astype(str).values == pd.Series(actual).astype(str).values).mean())

    trades, summary = simulate_value_bets(
        preds,
        min_edge=config.MIN_EDGE,
        initial_bankroll=config.INITIAL_BANKROLL,
        flat_stake=config.FLAT_STAKE,
    )
    cal = _calibration_table(y, p)

    f1_col = "fighter_1" if "fighter_1" in preds.columns else "fighter1"
    f2_col = "fighter_2" if "fighter_2" in preds.columns else "fighter2"
    per_event = []
    for ev in events:
        chunk = preds[preds[event_col] == ev]
        if chunk.empty:
            continue
        yy = chunk[config.TARGET_COLUMN].astype(int).to_numpy()
        pp = chunk["prob_f1_win"].astype(float).to_numpy()
        if "predicted_winner" in chunk.columns:
            pk = chunk["predicted_winner"].astype(str).to_numpy()
        else:
            pk = np.where(pp >= 0.5, chunk[f1_col].astype(str), chunk[f2_col].astype(str))
        act = np.where(yy == 1, chunk[f1_col].astype(str), chunk[f2_col].astype(str))
        metrics = evaluate_classification(yy, pp)
        per_event.append(
            {
                "event": str(ev),
                "fights": int(len(chunk)),
                "accuracy": float((pk == act).mean()),
                "log_loss": float(metrics.get("log_loss", float("nan"))),
                "avg_prob": float(np.mean(pp)),
            }
        )

    roi_pct = float(summary.get("roi_pct", 0) or 0)
    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ufc_root": str(root),
        "last_n": last,
        "events": [str(e) for e in events],
        "n_fights": int(len(preds)),
        "pick_accuracy": pick_acc,
        "classification": {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in classification.items()
        },
        "betting": {
            "roi": roi_pct / 100.0,
            "roi_pct": roi_pct,
            "pnl": float(summary.get("total_pnl", 0) or 0),
            "n_bets": int(summary.get("trades", 0) or 0),
            "hit_rate": float(summary.get("hit_rate", 0) or 0),
            "final_bankroll": float(summary.get("final_equity", 0) or 0),
            "raw": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in summary.items()
            },
        },
        "calibration": cal.to_dict(orient="records"),
        "per_event": per_event,
    }
    return report


def render_html_report(report: dict[str, Any]) -> str:
    cal_rows = "".join(
        f"<tr><td>{r['bin']}</td><td>{r['n']}</td>"
        f"<td>{'' if r['avg_pred'] is None else f'{r['avg_pred']:.3f}'}</td>"
        f"<td>{'' if r['emp_rate'] is None else f'{r['emp_rate']:.3f}'}</td></tr>"
        for r in report.get("calibration") or []
    )
    ev_rows = "".join(
        f"<tr><td>{e['event']}</td><td>{e['fights']}</td>"
        f"<td>{e['accuracy']:.1%}</td><td>{e['avg_prob']:.3f}</td></tr>"
        for e in report.get("per_event") or []
    )
    bet = report.get("betting") or {}
    clf = report.get("classification") or {}
    roi_pct = bet.get("roi_pct")
    if roi_pct is None:
        roi_pct = float(bet.get("roi") or 0) * 100.0
    hit = float(bet.get("hit_rate") or 0)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Backtest {report.get('generated_at')}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#0f172a;color:#e2e8f0}}
h1,h2{{color:#f8fafc}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #334155;padding:8px;text-align:left}} th{{background:#1e293b}}
.card{{background:#1e293b;padding:16px;border-radius:8px;margin:12px 0}}
.ok{{color:#34d399}} .bad{{color:#f87171}}
</style></head><body>
<h1>Recent-card backtest</h1>
<p>Generated {report.get('generated_at')} · last {report.get('last_n')} events · {report.get('n_fights')} fights</p>
<div class="card">
  <h2>Summary</h2>
  <ul>
    <li>Pick accuracy: <b>{report.get('pick_accuracy', 0):.1%}</b></li>
    <li>Log-loss: {clf.get('log_loss', '—')}</li>
    <li>Brier: {clf.get('brier_score', clf.get('brier', '—'))}</li>
    <li>Betting ROI: <b>{float(roi_pct):+.1f}%</b> · PnL {float(bet.get('pnl') or 0):+.2f} · bets {bet.get('n_bets', 0)} · hit {hit:.1%}</li>
  </ul>
</div>
<h2>Per event</h2>
<table><tr><th>Event</th><th>Fights</th><th>Accuracy</th><th>Avg P(f1)</th></tr>{ev_rows}</table>
<h2>Edge / probability calibration</h2>
<table><tr><th>Bin</th><th>N</th><th>Avg pred</th><th>Empirical</th></tr>{cal_rows}</table>
</body></html>"""


def save_report(report: dict[str, Any], *, reports_dir: Path | None = None) -> Path:
    root = Path(report["ufc_root"])
    out_dir = reports_dir or (root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    html_path = out_dir / f"backtest_{stamp}.html"
    json_path = out_dir / f"backtest_{stamp}.json"
    html_path.write_text(render_html_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return html_path
