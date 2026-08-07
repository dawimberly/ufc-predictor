"""Upset / miss autopsy on historical scored fights (correlational backtest).

Uses data/backtest_2025_results.csv — no model retrain.
Labels:
  - model_miss: predicted_winner != winner
  - market_upset: lower-implied (dog) side won when odds exist
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(
    os.environ.get("UPSET_AUTOPSY_SRC")
    or (ROOT / "data" / "backtest_2025_results_rescored.csv")
)
if not SRC.is_file():
    SRC = ROOT / "data" / "backtest_2025_results.csv"
OUT_DIR = ROOT / "data" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _american_to_dec(a: float) -> float | None:
    try:
        x = float(a)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or x == 0:
        return None
    if x > 0:
        return 1.0 + x / 100.0
    return 1.0 + 100.0 / abs(x)


def _implied_from_odds(row: pd.Series) -> tuple[float | None, float | None]:
    i1 = row.get("implied_prob_f1")
    i2 = row.get("implied_prob_f2")
    try:
        if pd.notna(i1) and pd.notna(i2) and float(i1) > 0 and float(i2) > 0:
            s = float(i1) + float(i2)
            if s > 0:
                return float(i1) / s, float(i2) / s
    except (TypeError, ValueError):
        pass
    d1 = _american_to_dec(row.get("f1_odds"))
    d2 = _american_to_dec(row.get("f2_odds"))
    if d1 is None or d2 is None or d1 <= 1 or d2 <= 1:
        return None, None
    raw1, raw2 = 1.0 / d1, 1.0 / d2
    s = raw1 + raw2
    return raw1 / s, raw2 / s


def main() -> None:
    df = pd.read_csv(SRC)
    df = df.copy()
    df["winner"] = df["winner"].astype(str)
    df["predicted_winner"] = df["predicted_winner"].astype(str)
    df["fighter_1"] = df["fighter_1"].astype(str)
    df["fighter_2"] = df["fighter_2"].astype(str)
    df["model_miss"] = (df["predicted_winner"] != df["winner"]).astype(int)
    # Prefer explicit correct column when present
    if "correct" in df.columns:
        df["model_miss"] = (pd.to_numeric(df["correct"], errors="coerce").fillna(1) == 0).astype(int)

    fav = []
    dog = []
    market_upset = []
    model_picked_fav = []
    pick_prob = []
    for _, r in df.iterrows():
        i1, i2 = _implied_from_odds(r)
        p1 = float(r.get("prob_f1_win") or 0.5)
        if i1 is None or i2 is None:
            fav.append(None)
            dog.append(None)
            market_upset.append(None)
            model_picked_fav.append(None)
            pick_prob.append(p1 if r["predicted_winner"] == r["fighter_1"] else 1.0 - p1)
            continue
        if i1 >= i2:
            f_name, d_name = r["fighter_1"], r["fighter_2"]
        else:
            f_name, d_name = r["fighter_2"], r["fighter_1"]
        fav.append(f_name)
        dog.append(d_name)
        market_upset.append(1 if r["winner"] == d_name else 0)
        model_picked_fav.append(1 if r["predicted_winner"] == f_name else 0)
        pick_prob.append(p1 if r["predicted_winner"] == r["fighter_1"] else 1.0 - p1)

    df["market_favorite"] = fav
    df["market_dog"] = dog
    df["market_upset"] = market_upset
    df["model_picked_favorite"] = model_picked_fav
    df["pick_prob"] = pick_prob

    with_odds = df[df["market_upset"].notna()].copy()
    summary = {
        "source": str(SRC),
        "n_fights": int(len(df)),
        "model_accuracy": float(1.0 - df["model_miss"].mean()),
        "model_miss_rate": float(df["model_miss"].mean()),
        "n_with_odds": int(len(with_odds)),
        "market_upset_rate": float(with_odds["market_upset"].mean()) if len(with_odds) else None,
        "model_picks_favorite_rate": float(with_odds["model_picked_favorite"].mean())
        if len(with_odds)
        else None,
        "when_model_picks_fav_accuracy": float(
            1.0
            - with_odds.loc[with_odds["model_picked_favorite"] == 1, "model_miss"].mean()
        )
        if len(with_odds) and (with_odds["model_picked_favorite"] == 1).any()
        else None,
        "when_model_picks_dog_accuracy": float(
            1.0
            - with_odds.loc[with_odds["model_picked_favorite"] == 0, "model_miss"].mean()
        )
        if len(with_odds) and (with_odds["model_picked_favorite"] == 0).any()
        else None,
        "upset_and_model_miss_rate": float(
            ((with_odds["market_upset"] == 1) & (with_odds["model_miss"] == 1)).mean()
        )
        if len(with_odds)
        else None,
        "upset_caught_by_model_rate": float(
            (
                (with_odds["market_upset"] == 1) & (with_odds["model_miss"] == 0)
            ).mean()
        )
        if len(with_odds)
        else None,
    }

    # Uncertainty correlation (when columns present)
    unc_lines: list[str] = []
    if "interval_width" in df.columns and "ensemble_disagreement" in df.columns:
        df["interval_width"] = pd.to_numeric(df["interval_width"], errors="coerce")
        df["ensemble_disagreement"] = pd.to_numeric(
            df["ensemble_disagreement"], errors="coerce"
        )
        hits_u = df[df["model_miss"] == 0]
        miss_u = df[df["model_miss"] == 1]
        summary["mean_width_hits"] = float(hits_u["interval_width"].mean())
        summary["mean_width_misses"] = float(miss_u["interval_width"].mean())
        summary["mean_disagree_hits"] = float(hits_u["ensemble_disagreement"].mean())
        summary["mean_disagree_misses"] = float(miss_u["ensemble_disagreement"].mean())
        # Width terciles miss rate
        try:
            df["width_tercile"] = pd.qcut(
                df["interval_width"], 3, labels=["narrow", "mid", "wide"], duplicates="drop"
            )
            wtab = (
                df.groupby("width_tercile", observed=False)
                .agg(n=("model_miss", "size"), miss_rate=("model_miss", "mean"))
                .reset_index()
            )
            unc_lines.append("### Miss rate by interval-width tercile")
            unc_lines.append(wtab.to_string(index=False))
        except ValueError:
            unc_lines.append("(width terciles unavailable — too little variance)")
        try:
            df["disagree_tercile"] = pd.qcut(
                df["ensemble_disagreement"].rank(method="first"),
                3,
                labels=["low", "mid", "high"],
            )
            dtab = (
                df.groupby("disagree_tercile", observed=False)
                .agg(n=("model_miss", "size"), miss_rate=("model_miss", "mean"))
                .reset_index()
            )
            unc_lines.append("### Miss rate by disagreement tercile")
            unc_lines.append(dtab.to_string(index=False))
        except ValueError:
            unc_lines.append("(disagreement terciles unavailable)")
        if len(with_odds):
            u = with_odds.copy()
            u["interval_width"] = pd.to_numeric(
                df.loc[u.index, "interval_width"], errors="coerce"
            )
            unc_lines.append(
                f"### Market upsets: mean width="
                f"{float(u.loc[u['market_upset']==1, 'interval_width'].mean()):.3f} vs "
                f"fav-won width="
                f"{float(u.loc[u['market_upset']==0, 'interval_width'].mean()):.3f}"
            )

    # Prob bucket miss rates
    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    df["prob_bucket"] = pd.cut(df["pick_prob"], bins=bins, right=False)
    by_prob = (
        df.groupby("prob_bucket", observed=False)
        .agg(n=("model_miss", "size"), miss_rate=("model_miss", "mean"), mean_prob=("pick_prob", "mean"))
        .reset_index()
    )

    # Feature diffs: hits vs misses (pick-side oriented where diffs are f1-centric:
    # flip sign when pick is f2)
    feature_cols = [
        c
        for c in (
            "elo_diff",
            "momentum_diff",
            "last5_winrate_diff",
            "win_rate_diff",
            "reach_diff",
            "age_diff",
            "striker_score_diff",
            "grappler_score_diff",
            "sig_strikes_per_min_diff",
            "td_defense_diff",
            "finish_rate_diff",
            "experience_diff",
            "sos_opp_win_rate_diff",
            "hv_short_notice_flag_diff",
            "hv_long_layoff_flag_diff",
            "first_fight_new_wc_flag_diff",
            "ko_losses_career_flag_diff",
            "finish_rate_l5_diff",
            "hv_control_clash",
            "wins_vs_better_record_l5_diff",
        )
        if c in df.columns
    ]
    oriented = df.copy()
    for c in feature_cols:
        oriented[c] = np.where(
            oriented["predicted_winner"] == oriented["fighter_1"],
            pd.to_numeric(oriented[c], errors="coerce"),
            -pd.to_numeric(oriented[c], errors="coerce"),
        )
    hits = oriented[oriented["model_miss"] == 0]
    misses = oriented[oriented["model_miss"] == 1]
    feat_rows = []
    for c in feature_cols:
        h = hits[c].mean(skipna=True)
        m = misses[c].mean(skipna=True)
        feat_rows.append(
            {
                "feature": c,
                "mean_on_hits": float(h) if pd.notna(h) else None,
                "mean_on_misses": float(m) if pd.notna(m) else None,
                "miss_minus_hit": float(m - h) if pd.notna(h) and pd.notna(m) else None,
            }
        )
    feat_df = pd.DataFrame(feat_rows).sort_values(
        "miss_minus_hit", key=lambda s: s.abs(), ascending=False
    )

    # Method mix on misses vs hits
    method_tab = None
    if "method" in df.columns:
        method_tab = (
            df.assign(method=df["method"].astype(str))
            .groupby(["model_miss", "method"], observed=False)
            .size()
            .unstack(fill_value=0)
        )

    # Market upset subset feature means
    upset_feat = None
    if len(with_odds):
        u = with_odds[with_odds["market_upset"] == 1]
        nu = with_odds[with_odds["market_upset"] == 0]
        rows = []
        for c in feature_cols:
            # orient to market favorite as f1-like: positive = fav stronger on feature
            def _orient(frame: pd.DataFrame) -> pd.Series:
                raw = pd.to_numeric(frame[c], errors="coerce")
                # if favorite is f2, flip
                fav_is_f1 = frame["market_favorite"] == frame["fighter_1"]
                return np.where(fav_is_f1, raw, -raw)

            um = pd.Series(_orient(u)).mean(skipna=True) if len(u) else np.nan
            nm = pd.Series(_orient(nu)).mean(skipna=True) if len(nu) else np.nan
            rows.append(
                {
                    "feature": c,
                    "mean_when_upset": float(um) if pd.notna(um) else None,
                    "mean_when_fav_won": float(nm) if pd.notna(nm) else None,
                    "upset_minus_fav": float(um - nm) if pd.notna(um) and pd.notna(nm) else None,
                }
            )
        upset_feat = pd.DataFrame(rows).sort_values(
            "upset_minus_fav", key=lambda s: s.abs(), ascending=False
        )

    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    summary_path = OUT_DIR / f"upset_autopsy_{stamp}.json"
    feat_path = OUT_DIR / f"upset_autopsy_feature_diffs_{stamp}.csv"
    prob_path = OUT_DIR / f"upset_autopsy_prob_buckets_{stamp}.csv"
    upset_path = OUT_DIR / f"upset_autopsy_market_feature_diffs_{stamp}.csv"
    md_path = OUT_DIR / f"upset_autopsy_{stamp}.md"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_prob.to_csv(prob_path, index=False)
    feat_df.to_csv(feat_path, index=False)
    if upset_feat is not None:
        upset_feat.to_csv(upset_path, index=False)

    lines = [
        f"# Upset / miss autopsy ({stamp})",
        "",
        f"Source: `{SRC}`",
        "",
        "Correlational backtest on scored historical fights — **not causal**.",
        "",
        "## Summary",
        f"- Fights: **{summary['n_fights']}**",
        f"- Model accuracy: **{summary['model_accuracy']:.1%}** (miss {summary['model_miss_rate']:.1%})",
        f"- With odds: **{summary['n_with_odds']}**",
        f"- Market upset rate (dog wins): **{(summary['market_upset_rate'] or 0):.1%}**",
        f"- Model picks favorite: **{(summary['model_picks_favorite_rate'] or 0):.1%}**",
        f"- Acc when model on favorite: **{(summary['when_model_picks_fav_accuracy'] or 0):.1%}**",
        f"- Acc when model on dog: **{(summary['when_model_picks_dog_accuracy'] or 0):.1%}**",
        f"- Share that are both market upset + model miss: **{(summary['upset_and_model_miss_rate'] or 0):.1%}**",
        f"- Share market upset but model correct (caught dog): **{(summary['upset_caught_by_model_rate'] or 0):.1%}**",
        "",
    ]
    if summary.get("mean_width_hits") is not None:
        lines += [
            "## Uncertainty (hits vs misses)",
            f"- Mean interval width: hits **{summary['mean_width_hits']:.3f}** / "
            f"misses **{summary['mean_width_misses']:.3f}**",
            f"- Mean disagreement: hits **{summary['mean_disagree_hits']:.4f}** / "
            f"misses **{summary['mean_disagree_misses']:.4f}**",
            "",
            *unc_lines,
            "",
        ]
    lines += [
        "## Miss rate by pick-prob bucket",
        by_prob.to_string(index=False),
        "",
        "## Largest feature diffs (miss - hit), pick-oriented",
        feat_df.head(12).to_string(index=False),
        "",
    ]
    if upset_feat is not None:
        lines += [
            "## Feature diffs on market upsets (favorite-oriented)",
            "Negative upset_minus_fav ~= favorite looked stronger on that feature but dog still won.",
            upset_feat.head(10).to_string(index=False),
            "",
        ]
    if method_tab is not None:
        lines += ["## Method counts (0=hit, 1=miss)", method_tab.to_string(), ""]
    lines += [
        "## Caveats",
        "- Association only; no causal identification.",
        "- Walk-forward imputer + frozen production ensemble when using rescored file.",
        "- Odds may be American; implied probs de-vigged when both sides present.",
        "",
        f"Artifacts: `{summary_path.name}`, `{feat_path.name}`, `{prob_path.name}`",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    text = md_path.read_text(encoding="utf-8")
    print(text[:4500].encode("ascii", "replace").decode("ascii"))
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
