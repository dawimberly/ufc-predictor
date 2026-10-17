"""Console and file reports for backtests."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _mode_label(mode: str, trainer_identity: str = "") -> str:
    mode = str(mode or "frozen_model")
    if mode == "walk_forward_simplified":
        identity = trainer_identity or "simplified_lgbm_isotonic"
        return (
            f"walk_forward_simplified ({identity}; "
            "not production ensemble / discovery)"
        )
    if mode == "walk_forward":
        return "walk_forward (production-equivalent refit on earlier fights)"
    return "frozen_model (production weights; may have seen scored fights)"


def _fmt_pct_or_na(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.1%}" if abs(float(value)) <= 1.5 else f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_roi_or_na(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):>7.1f}%"
    except (TypeError, ValueError):
        return "    N/A"


def print_backtest_summary(result) -> None:
    m = result.overall_metrics
    mode = str(getattr(result, "evaluation_mode", "") or "frozen_model")
    trainer = str(getattr(result, "trainer_identity", "") or "")
    eligibility = getattr(result, "as_of_eligibility", None) or {}
    n_total = int(eligibility.get("n_total", m.get("n_fights", 0) or 0))
    n_eligible = int(eligibility.get("n_as_of_eligible", getattr(result, "fights_with_odds", 0) or 0))

    print(f"\n  UFC BETTING BOT — BACKTEST {result.target_year}")
    print("  " + "=" * 72)
    print(f"  Evaluation mode  {_mode_label(mode, trainer)}")
    if trainer:
        print(f"  Trainer          {trainer}")
    print(f"  Fights scored     {n_total}")
    print(f"  Events            {len(result.per_event)}")
    print(f"  Accuracy          {m.get('accuracy', 0):.1%}")
    print(f"  Log loss          {m.get('log_loss', 0):.3f}")
    print(f"  Brier score       {m.get('brier_score', 0):.3f}")
    print(f"  ROC AUC           {m.get('roc_auc', float('nan')):.3f}")
    print(f"  As-of eligible    {n_eligible} / {n_total}")
    print(
        "  Exclusions        "
        f"closing_not_as_of={int(eligibility.get('n_closing_not_as_of', 0))}  "
        f"unproven={int(eligibility.get('n_unproven', 0))}  "
        f"missing={int(eligibility.get('n_missing', 0))}  "
        f"other={int(eligibility.get('n_other', 0))}"
    )
    print(
        "  As-of rule        opening quote timestamp ≤ event cutoff "
        "(labels alone are unproven)"
    )

    if n_eligible == 0:
        print("\n  STAKING / ROI")
        print("  " + "-" * 72)
        print("  N/A — zero as-of-eligible fights; no pricing performance claimed.")
        print(f"\n  Reports -> {result.report_dir}")
        print()
        return

    print("\n  SEGMENT ACCURACY")
    print("  " + "-" * 40)
    for seg, sm in result.segment_metrics.items():
        print(
            f"  {seg:<14} {int(sm.get('n_fights', 0)):>4} fights  "
            f"acc {sm.get('accuracy', 0):.1%}"
        )

    staking = getattr(result, "staking_modes", pd.DataFrame())
    if isinstance(staking, pd.DataFrame) and not staking.empty:
        print("\n  STAKING MODES (min edge from settings)")
        print("  " + "-" * 72)
        for _, row in staking.iterrows():
            if row.get("performance_available") is False or int(row.get("trades", 0) or 0) == 0:
                print(
                    f"  {str(row.get('staking', '')):<14} "
                    f"trades {int(row.get('trades', 0) or 0):>4}  "
                    f"hit N/A  ROI N/A  "
                    f"(as-of eligible present; no bets cleared edge/stakes)"
                )
                continue
            warn = ""
            if row.get("roi_warning"):
                warn = "  <<< check"
                logger.warning("%s: %s", row.get("staking"), row["roi_warning"])
            mc_dd = row.get("mc_expected_max_drawdown_pct")
            mc_var = row.get("mc_var_max_drawdown_pct")
            mc_ruin = row.get("mc_ruin_probability")
            mc_extra = ""
            if pd.notna(mc_dd):
                mc_extra = (
                    f"  MC expDD {float(mc_dd):.1f}%"
                    f"  VaR {float(mc_var):.1f}%"
                    f"  ruin {float(mc_ruin):.1%}"
                )
            print(
                f"  {str(row.get('staking', '')):<14} "
                f"trades {int(row.get('trades', 0)):>4}  "
                f"hit {_fmt_pct_or_na(row.get('hit_rate', 0))}  "
                f"ROI {_fmt_roi_or_na(row.get('roi_pct', 0))}  "
                f"maxDD {row.get('max_drawdown_pct', 0):.1f}%  "
                f"winStk {int(row.get('max_win_streak', 0))}{warn}{mc_extra}"
            )

    mc = getattr(result, "monte_carlo", None)
    if mc is not None and getattr(mc, "staking_summaries", None):
        print("\n  MONTE CARLO RISK ({:,} sims, {} outcomes)".format(
            mc.n_simulations, mc.outcome_mode
        ))
        print("  " + "-" * 72)
        for mode_name, stats in mc.staking_summaries.items():
            print(
                f"  {mode_name:<14} "
                f"expDD {stats.get('expected_max_drawdown_pct', 0):.1f}%  "
                f"VaR DD {stats.get('var_max_drawdown_pct', 0):.1f}%  "
                f"CVaR DD {stats.get('cvar_max_drawdown_pct', 0):.1f}%  "
                f"ruin {stats.get('ruin_probability', 0):.1%}  "
                f"VaR ret {stats.get('var_return_pct', 0):+.1f}%"
            )
        for w in getattr(mc, "warnings", []) or []:
            print(f"  WARNING: {w}")
            logger.warning("Monte Carlo: %s", w)

    if not result.bankroll_sweep.empty:
        print("\n  FRACTIONAL KELLY SWEEP")
        print("  " + "-" * 56)
        for _, row in result.bankroll_sweep.iterrows():
            if row.get("performance_available") is False or int(row.get("trades", 0) or 0) == 0:
                print(
                    f"  edge >= {row['min_edge']:.0%}  "
                    f"trades {int(row.get('trades', 0) or 0):>4}  "
                    f"hit N/A  ROI N/A"
                )
                continue
            roi = row.get("roi_pct", 0)
            flag = ""
            if pd.notna(roi) and roi > 500:
                flag = "  <<< unrealistic?"
                logger.warning(
                    "Kelly sweep edge>=%s%% ROI %.1f%% looks unrealistically high",
                    int(row.get("min_edge", 0) * 100),
                    roi,
                )
            print(
                f"  edge >= {row['min_edge']:.0%}  "
                f"trades {int(row.get('trades', 0)):>4}  "
                f"hit {_fmt_pct_or_na(row.get('hit_rate', 0))}  "
                f"ROI {_fmt_roi_or_na(roi)}  "
                f"maxDD {row.get('max_drawdown_pct', 0):.1f}%{flag}"
            )

    if not result.per_event.empty:
        print("\n  PER-EVENT (top 10 by fights)")
        print("  " + "-" * 72)
        top = result.per_event.sort_values("fights", ascending=False).head(10)
        for _, row in top.iterrows():
            odds_n = int(row.get("fights_as_of_eligible", row.get("fights_with_odds", 0)))
            print(
                f"  {str(row['event'])[:38]:<38} "
                f"{int(row['fights']):>3} fights  "
                f"acc {row['accuracy']:.1%}  "
                f"as-of {odds_n}"
            )

    print(f"\n  Reports -> {result.report_dir}")
    print()


def save_calibration_plot(bins: pd.DataFrame, path: Path) -> None:
    if bins.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.scatter(bins["mean_predicted"], bins["fraction_positive"], s=bins["count"] * 3)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration reliability")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roi_plot(sweep: pd.DataFrame, path: Path) -> None:
    if sweep.empty:
        return
    work = sweep.copy()
    if "roi_pct" in work.columns:
        work = work[work["roi_pct"].notna()]
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(work["min_edge"] * 100, work["roi_pct"], width=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Min edge (%)")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Fractional Kelly ROI by edge threshold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
