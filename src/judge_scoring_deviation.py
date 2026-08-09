"""Judge scoring deviation vs panel consensus (Phase 2 display / research).

Field names use \"scoring_deviation\" / \"panel_disagreement\" — never
corrupt/controversial as computed columns. Extreme-tail UI labels are separate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RELIABILITY_MIN_ROUNDS = 50
# Display-only extreme-tail label threshold (not a feature name).
EXTREME_DEVIATION_Z = 2.0


def _round_winner(score_f1: int, score_f2: int) -> str:
    if score_f1 > score_f2:
        return "f1"
    if score_f2 > score_f1:
        return "f2"
    return "draw"


def expand_round_rows(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten cached mmadecisions rows to one row per judge-round."""
    rows: list[dict[str, Any]] = []
    for dec in decisions:
        judges = dec.get("judges") or []
        if len(judges) < 2:
            continue
        n_rounds = int(dec.get("n_rounds") or 0)
        for r_i in range(n_rounds):
            scores = []
            for j in judges:
                rnds = j.get("rounds") or []
                if r_i >= len(rnds):
                    continue
                rr = rnds[r_i]
                scores.append(
                    (
                        j.get("judge_id"),
                        j.get("judge_name"),
                        int(rr["score_f1"]),
                        int(rr["score_f2"]),
                        _round_winner(int(rr["score_f1"]), int(rr["score_f2"])),
                    )
                )
            if len(scores) < 2:
                continue
            # Panel majority among judges
            votes = [s[4] for s in scores if s[4] != "draw"]
            if not votes:
                majority = "draw"
            else:
                # mode
                vals, counts = np.unique(votes, return_counts=True)
                majority = str(vals[np.argmax(counts)])
            split_round = len(set(votes)) > 1 if votes else False
            for jid, jname, s1, s2, w in scores:
                rows.append(
                    {
                        "decision_id": dec.get("decision_id"),
                        "event": dec.get("event"),
                        "fighter_1": dec.get("fighter_1"),
                        "fighter_2": dec.get("fighter_2"),
                        "decision_type": dec.get("decision_type"),
                        "is_ufc": bool(dec.get("is_ufc")),
                        "round": r_i + 1,
                        "judge_id": jid,
                        "judge_name": jname,
                        "score_f1": s1,
                        "score_f2": s2,
                        "round_winner": w,
                        "panel_majority": majority,
                        "disagrees_with_majority": int(
                            w != "draw" and majority != "draw" and w != majority
                        ),
                        "split_round": int(split_round),
                    }
                )
    return pd.DataFrame(rows)


def judge_deviation_summary(
    round_df: pd.DataFrame,
    *,
    min_rounds: int = RELIABILITY_MIN_ROUNDS,
) -> pd.DataFrame:
    """
    Per-judge disagreement rate vs panel majority, with empirical-Bayes shrink.

    Below ``min_rounds``, shrink toward pool mean weighted by round count.
    """
    if round_df is None or round_df.empty:
        return pd.DataFrame()

    pool_rate = float(round_df["disagrees_with_majority"].mean())
    g = round_df.groupby(["judge_id", "judge_name"], dropna=False)
    rows = []
    for (jid, jname), sub in g:
        n = int(len(sub))
        raw = float(sub["disagrees_with_majority"].mean()) if n else float("nan")
        # EB: (n/(n+m))*raw + (m/(n+m))*pool with m = min_rounds
        m = float(min_rounds)
        shrunk = (n / (n + m)) * raw + (m / (n + m)) * pool_rate if n else pool_rate
        split_sub = sub[sub["split_round"] == 1]
        split_n = int(len(split_sub))
        split_raw = (
            float(split_sub["disagrees_with_majority"].mean()) if split_n else float("nan")
        )
        reliable = n >= min_rounds
        rows.append(
            {
                "judge_id": jid,
                "judge_name": jname,
                "n_rounds": n,
                "n_split_rounds": split_n,
                "disagreement_rate_raw": raw,
                "disagreement_rate_shrunk": shrunk,
                "split_round_disagreement_rate_raw": split_raw,
                "pool_disagreement_rate": pool_rate,
                "reliable": reliable,
                # Display-only hint — not a model feature name
                "ui_extreme_tail_label": (
                    "controversial"
                    if reliable and abs(shrunk - pool_rate) > 0.08 and shrunk > pool_rate
                    else ""
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_rounds", ascending=False)
    return out.reset_index(drop=True)


def format_judge_display_note(row: dict[str, Any] | pd.Series) -> str:
    """Human note for overlay when reliability floor cleared."""
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    if not bool(get("reliable")):
        return ""
    name = str(get("judge_name") or "Judge")
    n = int(get("n_rounds") or 0)
    rate = get("disagreement_rate_shrunk")
    try:
        rate_s = f"{100.0 * float(rate):.1f}%"
    except (TypeError, ValueError):
        rate_s = "?"
    note = f"judge history: {n} rounds, panel disagreement {rate_s}"
    label = str(get("ui_extreme_tail_label") or "")
    if label:
        note = f"{note} [{label}]"
    return note
