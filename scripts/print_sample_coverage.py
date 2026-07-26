"""Quick coverage recompute for the prior 100-fighter sample (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from src.data_coverage import build_fighter_data_coverage, save_fighter_data_coverage_report
from src.sherdog import load_sherdog_fighters, sherdog_coverage
from src.wikipedia_fighters import load_wikipedia_fighters, wikipedia_coverage


def main() -> None:
    sample = json.loads(
        Path("reports/fighter_data_coverage_20260725.json").read_text(encoding="utf-8")
    )["refresh"]["sample"]
    sh_s = sherdog_coverage(sample)
    w_s = wikipedia_coverage(sample)
    print(
        "Sherdog sample",
        f"{100 * sh_s['pct_fighters']:.1f}%",
        "n",
        int(sh_s["n_matched"]),
    )
    print(
        "Wiki sample",
        f"{100 * w_s['pct_fighters']:.1f}%",
        "n",
        int(w_s["n_matched"]),
    )
    print("caches", len(load_sherdog_fighters()), len(load_wikipedia_fighters()))
    report = build_fighter_data_coverage(refresh=False, max_fetch=100, sample=sample)
    path = save_fighter_data_coverage_report(report)
    sh, w = report["sherdog"], report["wikipedia"]
    print(
        "FINAL Sherdog sample",
        f"{100 * float(sh['sample_pct_fighters']):.1f}%",
        "all",
        f"{100 * float(sh['pct_fighters']):.1f}%",
    )
    print(
        "FINAL Wiki sample",
        f"{100 * float(w['sample_pct_fighters']):.1f}%",
        "all",
        f"{100 * float(w['pct_fighters']):.1f}%",
    )
    print(path)


if __name__ == "__main__":
    main()
