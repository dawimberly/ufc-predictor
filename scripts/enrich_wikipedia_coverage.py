"""Re-enrich Sherdog mismatches + slow batched Wikipedia on the same 100-fighter sample."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_coverage import (
    build_fighter_data_coverage,
    enrich_fighter_sources,
    save_fighter_data_coverage_report,
    select_notable_fighters,
)
from src.sherdog import load_sherdog_fighters, load_sherdog_fights
from src.wikipedia_fighters import load_wikipedia_fighters, match_wikipedia_row


def _load_prior_sample() -> list[str] | None:
    path = Path("reports/fighter_data_coverage_20260725.json")
    if not path.exists():
        reports = sorted(Path("reports").glob("fighter_data_coverage_*.json"), reverse=True)
        path = reports[0] if reports else path
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    sample = (data.get("refresh") or {}).get("sample") or []
    return [str(n) for n in sample if str(n).strip()] or None


def main() -> None:
    prior = _load_prior_sample()
    sample = prior or select_notable_fighters(target=100)
    print(f"sample={len(sample)}" + (" (same prior set)" if prior else ""))
    print(f"wikipedia missing before enrich: {sum(1 for n in sample if match_wikipedia_row(n) is None)}")

    # Fail-soft: Sherdog alias retry + slow batched Wiki. Sparse Wiki never blocks pipeline.
    enrich = enrich_fighter_sources(
        max_fetch=len(sample),
        sample=sample,
        wiki_batch_size=5,
        wiki_batch_delay_sec=8.0,
        wiki_per_fighter_delay_sec=5.0,
    )
    sherdog = enrich.get("sherdog") or {}
    wiki = enrich.get("wikipedia") or {}
    print(
        "sherdog:",
        {
            k: sherdog.get(k)
            for k in ("requested", "fetched", "cached", "failed", "failure_reasons")
        },
    )
    print(
        "wikipedia:",
        {
            k: wiki.get(k)
            for k in (
                "requested",
                "fetched",
                "cached",
                "skipped_hard_cache",
                "failed",
                "failure_reasons",
                "batch_size",
                "batch_delay_sec",
                "per_fighter_delay_sec",
            )
        },
    )

    report = build_fighter_data_coverage(refresh=False, max_fetch=100, sample=sample)
    report["refresh"] = {
        "sample_size": len(sample),
        "sample": sample,
        "sherdog": sherdog,
        "wikipedia": wiki,
        "note": "alias retry + slow batched wiki; fail-soft",
    }
    path = save_fighter_data_coverage_report(report)
    sh, w, cb, prior_cov = (
        report["sherdog"],
        report["wikipedia"],
        report["compubox_style"],
        report["prior_sport"],
    )

    def pct(x: object) -> str:
        try:
            return f"{100.0 * float(x):.1f}%"  # type: ignore[arg-type]
        except Exception:
            return "n/a"

    print("\n=== FIGHTER DATA COVERAGE (same 100-fighter sample) ===")
    print(f"fighters={report['n_fighters']}  sample={report['sample_size']}")
    print(
        f"Sherdog (all): profiles {pct(sh.get('pct_fighters'))}  "
        f"history {pct(sh.get('pct_with_history'))}  "
        f"cache {sh.get('cache_fighters')}/{sh.get('cache_fights')} fights"
    )
    print(
        f"Sherdog (sample): profiles {pct(sh.get('sample_pct_fighters'))}  "
        f"history {pct(sh.get('sample_pct_with_history'))}"
    )
    print(
        f"Wikipedia (all): matched {pct(w.get('pct_fighters'))}  "
        f"bio {pct(w.get('pct_with_bio_fields'))}  "
        f"height {pct(w.get('pct_height'))} reach {pct(w.get('pct_reach'))} "
        f"stance {pct(w.get('pct_stance'))} team {pct(w.get('pct_team'))}  "
        f"cache={len(load_wikipedia_fighters())}"
    )
    print(
        f"Wikipedia (sample): matched {pct(w.get('sample_pct_fighters'))}  "
        f"bio {pct(w.get('sample_pct_with_bio_fields'))}"
    )
    print(f"CompuBox-style: {pct(cb.get('pct_fighters'))}  rows={cb.get('bout_rows')}")
    print(f"Prior-sport: {pct(prior_cov.get('pct_known'))}")
    print(f"HTML: {path}")
    print(f"Sherdog cache: {len(load_sherdog_fighters())} fighters / {len(load_sherdog_fights())} fights")


if __name__ == "__main__":
    main()
