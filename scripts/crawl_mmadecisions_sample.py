"""Crawl a bounded mmadecisions ID window (UFC RBR cards) into cache."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.mmadecisions import CACHE_DIR, DECISIONS_JSONL, crawl_decision_id_range, load_decision_cache

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("mma_crawl")


def main() -> int:
    # Recent-ish IDs spanning ~2024-2026 UFC decisions (site ID space).
    start = int(os.getenv("MMADEC_START_ID", "15200"))
    end = int(os.getenv("MMADEC_END_ID", "15800"))
    max_keep = int(os.getenv("MMADEC_MAX_KEEP", "120"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh run for assessment sample
    if DECISIONS_JSONL.is_file() and os.getenv("MMADEC_APPEND", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        DECISIONS_JSONL.unlink()
        logger.info("Cleared %s", DECISIONS_JSONL)

    summary = crawl_decision_id_range(
        start, end, ufc_only=True, sleep_s=0.3, max_keep=max_keep
    )
    rows = load_decision_cache()
    n_rounds = sum(int(r.get("n_rounds") or 0) * int(r.get("n_judges") or 0) for r in rows)
    out = {
        **summary,
        "cached_decisions": len(rows),
        "cached_judge_rounds": n_rounds,
        "sample_events": [r.get("event") for r in rows[:8]],
    }
    meta = CACHE_DIR / "crawl_meta.json"
    meta.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
