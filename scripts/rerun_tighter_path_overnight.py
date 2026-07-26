"""Re-run failed tighter-path WF and refresh overnight executive summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from scripts import overnight_ha_eval as ov

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

META = Path(f"reports/ha_overnight_eval_{ov.STAMP}_meta.json")
FIXED5 = Path(f"reports/ha_wf_fixed_stake_5_{ov.STAMP}_summary.json")
FIXED3 = Path(f"reports/ha_wf_fixed_stake_3_{ov.STAMP}_summary.json")
LIVE = Path(f"reports/ha_wf_live_profile_{ov.STAMP}_summary.json")
AUTOPSY_CSV = Path(f"reports/ha_loss_autopsy_{ov.STAMP}.csv")
AUTOPSY_MD = Path(f"reports/ha_loss_autopsy_{ov.STAMP}.md")
SLEEVE = Path(f"reports/ha_sleeve_eval_{ov.STAMP}.csv")


def _summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("summary") or {}


def main() -> int:
    ov.LOG.clear()
    ov._log(f"Re-run tighter path stamp={ov.STAMP}")

    # Reconstruct prior job results for the executive summary.
    results: dict = {}
    if META.is_file():
        prior = json.loads(META.read_text(encoding="utf-8"))
        # Keep prior autopsy/sleeve payloads if present; otherwise rebuild lightly.
        prior_results = prior.get("results") or {}
        results["autopsy"] = prior_results.get("autopsy") or {
            "csv": str(AUTOPSY_CSV),
            "md": str(AUTOPSY_MD),
        }
        results["sleeve"] = prior_results.get("sleeve") or {"csv": str(SLEEVE), "weak_segments": []}
        # Prefer fresh summaries from saved WF reports.
        results["fixed_stake"] = {
            "fixed_5": _summary(FIXED5),
            "fixed_3": _summary(FIXED3),
        }
        results["live"] = {"summary": _summary(LIVE)}
    else:
        results["fixed_stake"] = {"fixed_5": _summary(FIXED5), "fixed_3": _summary(FIXED3)}
        results["live"] = {"summary": _summary(LIVE)}
        results["autopsy"] = ov._safe("loss_autopsy", ov.job_loss_autopsy)
        results["sleeve"] = ov._safe("sleeve_eval", ov.job_sleeve_eval)

    # If autopsy/sleeve lack pattern fields, refresh them (fast).
    if not (results.get("autopsy") or {}).get("patterns"):
        results["autopsy"] = ov._safe("loss_autopsy", ov.job_loss_autopsy)
    if "weak_segments" not in (results.get("sleeve") or {}):
        results["sleeve"] = ov._safe("sleeve_eval", ov.job_sleeve_eval)

    results["tighter"] = ov._safe("tighter_path_wf", ov.job_tighter_path)
    summary_path = ov._safe("executive_summary", lambda: ov.job_executive_summary(results))

    meta = {
        "stamp": ov.STAMP,
        "as_of": str(ov.AS_OF.date()),
        "results": {
            k: (
                v
                if not isinstance(v, dict) or "summary" not in v
                else {"summary": v.get("summary"), "paths": v.get("paths"), "error": v.get("error")}
            )
            for k, v in results.items()
        },
        "executive_summary": summary_path,
        "log": ov.LOG,
        "note": "tighter_path re-run after overnight fail",
    }
    META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    ov._log("TIGHTER PATH RE-RUN COMPLETE")
    return 0 if not (results.get("tighter") or {}).get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
