#!/usr/bin/env python3
"""Re-runnable prop performance report (Over/Under 1.5 + other markets)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop_performance import main

if __name__ == "__main__":
    raise SystemExit(main())
