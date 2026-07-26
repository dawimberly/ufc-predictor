#!/usr/bin/env python3
"""Rank prop markets by reliability for high-accuracy betting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop_reliability import main

if __name__ == "__main__":
    raise SystemExit(main())
