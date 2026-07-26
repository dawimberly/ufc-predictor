"""Allow `python -m sports_bot.app.cli` from the UFC-Predictor root.

Puts SportsBettingBot/src on sys.path, then hands off to the real CLI.
Usage:
  cd /d C:\\UFC-Predictor
  python sports_bot_path_bootstrap.py backtest --last=10
Or set PYTHONPATH=C:\\SportsBettingBot\\src and run:
  python -m sports_bot.app.cli backtest --last=10
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SPORTS_SRC = Path(r"C:\SportsBettingBot\src")
if SPORTS_SRC.is_dir() and str(SPORTS_SRC) not in sys.path:
    sys.path.insert(0, str(SPORTS_SRC))

if __name__ == "__main__":
    # Mimic `python -m sports_bot.app.cli ...`
    sys.argv = ["sports_bot.app.cli", *sys.argv[1:]]
    runpy.run_module("sports_bot.app.cli", run_name="__main__")
