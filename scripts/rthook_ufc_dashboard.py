"""PyInstaller runtime hook — ensure project root + main module resolve when frozen."""

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        root = Path(meipass)
        exe_root = Path(sys.executable).resolve().parent
        for path in (str(root), str(exe_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
