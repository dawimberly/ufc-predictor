# Canonical UFC project

**Edit and build here only:** `C:\UFC-Predictor`

Trading bot stays separate at `C:\Users\Owner\PythonTrading\stock-bot`.

## Paths

| Item | Location |
|------|----------|
| Project root | `C:\UFC-Predictor` |
| Dashboard (preferred) | `START_DASHBOARD.bat` → `pythonw -u src\ufc_dashboard.py` |
| Dashboard EXE (optional) | `dist\ufc-dashboard.exe` (may crash on PyArrow/`arrow.dll`) |
| `.env` | `C:\UFC-Predictor\.env` |
| Odds cache | `data\cache\ufc_odds_api.csv` (+ prop `.once` marker) |

## Launch

```bat
cd /d C:\UFC-Predictor
START_DASHBOARD.bat
```

## Build (optional)

```bat
cd /d C:\UFC-Predictor
build_dashboard.bat
```

## Compat

`C:\UFC-Bot\ufc-predictor` may be a junction or older mirror. Prefer `C:\UFC-Predictor`.
