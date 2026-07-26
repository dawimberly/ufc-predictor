# UFC Predictor

Standalone UFC fight prediction and high-accuracy (HA) betting-signal pipeline. **Not tied to PythonTrading** — all paths are relative to this project root (`C:\UFC-Predictor`).

## Project layout

```
UFC-Predictor/
├── src/                       Python modules + dashboard
├── data/
│   ├── raw/                   fights.csv
│   ├── processed/             fight_features.csv
│   ├── cache/                 odds, fighter cache, background snapshots
│   └── logs/
├── models/                    ensemble_winner.joblib
├── assets/                    app icon
├── dist/                      optional frozen EXEs (prefer Python launch)
├── ufc_betting_bot/           vendored edge/Kelly/backtest helpers
├── START_DASHBOARD.bat        recommended GUI launcher
├── config.py
├── main.py
└── README.md
```

## Quick start

```bash
cd C:\UFC-Predictor
pip install -r requirements.txt
copy .env.example .env
# edit .env → set THE_ODDS_API_KEY, ENABLE_PROPS=true, ODDS_FETCH_ONCE=true
python scripts/preflight.py
python main.py --backtest-2025
```

### Dashboard (recommended)

Prefer **Python**, not the frozen EXE (PyArrow/`arrow.dll` can crash the windowed build):

```bat
START_DASHBOARD.bat
```

Or:

```bash
pythonw -u src/ufc_dashboard.py
```

Working directory must be the project root so `.env` and `data/` resolve correctly.

## Dashboard

| Tab | Purpose |
|-----|---------|
| **Overview** | Card summary, fight table, HA / fail-closed messaging |
| **Odds API** | Primary free-tier moneylines + edges |
| **Odds API Props** | Over/Under 1.5 totals (HA actionable = **Over 1.5 only**) |
| **Next Two Cards** | Upcoming UFC.com cards (closest first) |
| **Ollama Analysis** | Local LLM Top 5: **ACTIONABLE** ($ sized) + **ADVISORY** ($0 research) |
| **Risk Analysis** | Monte Carlo drawdown / ruin stats |
| **Grok Analysis** | Optional cloud narrative (off by default) |

**Toolbar:** Profile (Paper/Live), **Refresh**, **Soft Update**, **Restart**, Fullscreen.

| Control | Behavior |
|---------|----------|
| **Refresh** | Reload next two UFC.com cards + predictions; reuses odds cache when `ODDS_FETCH_ONCE=true` |
| **Soft Update** | Reload `.env` + attach Odds API lines/props from cache (does not burn credits when fetch-once is on) |
| **Bankroll $** | Total roll (persisted). Auto card budget = bankroll × profile risk % |

### Green edge vs NO BET

- **Green** in the fight table = model shows a **positive edge** vs the price. That is **not** a bet ticket.
- **NO BET / don’t bet** = nothing cleared **HA gates** (wide uncertainty interval, ensemble disagreement, missing/untrusted odds, etc.).
- Ollama **ADVISORY** rows are research only (`$0`). Only green **$** sized rows are actionable.

## Odds API credits (`ODDS_FETCH_ONCE`)

Free-tier [The Odds API](https://the-odds-api.com) is **quota-limited**. Defaults protect credits:

| Variable | Default | Purpose |
|----------|---------|---------|
| `THE_ODDS_API_KEY` | — | Required for live/cache moneylines |
| `ODDS_FETCH_ONCE` | `true` | Reuse first download forever until you delete the cache files |
| `ODDS_CACHE_TTL_MINUTES` | `20` | Used only when `ODDS_FETCH_ONCE=false` |
| `DRAFTKINGS_ENABLED` | `false` | Keep off — DK props burn API quota per event |
| `BETNOW_ENABLED` / `MYBOOKIE_ENABLED` | `false` | Optional scrapers |

**Cache files** (delete these only when you intentionally want a fresh live pull):

- `data/cache/ufc_odds_api.csv`
- `data/cache/the_odds_api_prop_odds.csv`
- `data/cache/the_odds_api_prop_odds.once`

With fetch-once on, Soft Update / Refresh / nightly background jobs **must not** re-hit the API while those files exist. Fail-closed copy when odds are unusable:

`NO BET — no usable odds (fail-closed)`

## Profiles (`UFC_PROFILE`)

| Profile | Use | Card cap (typical) | Kelly | Min alert edge |
|---------|-----|-------------------|-------|----------------|
| `paper` (default) | Simulation / dashboard | Higher % of bankroll | 0.35 | 3.5% |
| `live` | Real money | Hard USD cap ($12 default) | 0.12 | 8% |

Legacy `research` maps to `paper`. Set in `.env`: `UFC_PROFILE=live`

## CLI

```bash
python -m src.cli_entry --next-two --odds
python main.py --preflight
python main.py --watch --auto-odds --dry-run
python main.py --backtest-2025
```

`launch_predict.bat` wraps the CLI (`--next-two --odds`).

## Architecture

```
data_loader → feature_engineering (+ fighter_cache) → model_trainer (LGBM+XGB)
      → predictor → uncertainty_gates + high_accuracy_strategy
      → dashboard_service (books / props / Soft Update)
      → ufc_dashboard (+ optional Ollama / Grok)
      → background_runner (scheduled; cache-first when ODDS_FETCH_ONCE)
```

| Layer | Modules | Role |
|-------|---------|------|
| Data | `data_loader` | UFC.com cards, multi-source history |
| Model | `predictor`, `ensemble` | Calibrated LGBM+XGB |
| Gates | `uncertainty_gates`, `high_accuracy_strategy` | Fail-closed HA sizing |
| Odds | `odds_providers/*`, `odds_api_client` | Cache-first Odds API + optional books |
| Dashboard | `ufc_dashboard`, `dashboard_service` | GUI, Soft Update, Top 5 slip |
| Narrative | `grok_analysis` | Ollama (local) / Grok (optional cloud) |

## Background runner

```bash
python src/background_runner.py --mode auto --trigger startup
```

Scheduled tasks (midnight / nightly) run full analysis with **cache-first odds** when `ODDS_FETCH_ONCE=true`. Snapshots live under `data/cache/background/`. Past / mismatched cards are rejected on dashboard startup.

## EXE builds (optional)

```bat
build_dashboard.bat
build_exe.bat
```

Prefer `START_DASHBOARD.bat` / Python for day-to-day use. Frozen dashboard may fail on some machines with PyArrow DLL errors.

## Safety

- **HA fail-closed** — no sized bets without usable odds + uncertainty clearance
- **Daily loss circuit breaker** — `src/circuit_breaker.py`
- **Peak drawdown halt** — `risk_manager.DrawdownHalt`
- **Alert cooldown + fingerprint dedup**
- **Dry-run** — `ALERT_DRY_RUN=true` or `--dry-run`

## Configuration

Copy `.env.example` → `.env`. Important keys:

| Variable | Default | Purpose |
|----------|---------|---------|
| `UFC_PROFILE` | paper | paper vs live risk caps |
| `INITIAL_BANKROLL` | 75 | Starting bankroll |
| `THE_ODDS_API_KEY` | — | Odds API key |
| `ODDS_FETCH_ONCE` | true | One download, reuse until cache deleted |
| `ENABLE_PROPS` | false | Prop tabs (Over 1.5 HA when enabled) |
| `DRAFTKINGS_ENABLED` | false | Keep false to protect API quota |
| `GROK_ENABLED` | false | Optional cloud narrative |

## Ops artifacts

| File | Purpose |
|------|---------|
| `data/logs/dashboard.log` | GUI + odds activity |
| `data/logs/background_runner.log` | Scheduled runner |
| `data/budget.json` | Bankroll / book toggles |
| `data/cache/background/manifest.json` | Snapshot metadata |
| `data/cache/heartbeat.json` | Runner liveness |

## Tests

```bash
python -m pytest tests/ -q
```

## Design notes

- **Leakage-safe features**: rolling stats use only prior fights.
- **No paid odds required**: The Odds API free tier + scrapers optional.
- **Credit-safe by default**: `ODDS_FETCH_ONCE` + hard block on repeat live pulls.
- **Separate from PythonTrading**: no merge with the Alpaca stock bot.
