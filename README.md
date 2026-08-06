# UFC Predictor

Standalone UFC fight prediction and high-accuracy (HA) betting-signal pipeline. **Not tied to PythonTrading** — all paths are relative to this project root (`C:\UFC-Predictor`).

## Project layout

```
UFC-Predictor/
├── src/                       Python modules + dashboard
│   ├── bet_tiers.py           Blue/Green/Yellow/Red color rules
│   ├── bet_slip.py            Top recommended dedupe + ranking
│   └── ufc_dashboard.py       CustomTkinter GUI
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

### Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | Card sections, color-ranked fight tables, Top recommended |
| **Odds API** | Primary free-tier moneylines + edges |
| **Odds API Props** | Over/Under 1.5 (HA Blue props = **Over 1.5 only**, live) |
| **MyBookie** | Optional moneyline scraper (`MYBOOKIE_ENABLED=true`) |
| **Props - MyBookie** | Optional MyBookie prop lines |
| **Next Two Cards** | Upcoming UFC.com cards (closest first) |
| **Risk Analysis** | Monte Carlo drawdown / ruin |
| **Ollama Analysis** | Local LLM narrative over HA Top 5 (+ fun $0 fillers) |
| **Arb Scanner** | Cross-book arb scan |

BetNow / DraftKings (and their Props tabs) appear only when those scrapers are enabled in `.env` (keep DraftKings off to protect Odds API quota).

### Toolbar

**Profile** (Paper/Live) · **Event** · **Refresh** · **Soft Update** · **Restart** · **Full** · **Bankroll $**

| Control | Behavior |
|---------|----------|
| **Refresh** | Load UFC.com next-two cards + predictions; reuses odds cache when `ODDS_FETCH_ONCE=true` |
| **Soft Update** | Reload `.env` (config) + attach book lines/props from cache — no extra Odds API burn when fetch-once is on |
| **Restart** | Quit and relaunch so `.env` / code load cleanly |
| **Full** | Toggle fullscreen |
| **Bankroll $** | Persisted roll; card budget = bankroll × profile risk % |

### Color legend

Row color = pick-side math + HA decision (not fight-name vibes). Tables sort **Blue → Green → Yellow → Red**, then edge↓, model prob↓, fight name↑.

| Color | Meaning | Money |
|-------|---------|-------|
| **Blue** | Clears HA gates | Real ticket ($) |
| **Green** | Strong lean / +EV but HA SKIP (e.g. `wide_interval`) | Fun `$0` |
| **Yellow** | Caution — thin edge / borderline | Fun `$0` |
| **Red** | Don't bet — negative edge, low prob, or `no_odds` | Fun `$0` |

Top recommended caps at **5**, deduped across books; Blue preferred; Red omitted when non-red options exist.

### AI narrative

**Ollama Analysis** is the default narrative tab (local; default model `qwen2.5-coder:7b`). It never invents bets — HA tickets still show if the LLM times out.

Optional **Grok / xAI** cloud narrative via `GROK_ENABLED` + `GROK_API_KEY` / `XAI_API_KEY` — off by default; **not required**.

## Odds sources

1. **The Odds API** (free tier) — primary moneylines + props when enabled  
2. **Optional scrapers** — MyBookie / BetNow / DraftKings when toggled on  
3. **Fail-closed** — no usable odds → `NO BET — no usable odds (fail-closed)`  
4. **Quota-safe cache** — `ODDS_FETCH_ONCE=true` reuses the first download until you delete cache files

| Variable | Default | Purpose |
|----------|---------|---------|
| `THE_ODDS_API_KEY` | — | Required for Odds API |
| `ODDS_FETCH_ONCE` | `true` | Reuse first download until cache deleted |
| `ODDS_CACHE_TTL_MINUTES` | `20` | Only when fetch-once is off |
| `MYBOOKIE_ENABLED` | `false` | Optional scraper |
| `BETNOW_ENABLED` / `DRAFTKINGS_ENABLED` | `false` | Optional; keep DK off for quota |

Delete only when you want a fresh live pull:

- `data/cache/ufc_odds_api.csv`
- `data/cache/the_odds_api_prop_odds.csv`
- `data/cache/the_odds_api_prop_odds.once`

## Profiles & HA skips

| Profile | Use |
|---------|-----|
| **Paper** (default) | Simulation / dashboard — looser card % |
| **Live** | Real money — hard USD card cap |

Set `UFC_PROFILE=paper` or `live` in `.env`. Legacy `research` → paper.

Common Kelly / alert SKIP labels (still shown as Green/Yellow/Red, never Blue):

| Reason | Meaning |
|--------|---------|
| `SKIP:wide` / `wide_interval` | Confidence interval too wide |
| `high_disagreement` | Ensemble models disagree |
| `low_model_prob` | Pick below min model probability |
| `no_odds` | No matched / usable price |
| `min_edge` | Edge below profile floor |

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
      → bet_tiers + bet_slip (color rank + Top 5)
      → ufc_dashboard (+ Ollama; optional Grok)
      → background_runner (cache-first when ODDS_FETCH_ONCE)
```

| Layer | Modules | Role |
|-------|---------|------|
| Data | `data_loader` | UFC.com cards, multi-source history |
| Model | `predictor`, `ensemble` | Calibrated LGBM+XGB |
| Gates | `uncertainty_gates`, `high_accuracy_strategy` | Fail-closed HA sizing |
| Color | `bet_tiers` | Blue/Green/Yellow/Red |
| Odds | `odds_providers/*`, `odds_api_client` | Odds API + optional scrapers |
| Dashboard | `ufc_dashboard`, `dashboard_service`, `bet_slip` | GUI + Top 5 |
| Narrative | `ollama_client`, `grok_analysis` | Ollama default; Grok optional |

## Background runner

```bash
python src/background_runner.py --mode auto --trigger startup
```

Scheduled tasks run full analysis with **cache-first odds** when `ODDS_FETCH_ONCE=true`. Snapshots under `data/cache/background/`.

## EXE builds (optional)

```bat
build_dashboard.bat
build_exe.bat
```

Prefer `START_DASHBOARD.bat` / Python day-to-day. Frozen builds may hit PyArrow DLL errors.

## Safety

- **HA fail-closed** — no sized bets without usable odds + uncertainty clearance
- **Blue = real money only** — Green/Yellow/Red never get HA stake
- **Daily loss circuit breaker** — `src/circuit_breaker.py`
- **Peak drawdown halt** — `risk_manager.DrawdownHalt`
- **Alert cooldown + fingerprint dedup**
- **Dry-run** — `ALERT_DRY_RUN=true` or `--dry-run`

## Configuration

Copy `.env.example` → `.env`. Important keys:

| Variable | Default | Purpose |
|----------|---------|---------|
| `UFC_PROFILE` | paper | Paper vs Live risk caps |
| `INITIAL_BANKROLL` | 75 | Starting bankroll |
| `THE_ODDS_API_KEY` | — | Odds API key |
| `ODDS_FETCH_ONCE` | true | One download, reuse until cache deleted |
| `ENABLE_PROPS` | false | Prop tabs (Over 1.5 HA when on) |
| `MYBOOKIE_ENABLED` | false | MyBookie + Props - MyBookie tabs |
| `DRAFTKINGS_ENABLED` | false | Keep false to protect API quota |
| `OLLAMA_ENABLED` | true | Local Ollama Analysis tab |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Prefer 7b; 14b often times out |
| `GROK_ENABLED` | false | Optional cloud narrative (not required) |

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

Includes color-tier rules, fight-table sort, Top recommended dedupe, and Ollama props wiring.

## Design notes

- **Leakage-safe features**: rolling stats use only prior fights.
- **No paid odds required**: Odds API free tier + optional scrapers.
- **Credit-safe by default**: `ODDS_FETCH_ONCE` + fail-closed without usable lines.
- **Separate from PythonTrading**: no merge with the Alpaca stock bot.
