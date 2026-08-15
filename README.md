# UFC Predictor

Standalone UFC fight prediction and high-accuracy (HA) betting-signal pipeline. **Not tied to PythonTrading** — all paths are relative to this project root (`C:\UFC-Predictor`).

**Repos:** [dawimberly/ufc-predictor](https://github.com/dawimberly/ufc-predictor) · [infinite-robots/ufc-predictor](https://github.com/infinite-robots/ufc-predictor) (private)

## Recent changes

- **2026-08-15** — Land unmerged dashboard work on master: empty-book odds restore, fetch-once slate locks, instant fight-stats chat, MyBookie KO/sub/decision props with model edges, totals label fix, and no false Blue on research lines.
- **2026-08-14** — Show MyBookie method props (KO/sub/decision) on Props with model edges, fix totals labels, and stop false Blue on research lines.
- **2026-08-12** — Fix Odds API fetch-once slate locks and make Ollama fight stats instant.
- **2026-08-11** — Empty-book odds: fixed `UnboundLocalError` in book odds merge that blanked all books; Soft Update / Quick Odds restore matched lines. See `data/reports/odds_empty_incident.md`.
- **2026-08-11** — Sky Blue on Odds API / MyBookie fight tables now matches Ollama (Kelly `paper_wide_override` status is parsed for color).

Every **Commit** (UI or CLI) auto-appends a **Recent changes** bullet from the commit subject and **pushes `origin`** (personal + Infinite Robots push URLs). Skip with `[skip-readme]` / `[no-push]` in the message, or `SKIP_README_HOOK=1` / `SKIP_AUTO_PUSH=1`.

## Project layout

```
UFC-Predictor/
├── src/                       Python modules + dashboard
│   ├── bet_tiers.py           BET THIS / FUN ONLY action verbs + color tiers
│   ├── bet_slip.py            Top recommended dedupe + ranking
│   ├── high_value_features.py Phase-1 HV feature block (production default)
│   ├── strategy.py            HA sizing + auto 2/3-leg parlay recs
│   ├── uncertainty_gates.py   Conformal CI gates + Paper wide override
│   ├── fight_context.py       Display-only context strip
│   ├── weigh_in.py            Weigh-in photos / missed-weight notes
│   ├── fighter_flags.py       Integrity skip / badge flags
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

Working directory must be the project root so `.env` and `data/` resolve correctly. Desktop shortcuts can be rebuilt with `scripts/create_dashboard_shortcut.ps1` (prefers Python 3.14).

## Dashboard

### Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | Card sections, color-ranked fight tables, Top recommended |
| **Odds API** | Primary free-tier moneylines + edges |
| **Odds API Props** | Over/Under 1.5 (HA Blue props = **Over 1.5 only**, live) |
| **MyBookie** | Optional moneyline scraper (`MYBOOKIE_ENABLED=true`) |
| **Props - MyBookie** | Optional MyBookie prop lines (totals + KO/sub/decision method props; method lines are research-only, never HA Blue) |
| **Next Two Cards** | Upcoming UFC.com cards (closest first) |
| **Risk Analysis** | Monte Carlo drawdown / ruin |
| **Ollama Analysis** | Local LLM narrative over HA Top 5 — leads with **WHAT TO BET (sized)** vs **FUN ONLY ($0)** |
| **Arb Scanner** | Cross-book arb scan |

BetNow / DraftKings (and their Props tabs) appear only when those scrapers are enabled in `.env` (keep DraftKings off to protect Odds API quota).

### Toolbar

**Profile** (Paper/Live) · **Event** · **Refresh** · **Soft Update** · **Restart** · **Full** · **Bankroll $**

| Control | Behavior |
|---------|----------|
| **Refresh** | Load UFC.com next-two cards + predictions; reuses odds cache when `ODDS_FETCH_ONCE=true` |
| **Soft Update** | Reload `.env` (config) + attach book lines/props from cache — no extra Odds API burn when fetch-once is on |
| **Restart** | Quit and relaunch so `.env` / code load cleanly (needed after code changes; Soft Update does not reload modules) |
| **Full** | Toggle fullscreen |
| **Bankroll $** | Persisted roll; card budget = bankroll × profile risk % |

### Color legend + action verbs

Row color = pick-side math + HA decision (not fight-name vibes). Tables sort **Deep Blue → Sky Blue → Green → Yellow → Red**, then edge↓, model prob↓, fight name↑.

Overview Top Recommended and Ollama Analysis lead with a plain **WHAT TO BET** line so sized vs fun never blur:

| Color | Action verb | Meaning | Money |
|-------|-------------|---------|-------|
| **Deep Blue** `#3b82f6` | **BET THIS** | Clears full HA gates | Real ticket ($) |
| **Sky Blue** `#57B9FF` | **TINY PAPER BET** | Paper-only `paper_wide_override` | Paper $ only (not Live) |
| **Green** | **FUN ONLY** | Strong lean / +EV but HA SKIP (e.g. `wide_interval`) | `$0` research — not bankroll |
| **Yellow** | **CAUTION — SKIP SIZED** | Thin edge / borderline | `$0` |
| **Red** | **DO NOT BET** | Negative edge, low prob, or `no_odds` | `$0` |

If no Blue/Sky Blue tickets exist, the header says **WHAT TO BET (sized): NONE** and may list FUN ONLY leans separately. Top recommended caps at **5**, deduped across books; Blue preferred over Sky Blue; Red omitted when non-red options exist.

### Paper wide override (Sky Blue)

When conformal CI width triggers `SKIP:wide` / `wide_interval`, **Paper** can still size a tiny ticket if:

- override enabled (`PAPER_WIDE_OVERRIDE_ENABLED=true`)
- pure `wide_interval` (no other hard skips)
- edge ≥ 8% and model prob ≥ 70% (defaults)
- Kelly multiplier 0.20, stake hard-capped at **1% bankroll**, max **2** override singles per card

**Live stays fail-closed** — wide CI never becomes a Live HA ticket. 2025 re-score autopsy: wide-CI miss rate ~44% vs ~3% narrow — validates Live fail-closed + Paper sky-blue exception.

### Auto parlays (Ollama Analysis)

Advisory **2-leg** and **3-leg** research parlays are built from HA singles / high model probs (`build_auto_parlay_recommendations`). Shown in the Ollama Analysis tab as styled cards — **$0 advisory only**, not Live HA-sized.

### AI narrative

**Ollama Analysis** is the default narrative tab (local; default model `qwen2.5-coder:7b`). It never invents bets — HA tickets still show if the LLM times out.

Prompts and the Stats / Best bets briefing use the same action verbs (**BET THIS** / **FUN ONLY** / **DO NOT BET**). Ask “best bets” → sized tickets first with `$`, then optional FUN ONLY leans, never treating Green as bankroll.

Optional **Grok / xAI** cloud narrative via `GROK_ENABLED` + `GROK_API_KEY` / `XAI_API_KEY` — off by default; **not required**.

### Context strip (display only)

Selecting a fight can show weigh-in photos / missed-weight notes (`weigh_in`) and integrity flags (`fighter_flags`) — **context only**, not model features. Always-on strip lines: market implied + Disagree; decision profile / judges when known. DROP research blocks (pathway, market, home, local, judge-geo, decision-profile, overseas, controversy) stay out of production features — see `data/reports/research_keep_drop.md`.

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
| **Paper** (default) | Simulation / dashboard — looser card %; Sky Blue override allowed |
| **Live** | Real money — hard USD card cap; wide CI fail-closed |

Set `UFC_PROFILE=paper` or `live` in `.env`. Legacy `research` → paper.

Common Kelly / alert SKIP labels (still shown as Green/Yellow/Red, never Deep Blue):

| Reason | Meaning |
|--------|---------|
| `SKIP:wide` / `wide_interval` | Confidence interval too wide |
| `paper_wide_override` | Paper-only tiny stake after wide skip → Sky Blue |
| `high_disagreement` | Ensemble models disagree |
| `low_model_prob` | Pick below min model probability |
| `no_odds` | No matched / usable price |
| `min_edge` | Edge below profile floor |

## Production features & Soft Update

**Production model** = **HV features** (`ENABLE_HIGH_VALUE_FEATURES=true`) + **HA uncertainty gates**. Sized tickets only clear those gates (Deep Blue / Paper Sky Blue). Ledger: `data/reports/research_keep_drop.md`. Freeze snapshot: `data/reports/PRODUCTION_FREEZE.md`.

**KEEP:** HV (+ gates). **DROP (flags default false):** pathway, market, home-country, local advantage, judge-geo (model), decision-profile, overseas travel, controversy. Do not flip `ADD_*` / `ENABLE_PATHWAY_*` / `ENABLE_MARKET_*` unless you re-run the A/B and the keep rule passes. Preflight warns (Live fails) if DROP flags are on.

**Display-only extras** (not model features): fight context strip (market implied, Disagree, decision profile / judges when known), weigh-in photos, integrity badges, overseas notes, Ollama/Grok narrative, FUN ONLY greens.

### Soft Update (canonical odds path)

1. **Refresh Next Two** — load UFC.com cards + model predictions; with `ODDS_FETCH_ONCE=true`, first Odds API download is cached.
2. **Soft Update** — reload `.env` + re-attach book moneylines/props from **cache** (no live Odds API burn while cache files exist). Soft-fail per book: missing lines stay blank (no fake −100% edges).
3. Fresh live pull only when needed: **Quick Odds + Props**, or delete `data/cache/ufc_odds_api.csv` (+ prop once markers).
4. After **code** changes: **Restart** (Soft Update does not reload Python modules).

Evaluate-only 2025 check (no retrain):

```bash
python scripts/run_production_backtest_2025.py
```

Trading-bot Task Scheduler jobs are out of scope for this UFC repo — do not edit them here.

## Model features (HV)

Phase-1 **high-value** features are on by default (`ENABLE_HIGH_VALUE_FEATURES=true`, schema v5) after 2025 A/B (~+0.008 AUC). Toggle off in `.env` for ablation; do not retrain casually — production ensemble already includes HV. Research A/B helpers stay available but are **not** part of the freeze path:

```bash
python scripts/ab_high_value_features.py
python scripts/productionize_hv_features.py
python scripts/run_production_backtest_2025.py
```
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
data_loader → feature_engineering (+ fighter_cache + HV) → model_trainer (LGBM+XGB)
      → predictor → uncertainty_gates (+ Paper wide override) + high_accuracy_strategy
      → dashboard_service (books / props / Soft Update)
      → bet_tiers + bet_slip (color rank + Top 5)
      → strategy (auto 2/3-leg parlays) → grok_analysis / Ollama
      → ufc_dashboard
      → background_runner (cache-first when ODDS_FETCH_ONCE)
```

| Layer | Modules | Role |
|-------|---------|------|
| Data | `data_loader` | UFC.com cards, multi-source history |
| Features | `feature_engineering`, `high_value_features` | Leakage-safe + HV block |
| Model | `predictor`, `ensemble` | Calibrated LGBM+XGB |
| Gates | `uncertainty_gates`, `high_accuracy_strategy` | Fail-closed HA sizing |
| Color | `bet_tiers` | BET THIS / FUN ONLY action verbs + color tiers |
| Odds | `odds_providers/*`, `odds_api_client` | Odds API + optional scrapers |
| Dashboard | `ufc_dashboard`, `dashboard_service`, `bet_slip` | GUI + Top 5 |
| Context | `fight_context`, `weigh_in`, `fighter_flags` | Display-only strip / skip flags |
| Narrative | `ollama_client`, `grok_analysis`, `strategy` | Ollama + auto parlays; Grok optional |

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
- **Live wide CI fail-closed** — Paper sky-blue override never applies to Live
- **BET THIS (Deep Blue / Sky Blue) = money tickets** — FUN ONLY / Yellow / Red never get HA stake
- **Sky Blue caps** — 1% bankroll + max 2 override singles/card
- **Ollama clarity** — sized NO BET vs FUN ONLY leans stated up front

## Configuration

Copy `.env.example` → `.env`. Important keys:

| Variable | Default | Purpose |
|----------|---------|---------|
| `UFC_PROFILE` | paper | Paper vs Live risk caps |
| `INITIAL_BANKROLL` | 75 | Starting bankroll |
| `THE_ODDS_API_KEY` | — | Odds API key |
| `ODDS_FETCH_ONCE` | true | One download, reuse until cache deleted |
| `ENABLE_PROPS` | false | Prop tabs (Over 1.5 HA when on) |
| `ENABLE_HIGH_VALUE_FEATURES` | true | Phase-1 HV feature block |
| `PAPER_WIDE_OVERRIDE_ENABLED` | true | Paper sky-blue tiny stakes on wide CI |
| `PAPER_WIDE_OVERRIDE_MIN_EDGE` | 0.08 | Min edge for override |
| `PAPER_WIDE_OVERRIDE_MIN_PROB` | 0.70 | Min model prob for override |
| `PAPER_WIDE_OVERRIDE_KELLY_MULT` | 0.20 | Kelly shrink for override |
| `PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC` | 0.01 | Hard stake cap vs bankroll |
| `PAPER_WIDE_OVERRIDE_MAX_PER_CARD` | 2 | Max override singles per card |
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

Includes color-tier / action-verb rules (incl. Sky Blue), fight-table sort, Top recommended dedupe, Paper wide override, auto parlays, HV features, fighter flags, weigh-in context, and Ollama props wiring.

## Design notes

