# UFC Predictor

Standalone UFC fight prediction and high-accuracy (HA) betting-signal pipeline. **Not tied to PythonTrading** — all paths are relative to this project root (`C:\UFC-Predictor`).

**Repos:** [dawimberly/ufc-predictor](https://github.com/dawimberly/ufc-predictor) · [infinite-robots/ufc-predictor](https://github.com/infinite-robots/ufc-predictor) (private)

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
| **Props - MyBookie** | Optional MyBookie prop lines |
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

When conformal CI width triggers `SKIP:wide` / `wide_interval`, **Paper** can still size a tiny ticket if the market-blended probability clears a modest band:

- override enabled (`PAPER_WIDE_OVERRIDE_ENABLED=true`)
- pure `wide_interval` (no other hard skips)
- probability = 40% raw model + 60% de-vigged market (`prob_f1_raw` when present)
- blended prob 58–75%, edge versus the price 2–8%, decimal odds 1.40–2.30
- Kelly multiplier 0.20, stake hard-capped at **1% bankroll** (card-pool allocation cannot raise it), max **3** of these singles per card

A raw model price like 96% on a short favorite stays a skip. That was the losing book.

**Live stays fail-closed** — wide CI never becomes a Live HA ticket. 2025 re-score autopsy: wide-CI miss rate ~44% vs ~3% narrow — validates Live fail-closed + Paper sky-blue exception.

HA-sized 2-leg parlays use **Deep Blue legs only** (uncertainty action `allow`, narrow CI). A Sky Blue override is a single at the 1% cap and is never a parlay leg. Walk-forward conformal half-width is not squeezed by default (`HA_WF_CONFORMAL_Q_CAP` empty), so a wide interval stays wide and cannot be sized as BET THIS.

### Auto parlays (Ollama Analysis)

Advisory **2-leg** and **3-leg** research parlays are built from HA singles / high model probs (`build_auto_parlay_recommendations`). Shown in the Ollama Analysis tab as styled cards — **$0 advisory only**, not Live HA-sized.

### AI narrative

**Ollama Analysis** is the default narrative tab (local; default model `qwen2.5-coder:7b`). It never invents bets — HA tickets still show if the LLM times out.

Prompts and the Stats / Best bets briefing use the same action verbs (**BET THIS** / **FUN ONLY** / **DO NOT BET**). Ask “best bets” → sized tickets first with `$`, then optional FUN ONLY leans, never treating Green as bankroll.

Optional **Grok / xAI** cloud narrative via `GROK_ENABLED` + `GROK_API_KEY` / `XAI_API_KEY` — off by default; **not required**.

### Context strip (display only)

Selecting a fight can show weigh-in photos / missed-weight notes (`weigh_in`) and integrity flags (`fighter_flags`) — **context only**, not model features. Controversial methods / decision-profile / home-country / pathway A/Bs are **DROP** for production features; leave those flags off unless a keep rule is re-run and passes.

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

## Model features (HV)

Phase-1 **high-value** features are on by default (`ENABLE_HIGH_VALUE_FEATURES=true`, schema v5) after 2025 A/B (~+0.008 AUC). Toggle off in `.env` for ablation; do not retrain casually — production ensemble already includes HV.

### Programmed matchup rules

After the ensemble scores a fight, `src/programmed_rules.py` adds a small signed log-odds nudge (capped by `STYLE_BONUS_MAX`, default 0.05). The rules are explicit, leakage-safe, and not in `FEATURE_COLUMNS` — no retrain:

| Rule | When it fires |
|------|----------------|
| Southpaw vs orthodox | Signed toward the southpaw (card order does not favor fighter 1) |
| Striker vs grappler / style-clash grappler | Clear style split |
| Reach that lands | ≥3" reach and striking accuracy (plus volume) on the same side |
| Height and reach | Both physical edges ≥2" and the same direction |
| Short-notice camp | One fighter on short notice; shrinks if their short-notice record is good |
| Layoff rust | 365-day flag if present, otherwise 180-day; the two do not stack |
| New division | First fight in a new weight class |
| Prior KO loss | Been stopped before; larger if the opponent has the power |
| Past division peak | ≥3 years further past the division peak |
| Open wrestling path | Style clash plus takedown accuracy, unless a takedown-defense wall blocks it |
| Form streak / experience | Last-5 and momentum agree; veteran edge only with recent form |

The fight context strip and fight briefs name the rules that fired. Turning off the style adjustment (`apply_style_bonus=False`) turns these off with it.

Helpers:

```bash
python scripts/ab_high_value_features.py
python scripts/productionize_hv_features.py
python scripts/rescore_2025_upset.py
python scripts/upset_autopsy_backtest.py
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
      → predictor (+ programmed_rules) → uncertainty_gates (+ Paper wide override) + high_accuracy_strategy
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
| Gates | `uncertainty_gates`, `programmed_rules`, `high_accuracy_strategy` | Fail-closed HA sizing; signed matchup nudges |
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
| `ENABLE_HIGH_VALUE_FEATURES` | true | Phase-1 HV feature block |
| `PAPER_WIDE_OVERRIDE_ENABLED` | true | Paper market-blend singles on wide CI |
| `MARKET_BLEND_MARKET_WEIGHT` | 0.60 | Weight on the de-vigged market |
| `MARKET_BLEND_MIN_PROB` / `MAX` | 0.58 / 0.75 | Blended probability band |
| `MARKET_BLEND_MIN_EDGE` / `MAX` | 0.02 / 0.08 | Edge versus the price |
| `MARKET_BLEND_MIN_ODDS` / `MAX` | 1.40 / 2.30 | Decimal price band |
| `PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC` | 0.01 | Hard stake cap vs bankroll |
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

- **Leakage-safe features**: rolling stats use only prior fights.
- **No paid odds required**: Odds API free tier + optional scrapers.
- **Credit-safe by default**: `ODDS_FETCH_ONCE` + fail-closed without usable lines.
- **Separate from PythonTrading**: no merge with the Alpaca stock bot.
