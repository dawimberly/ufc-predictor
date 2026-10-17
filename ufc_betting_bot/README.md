# UFC Betting Bot

Standalone UFC value-betting layer — **separate from the crypto trading bot** in this repo. Uses the sibling [`ufc-predictor`](../ufc-predictor/) package for fight features and the trained ensemble model.

## Structure

```
ufc_betting_bot/
├── config/settings.py      # Bankroll rules, paths, odds URLs
├── modules/
│   ├── odds.py             # Historical odds merge (Kaggle + Odds API cache)
│   ├── edge.py             # Implied prob + edge (only when odds exist)
│   ├── bankroll.py         # Fractional Kelly + daily loss limit
│   └── model_bridge.py     # Adapter to ufc-predictor
├── backtester/
│   └── backtest_2025.py    # Year-scoped event backtest (frozen or walk-forward)
├── live_runner/runner.py   # Dry-run signal generation
├── dashboard/app.py        # Streamlit UI
└── main.py                 # CLI
```

## Quick start

```bash
cd ufc_betting_bot
pip install -r requirements.txt
cp .env.example .env

# Merge odds + run 2025 backtest (default: frozen_model)
python main.py --backtest-2025 --refresh-odds
# True walk-forward (refit on earlier fights only):
python main.py --backtest-2025 --walk-forward

# Dashboard
streamlit run dashboard/app.py
```

## Bankroll rules (defaults)

| Rule | Default | Env var |
|------|---------|---------|
| Fractional Kelly | 25% of full Kelly | `UFC_BOT_KELLY_FRACTION=0.25` |
| Max stake per bet | **2%** of bankroll | `UFC_BOT_MAX_BET_FRACTION=0.02` |
| Min stake fraction | 0.5% | `UFC_BOT_MIN_BET_FRACTION=0.005` |
| Daily loss limit | **5%** of day-open bankroll | `UFC_BOT_DAILY_LOSS_LIMIT=0.05` |
| Min edge to bet | 5% | `UFC_BOT_MIN_EDGE=0.05` |

Bets are skipped when odds are missing, edge is below threshold, or the daily loss limit is hit.

## Evaluation modes

| Mode | Flag | Meaning |
|------|------|---------|
| **frozen_model** | default `--backtest-2025` | Scores the production artifact. Weights may have seen the fights being scored. |
| **walk_forward_simplified** | `--backtest-2025 --walk-forward` | Past-only imputer + small LightGBM + optional isotonic. **Not** the production ensemble / interaction-discovery pipeline. |

Strict as-of pricing requires an **opening** quote with a timestamp ≤ the event cutoff. Closing-only, unlabeled, and label-only (no timestamp) prices are excluded from selection, sizing, and settlement. Reports show eligible vs excluded counts; zero eligible fights do not claim ROI.

**Feature chronology:** walk-forward scores a precomputed feature matrix; `FEATURE_COLUMNS` values are proven past-only via truncated-rebuild equality tests (not rebuild-per-fold). Historical profitability remains unverified until opening-line coverage exists.

Odds coverage: **Ultimate UFC dataset** (Kaggle mirror) + jansen88 historical lines + optional jerzyszocik daily snapshots, joined on event + date + fighter names. Merge persists opening/closing columns only when genuine pre-fight observation timestamps exist; unlabeled lines become closing (or stay unlabeled), never invented openings.

## CLI

```bash
python main.py merge-odds              # Merge odds into ufc-predictor/data/raw/fights.csv
python main.py backtest-2025           # frozen_model (default)
python main.py backtest-2025 --walk-forward   # walk_forward_simplified
python main.py backtest-2025 --year 2024
python main.py live                    # Dry-run signals for next card
```

## Next milestones

1. **Calibration** — keep WF isotonic / conformal on past-only holdouts; tighten live edge calc
2. **Opening-line coverage** — persist opening vs closing columns in the odds merge (strict as-of already refuses unlabeled)
3. **Live execution** — DraftKings / Odds API polling + bet slip export (no auto-bet yet)
4. **Per-bookmaker line shop** — best available price instead of averaged odds
5. **Alerting** — Telegram/discord when edge > threshold on upcoming main events
6. **Paper ledger** — track hypothetical vs actual fills with CLV (closing line value)

## Tests

```bash
cd ufc_betting_bot
python -m pytest tests/ -v
```
