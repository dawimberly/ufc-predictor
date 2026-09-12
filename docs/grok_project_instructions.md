# Grok Project setup — UFC Predictor

Copy each section below into the matching field of your Grok Project
(**Instructions**, **Your Automations**, **Files**).

> Note: Grok automations run inside Grok (web-connected), not on your machine —
> they cannot execute this repo. Use them for prep, research, and reminders.

---

## 1) Instructions

Paste this into the Project **Instructions** field:

```
You help me develop "UFC Predictor" — a standalone Python 3.12 UFC fight
prediction + high-accuracy (HA) betting-signal bot. Not part of any other project.

STACK: lightgbm/xgboost ensemble (calibrated), pandas/numpy/scikit-learn, shap;
CustomTkinter desktop GUI (src/ufc_dashboard.py); CLI (main.py, src/cli_entry.py);
pytest tests. Config in config.py from .env. data/ and models/ are gitignored and
generated locally, not shipped.

RUN: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
&& cp .env.example .env; preflight: python scripts/preflight.py; backtest:
python main.py --backtest; predict: python -m src.cli_entry --next-two --odds;
GUI: python src/ufc_dashboard.py (run from repo root); tests: python -m pytest tests/ -q.

NARRATIVE ENGINE: despite grok_* names, the live analysis engine is LOCAL OLLAMA
(default qwen2.5-coder:7b), NOT xAI. query_grok/analyze_card_with_grok route to
Ollama; GROK_API_BASE (api.x.ai) is defined but never called. xAI keys only drive
narrative Kelly-tilt bounds today.

RULES (never violate):
- Fail-closed: no sized bet without usable odds + uncertainty clearance;
  no odds => "NO BET — no usable odds (fail-closed)".
- Color tiers sort Deep Blue > Sky Blue > Green > Yellow > Red:
  Deep Blue=BET THIS (real $), Sky Blue=TINY PAPER BET (paper only),
  Green=FUN ONLY ($0), Yellow=CAUTION/SKIP ($0), Red=DO NOT BET.
  Only Deep/Sky Blue are money tickets; never treat Green as bankroll.
- Paper (default) vs Live (hard USD caps, wide-CI fail-closed) via UFC_PROFILE.
- Features must be leakage-safe (rolling stats use only prior fights).
- LLM narrative may only scale stake within Kelly bounds — never change pick,
  edge, or probability, and never invent bets.
- Odds are quota-safe: ODDS_FETCH_ONCE=true reuses cache; keep DraftKings off.

STYLE: prefer editing existing modules over new files; match existing style;
after model/strategy changes, recommend running pytest + a backtest.
```

---

## 2) Your Automations

Add these as separate scheduled automations (research/reminders only — Grok
cannot run the bot):

- `Every Thursday 9am: list the upcoming UFC card for this weekend (main + prelims), with each bout, weight class, and a one-line styles/matchup note. Flag fights worth researching.`
- `Every Saturday 8am: give me a pre-card research brief for tonight's UFC event — recent form, injuries/short-notice changes, and any news that could move lines. No betting advice, research only.`
- `Every Monday 9am: summarize last weekend's UFC results (winners + method/round) so I can settle and score my predictions.`
- `Monthly: check for notable releases or breaking changes in lightgbm, xgboost, scikit-learn, and the Ollama models I use (qwen2.5-coder), and flag anything that could affect the bot.`

---

## 3) Files

Upload these repo files (highest signal, low noise):

- `README.md` — full feature/tab/config reference
- `CANONICAL.md` — canonical paths + launch notes
- `.env.example` — every config knob with defaults
- `config.py` — authoritative settings + profile logic
- `requirements.txt` — dependency pins
- Optional for sizing/gating work: `src/bet_tiers.py`, `src/high_accuracy_strategy.py`, `src/uncertainty_gates.py`

Do not bulk-upload `src/` — the big files (`ufc_dashboard.py` ~300KB, `strategy.py`
~98KB, `feature_engineering.py` ~96KB) eat context. Paste only the relevant
function when working on those.
