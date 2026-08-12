# Odds empty incident (2026-08-11)

## Symptoms

Dashboard book tabs showed **0/N** / no lines after Soft Update or Quick Odds + Props
(BetNow / DraftKings / MyBookie / Odds API).

## Root cause

Not the model and not suspect-edge blanking of good lines.

1. **Code bug (primary):** In `_load_book_odds` (`dashboard_service.py`), a *local*
   `from src.predictor import merge_predictions_with_odds` inside the soft-fail
   `except` made Python treat that name as local for the **entire** function.
   The happy path then raised `UnboundLocalError: cannot access local variable
   'merge_predictions_with_odds'…`, which was caught and surfaced as
   `NO BET — fail-closed` / book unavailable — **every book matched 0** even when
   Odds API / MyBookie caches had real lines.

2. **Config (secondary):** `BETNOW_ENABLED=false`, `DRAFTKINGS_ENABLED=false` in
   runtime `.env` — those scrapers are not loaded (Odds API + MyBookie only when
   MyBookie is on). `BETNOW_COOKIE` is a placeholder; session may still hit login.

3. **Stale slate (secondary):** Old background snapshot (Belgrade, ~2026-07-26)
   vs current Odds API / MyBookie cards causes `name_mismatch` 0/N when that
   snapshot is used. Startup already skips mismatched / aged snapshots; Refresh
   Next Two is required when the loaded card is wrong.

## Fix

- Removed the shadowing local import; soft-fail uses the module-level
  `merge_predictions_with_odds`.
- Clearer per-book warnings (MyBookie name_mismatch; disabled-book UI hints).
- Quick Odds logs masked `odds_env` + `odds_status book=… matched=…` lines.

## Verify

- Quick Odds on current upcoming cards: Odds API / MyBookie show matched lines
  (or an honest warning), not `UnboundLocalError`.
- `pytest tests/test_odds_reliability.py` passes.
- Reliability guards unchanged (still blank unmatched / |edge| > 30% only).

## Operator notes

| Check | Action |
|--------|--------|
| `THE_ODDS_API_KEY` in `dist\.env` / project `.env` | Real key (EXE loads `dist\.env`) |
| BetNow / DK tabs | `BETNOW_ENABLED=true` / `DRAFTKINGS_ENABLED=true` + real session |
| Wrong card | Refresh Next Two, then Soft Update |
| `--debug` | Look for `odds_status book=` / `401` / `auth_mode=` |
