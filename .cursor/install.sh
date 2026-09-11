#!/usr/bin/env bash
# Cloud Agent bootstrap for the UFC Predictor project.
# Idempotent: safe to run repeatedly and against a cached/partial snapshot.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[install] system packages (tkinter for the CustomTkinter GUI + venv/build headers)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3-tk python3-venv python3-dev

echo "[install] python virtual environment (.venv)"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

echo "[install] python dependencies"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[install] local config (.env from template, never overwriting an existing one)"
if [ ! -f ".env" ]; then
  cp .env.example .env
fi

# Seed the public historical dataset and train the ensemble so the app is usable
# out of the box (preflight PASS, GUI "Model ready", backtests runnable).
# Fail-soft: if the public dataset host is unavailable the environment still
# works for the test suite and the GUI empty-cache state.
if [ ! -f "models/ensemble_winner.joblib" ]; then
  echo "[install] seeding historical fight data (public HuggingFace/GitHub CSV)"
  if python -c "from src.data_loader import load_historical_data as l; l(source='huggingface', incremental=False)"; then
    echo "[install] building features + training ensemble model"
    python scripts/rebuild_features_train.py || echo "[install] model training skipped (non-fatal)"
  else
    echo "[install] dataset download failed (non-fatal) — GUI and tests still work"
  fi
else
  echo "[install] model already present — skipping data/model seeding"
fi

echo "[install] done"
