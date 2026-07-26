"""Rebuild features from cached fights.csv and train (schema bump path)."""

from __future__ import annotations

import sys

import config
from src.data_loader import ensure_data_dirs, load_fights
from src.feature_engineering import build_feature_matrix, save_features
from src.model_trainer import train_model


def main() -> int:
    ensure_data_dirs()
    print("Loading fights...", flush=True)
    fights = load_fights()
    print(f"fights={len(fights)}", flush=True)
    print(f"Building features (schema {config.FEATURE_SCHEMA_VERSION})...", flush=True)
    features = build_feature_matrix(fights)
    path = save_features(features)
    print(f"Saved {len(features)} rows -> {path}", flush=True)
    print("Training...", flush=True)
    result = train_model(features, tune="none", calibration_method=config.CALIBRATION_METHOD)
    print(
        f"AUC={result.metrics.get('roc_auc')} acc={result.metrics.get('accuracy')}",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
