"""Orchestrator: stage-1 LSTM -> stage-2 XGBoost (skip LSTM if artifact exists)."""
from __future__ import annotations

import omp_compat  # noqa: F401  — must precede torch/xgboost on macOS

import argparse
import json
import os

from config_loader import HybridConfig
from train_lstm import train_lstm
from train_xgb import train_xgb
from windows import build_all_files


def train_hybrid(config: HybridConfig, force_lstm: bool = False) -> dict:
    os.makedirs(config.save_dir, exist_ok=True)
    paths = config.artifact_paths()
    print(f"Experiment dir: {config.save_dir}")
    print(f"Data files ({len(config.data_paths)}):")
    for p in config.data_paths:
        print(f"  - {p}")

    files = build_all_files(config)
    result = {"save_dir": config.save_dir}

    if (not force_lstm) and os.path.exists(paths["lstm_best"]):
        print(f"[skip] Found existing {paths['lstm_best']}; skipping stage 1")
        result["lstm"] = {"skipped": True, "lstm_best": paths["lstm_best"]}
    else:
        print("=" * 60)
        print("Stage 1: LSTM")
        print("=" * 60)
        result["lstm"] = train_lstm(config, files=files)

    print("=" * 60)
    print("Stage 2: XGBoost")
    print("=" * 60)
    result["xgb"] = train_xgb(config, files=files)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid LSTM+XGBoost baseline")
    parser.add_argument("--config", type=str, default="configs/config_hybrid_5m_h12.yaml")
    parser.add_argument(
        "--force-lstm",
        action="store_true",
        help="Retrain LSTM even if lstm_best.pt already exists",
    )
    args = parser.parse_args()
    config = HybridConfig(args.config)
    result = train_hybrid(config, force_lstm=args.force_lstm)
    # Drop bulky history from stdout summary.
    slim = {
        "save_dir": result["save_dir"],
        "lstm": {k: v for k, v in result["lstm"].items() if k != "history"},
        "xgb": result["xgb"],
    }
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
