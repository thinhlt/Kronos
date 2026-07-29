"""Stage-2 XGBoost trainer + LSTM-only / XGB-only ablations (native xgboost API)."""
from __future__ import annotations

import omp_compat  # noqa: F401  — must precede torch/xgboost on macOS

import json
import os
from typing import List, Tuple

import numpy as np
import xgboost as xgb

from embed import design_column_names, extract_design_matrix, tabular_column_names
from train_lstm import load_lstm_encoder
from windows import build_all_files


def _xgb_params(config) -> dict:
    return {
        "max_depth": config.xgb_max_depth,
        "eta": config.xgb_eta,
        "subsample": config.xgb_subsample,
        "colsample_bytree": config.xgb_colsample_bytree,
        "lambda": config.xgb_reg_lambda,
        "gamma": config.xgb_gamma,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": config.seed,
        "nthread": max(1, config.num_workers),
    }


def _fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    config,
) -> Tuple[xgb.Booster, int]:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)
    booster = xgb.train(
        params=_xgb_params(config),
        dtrain=dtrain,
        num_boost_round=config.xgb_n_estimators,
        evals=[(ddev, "dev")],
        early_stopping_rounds=config.xgb_early_stopping_rounds,
        verbose_eval=False,
    )
    try:
        best_iter = int(booster.best_iteration)
    except AttributeError:
        best_iter = config.xgb_n_estimators - 1
    return booster, best_iter


def write_feature_spec(config, target_std: float, column_names: List[str]) -> str:
    paths = config.artifact_paths()
    spec = {
        "feature_list": list(config.feature_list),
        "enabled_indicators": dict(config.enabled_indicators),
        "d_in": config.d_in,
        "lookback_window": config.lookback_window,
        "horizon": config.horizon,
        "train_stride": config.train_stride,
        "backtest_stride": config.backtest_stride,
        "clip": config.clip,
        "lstm_fraction": config.lstm_fraction,
        "purge_bars": config.purge_bars,
        "lstm_dev_fraction": config.lstm_dev_fraction,
        "xgb_dev_fraction": config.xgb_dev_fraction,
        "hidden_size": config.lstm_hidden_size,
        "target_std": float(target_std),
        "design_columns": column_names,
        "tabular": {
            "last_bar_features": config.tabular_last_bar,
            "realized_vol": config.tabular_realized_vol,
            "calendar": config.tabular_calendar,
        },
        "ablations": {
            "lstm_only": config.ablation_lstm_only,
            "xgb_only": config.ablation_xgb_only,
        },
        "exp_name": config.exp_name,
    }
    with open(paths["feature_spec"], "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return paths["feature_spec"]


def train_xgb(config, files=None) -> dict:
    """Fit hybrid XGBoost on s2_fit embeddings; early-stop on s2_dev."""
    os.makedirs(config.save_dir, exist_ok=True)
    paths = config.artifact_paths()
    if not os.path.exists(paths["lstm_best"]):
        raise FileNotFoundError(
            f"Missing {paths['lstm_best']}; run train_lstm first or train_hybrid"
        )

    if files is None:
        files = build_all_files(config)

    device = config.device()
    encoder, ckpt = load_lstm_encoder(paths["lstm_best"], device)
    target_std = float(ckpt["target_std"])

    print("Extracting stage-2 fit design matrix...")
    X_fit, y_fit, _ = extract_design_matrix(
        encoder=encoder,
        files=files,
        region_name="s2_fit",
        lookback=config.lookback_window,
        horizon=config.horizon,
        stride=config.train_stride,
        feature_list=config.feature_list,
        clip=config.clip,
        device=device,
        batch_size=config.lstm_batch_size,
        last_bar=config.tabular_last_bar,
        realized_vol=config.tabular_realized_vol,
        calendar=config.tabular_calendar,
        include_z=True,
    )
    print("Extracting stage-2 Dev Tail design matrix...")
    X_dev, y_dev, _ = extract_design_matrix(
        encoder=encoder,
        files=files,
        region_name="s2_dev",
        lookback=config.lookback_window,
        horizon=config.horizon,
        stride=config.train_stride,
        feature_list=config.feature_list,
        clip=config.clip,
        device=device,
        batch_size=config.lstm_batch_size,
        last_bar=config.tabular_last_bar,
        realized_vol=config.tabular_realized_vol,
        calendar=config.tabular_calendar,
        include_z=True,
    )
    if len(y_fit) == 0 or len(y_dev) == 0:
        raise ValueError(
            f"Stage-2 empty: fit={len(y_fit)} dev={len(y_dev)}. "
            "Check lstm_fraction / purge / lookback."
        )
    print(f"X_fit={X_fit.shape} X_dev={X_dev.shape}")

    column_names = design_column_names(
        config.feature_list,
        config.lstm_hidden_size,
        config.tabular_last_bar,
        config.tabular_realized_vol,
        config.tabular_calendar,
        include_z=True,
    )

    print("Fitting hybrid XGBoost...")
    hybrid, best_iter = _fit_xgb(X_fit, y_fit, X_dev, y_dev, config)
    hybrid.save_model(paths["xgb_model"])
    print(f"  best_iteration={best_iter} -> {paths['xgb_model']}")

    result = {
        "xgb_model": paths["xgb_model"],
        "best_iteration": best_iter,
        "n_fit": int(len(y_fit)),
        "n_dev": int(len(y_dev)),
        "target_std": target_std,
    }

    if config.ablation_xgb_only:
        tab_names = tabular_column_names(
            config.feature_list,
            config.tabular_last_bar,
            config.tabular_realized_vol,
            config.tabular_calendar,
        )
        z_dim = config.lstm_hidden_size
        X_fit_tab = X_fit[:, z_dim:]
        X_dev_tab = X_dev[:, z_dim:]
        print(f"Fitting XGB-only ablation on {len(tab_names)} tabular columns...")
        xgb_only, xgb_only_iter = _fit_xgb(X_fit_tab, y_fit, X_dev_tab, y_dev, config)
        xgb_only.save_model(paths["xgb_only_model"])
        result["xgb_only_model"] = paths["xgb_only_model"]
        result["xgb_only_best_iteration"] = xgb_only_iter

    spec_path = write_feature_spec(config, target_std, column_names)
    result["feature_spec"] = spec_path
    print(f"Wrote {spec_path}")
    return result


if __name__ == "__main__":
    import argparse

    from config_loader import HybridConfig

    parser = argparse.ArgumentParser(description="Train stage-2 XGBoost")
    parser.add_argument("--config", type=str, default="configs/config_hybrid_5m_h12.yaml")
    args = parser.parse_args()
    cfg = HybridConfig(args.config)
    print(json.dumps(train_xgb(cfg), indent=2))
