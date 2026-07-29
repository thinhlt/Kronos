"""Config loader for the hybrid LSTM + XGBoost baseline pipeline."""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Union

import yaml

from indicators import DEFAULT_ENABLED_INDICATORS, build_feature_list

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_data_paths(data_path: Union[str, List[str]], base_dir: str | None = None) -> List[str]:
    """Normalize data_path (single path, glob, or list) into concrete file paths."""
    if data_path is None:
        return []

    raw_entries = data_path if isinstance(data_path, list) else [data_path]
    base = base_dir or SCRIPT_DIR

    resolved: List[str] = []
    for entry in raw_entries:
        path = entry if os.path.isabs(entry) else os.path.normpath(os.path.join(base, entry))
        if any(ch in path for ch in "*?["):
            matches = sorted(glob.glob(path))
            if not matches:
                raise FileNotFoundError(f"Glob pattern matched no files: {path}")
            resolved.extend(matches)
        else:
            resolved.append(path)

    missing = [p for p in resolved if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"data_path entries not found: {missing}")

    seen = set()
    deduped = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


class HybridConfig:
    """Materialises the Gate-4 YAML schema for training and backtest."""

    def __init__(self, config_path: str, data_path_override: List[str] | None = None):
        if not os.path.isabs(config_path):
            config_path = os.path.join(SCRIPT_DIR, config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        self.config_path = config_path
        self.raw = raw

        data = raw.get("data", {})
        features = raw.get("features", {})
        lstm = raw.get("lstm", {})
        xgb = raw.get("xgboost", {})
        experiment = raw.get("experiment", {})
        device = raw.get("device", {})

        self.lookback_window = int(data.get("lookback_window", 128))
        self.horizon = int(data.get("horizon", 12))
        self.train_stride = int(data.get("train_stride", self.horizon))
        self.backtest_stride = int(data.get("backtest_stride", 1))
        self.clip = float(data.get("clip", 5.0))
        self.train_ratio = float(data.get("train_ratio", 0.9))
        self.val_ratio = float(data.get("val_ratio", 0.1))
        self.test_ratio = float(data.get("test_ratio", 0.0))
        self.lstm_fraction = float(data.get("lstm_fraction", 0.75))
        self.purge_bars = int(data.get("purge_bars", self.horizon))

        enabled = dict(DEFAULT_ENABLED_INDICATORS)
        enabled.update(features.get("indicators") or {})
        self.enabled_indicators = enabled
        self.feature_list = build_feature_list(enabled)
        self.d_in = len(self.feature_list)

        tabular = features.get("tabular") or {}
        self.tabular_last_bar = bool(tabular.get("last_bar_features", True))
        self.tabular_realized_vol = bool(tabular.get("realized_vol", True))
        self.tabular_calendar = bool(tabular.get("calendar", True))

        self.lstm_hidden_size = int(lstm.get("hidden_size", 64))
        self.lstm_num_layers = int(lstm.get("num_layers", 1))
        self.lstm_dropout = float(lstm.get("dropout", 0.0))
        self.lstm_epochs = int(lstm.get("epochs", 20))
        self.lstm_batch_size = int(lstm.get("batch_size", 512))
        self.lstm_learning_rate = float(lstm.get("learning_rate", 1e-3))
        self.lstm_weight_decay = float(lstm.get("weight_decay", 0.01))
        self.lstm_dev_fraction = float(lstm.get("dev_fraction", 0.1))
        self.lstm_early_stopping_patience = int(lstm.get("early_stopping_patience", 3))

        self.xgb_max_depth = int(xgb.get("max_depth", 6))
        self.xgb_eta = float(xgb.get("eta", 0.05))
        self.xgb_subsample = float(xgb.get("subsample", 0.8))
        self.xgb_colsample_bytree = float(xgb.get("colsample_bytree", 0.8))
        self.xgb_reg_lambda = float(xgb.get("reg_lambda", 1.0))
        self.xgb_gamma = float(xgb.get("gamma", 0.0))
        self.xgb_n_estimators = int(xgb.get("n_estimators", 2000))
        self.xgb_early_stopping_rounds = int(xgb.get("early_stopping_rounds", 50))
        self.xgb_dev_fraction = float(xgb.get("dev_fraction", 0.15))

        self.exp_name = experiment.get("exp_name", "hybrid_5m_h12")
        base_path = experiment.get("base_path", "./finetuned/")
        if not os.path.isabs(base_path):
            base_path = os.path.normpath(os.path.join(SCRIPT_DIR, base_path))
        self.base_path = base_path
        self.save_dir = os.path.join(self.base_path, self.exp_name)
        ablations = experiment.get("ablations") or {}
        self.ablation_lstm_only = bool(ablations.get("lstm_only", True))
        self.ablation_xgb_only = bool(ablations.get("xgb_only", True))

        self.use_cuda = bool(device.get("use_cuda", True))
        self.device_id = int(device.get("device_id", 0))
        self.seed = int(raw.get("seed", 42))
        self.num_workers = int(raw.get("num_workers", 4))

        if data_path_override is not None:
            self.data_paths = list(data_path_override)
        else:
            self.data_paths = resolve_data_paths(data.get("data_path"), base_dir=SCRIPT_DIR)

        if not self.data_paths:
            raise ValueError("HybridConfig requires at least one data path")
        if not (0.0 < self.lstm_fraction < 1.0):
            raise ValueError(f"lstm_fraction must be in (0, 1), got {self.lstm_fraction}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.lookback_window < 1:
            raise ValueError(f"lookback_window must be >= 1, got {self.lookback_window}")

    def device(self):
        import torch

        if self.use_cuda and torch.cuda.is_available():
            return torch.device(f"cuda:{self.device_id}")
        if self.use_cuda and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def artifact_paths(self) -> Dict[str, str]:
        return {
            "save_dir": self.save_dir,
            "lstm_best": os.path.join(self.save_dir, "lstm_best.pt"),
            "xgb_model": os.path.join(self.save_dir, "xgb_model.json"),
            "xgb_only_model": os.path.join(self.save_dir, "xgb_only_model.json"),
            "feature_spec": os.path.join(self.save_dir, "feature_spec.json"),
        }
