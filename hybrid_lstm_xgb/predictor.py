"""Inference wrapper for the hybrid LSTM + XGBoost baseline."""
from __future__ import annotations

import omp_compat  # noqa: F401  — must precede torch/xgboost on macOS

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import xgboost as xgb

from embed import build_tabular_row, design_column_names
from feature_normalize import normalize_features_torch
from metrics import reconstruct_close
from train_lstm import load_lstm_encoder


class HybridPredictor:
    """Predict Horizon Return / reconstructed close from a lookback history window."""

    def __init__(
        self,
        save_dir: str,
        device: Optional[torch.device] = None,
        load_ablations: bool = True,
    ):
        self.save_dir = save_dir
        spec_path = os.path.join(save_dir, "feature_spec.json")
        lstm_path = os.path.join(save_dir, "lstm_best.pt")
        xgb_path = os.path.join(save_dir, "xgb_model.json")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Missing feature_spec.json in {save_dir}")
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Missing lstm_best.pt in {save_dir}")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"Missing xgb_model.json in {save_dir}")

        with open(spec_path, "r", encoding="utf-8") as f:
            self.spec: Dict = json.load(f)

        self.feature_list: List[str] = list(self.spec["feature_list"])
        self.lookback = int(self.spec["lookback_window"])
        self.horizon = int(self.spec["horizon"])
        self.clip = float(self.spec["clip"])
        self.target_std = float(self.spec["target_std"])
        self.hidden_size = int(self.spec["hidden_size"])
        tabular = self.spec.get("tabular") or {}
        self.last_bar = bool(tabular.get("last_bar_features", True))
        self.realized_vol = bool(tabular.get("realized_vol", True))
        self.calendar = bool(tabular.get("calendar", True))

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        self.encoder, _ = load_lstm_encoder(lstm_path, device)
        self.xgb = xgb.Booster()
        self.xgb.load_model(xgb_path)

        self.xgb_only = None
        xgb_only_path = os.path.join(save_dir, "xgb_only_model.json")
        if load_ablations and os.path.exists(xgb_only_path):
            self.xgb_only = xgb.Booster()
            self.xgb_only.load_model(xgb_only_path)

        expected = design_column_names(
            self.feature_list,
            self.hidden_size,
            self.last_bar,
            self.realized_vol,
            self.calendar,
            include_z=True,
        )
        if self.spec.get("design_columns") and list(self.spec["design_columns"]) != expected:
            raise ValueError(
                "feature_spec design_columns mismatch with current tabular settings; "
                "retrain or fix feature_spec.json"
            )

    def _validate_feature_list(self, feature_list: Sequence[str]) -> None:
        if list(feature_list) != self.feature_list:
            raise ValueError(
                f"Feature list mismatch.\n  expected={self.feature_list}\n  got={list(feature_list)}"
            )

    @torch.no_grad()
    def predict_log_returns(
        self,
        hist_features: np.ndarray,
        hist_stamps: np.ndarray,
        hist_closes: np.ndarray,
        feature_list: Sequence[str],
        models: Sequence[str] = ("hybrid",),
    ) -> Dict[str, float]:
        """Predict log returns for one history window.

        hist_features: (lookback, F) raw (unnormalised) features
        hist_stamps: (lookback, 5) minute/hour/weekday/day/month
        hist_closes: (lookback,) raw closes (for realized vol)
        """
        self._validate_feature_list(feature_list)
        if hist_features.shape != (self.lookback, len(self.feature_list)):
            raise ValueError(
                f"Expected hist_features shape {(self.lookback, len(self.feature_list))}, "
                f"got {hist_features.shape}"
            )

        x = torch.from_numpy(np.asarray(hist_features, dtype=np.float32)).unsqueeze(0).to(self.device)
        x_norm = normalize_features_torch(x, self.feature_list, clip=self.clip)
        z = self.encoder.encode(x_norm).detach().cpu().numpy().astype(np.float32)[0]
        tab = build_tabular_row(
            hist_features_norm=x_norm.detach().cpu().numpy()[0],
            hist_stamps=np.asarray(hist_stamps, dtype=np.float32),
            hist_closes_raw=np.asarray(hist_closes, dtype=np.float32),
            feature_list=self.feature_list,
            last_bar=self.last_bar,
            realized_vol=self.realized_vol,
            calendar=self.calendar,
        )

        out: Dict[str, float] = {}
        for name in models:
            if name == "hybrid":
                row = np.concatenate([z, tab], axis=0).reshape(1, -1)
                out[name] = float(self.xgb.predict(xgb.DMatrix(row))[0])
            elif name == "lstm_only":
                # Stage-1 head predicts standardised return.
                pred_std = float(self.encoder(x_norm).detach().cpu().numpy()[0])
                out[name] = pred_std * self.target_std
            elif name == "xgb_only":
                if self.xgb_only is None:
                    raise RuntimeError("xgb_only model was not loaded")
                out[name] = float(self.xgb_only.predict(xgb.DMatrix(tab.reshape(1, -1)))[0])
            else:
                raise ValueError(f"Unknown model name: {name}")
        return out

    def predict_close_end(
        self,
        hist_features: np.ndarray,
        hist_stamps: np.ndarray,
        hist_closes: np.ndarray,
        entry_close: float,
        feature_list: Sequence[str],
        models: Sequence[str] = ("hybrid",),
    ) -> Dict[str, Dict[str, float]]:
        logrets = self.predict_log_returns(
            hist_features, hist_stamps, hist_closes, feature_list, models=models
        )
        return {
            name: {
                "pred_log_return": lr,
                "pred_return": float(np.exp(lr) - 1.0),
                "pred_close_end": reconstruct_close(entry_close, lr),
            }
            for name, lr in logrets.items()
        }
