"""
Inference-time predictor for the technical-indicator-augmented Kronos model.

Deliberately isolated from model.kronos.KronosPredictor: that class is shared
by finetune_csv/ and finetune/ and hardcodes the base 6-column feature set.
KronosPredictorTA instead accepts a `feature_list`/`enabled_indicators` pair
matching whatever the model was trained with, computes the same technical
indicators at inference time via `indicators.py`, and uses the same
open-anchor / z-score normalization as training (`feature_normalize.py`).
"""
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append('../')
from model.kronos import auto_regressive_inference, calc_time_stamps

from feature_normalize import denormalize_features, normalize_features
from indicators import ensure_features


class KronosPredictorTA:

    def __init__(self, model, tokenizer, feature_list, enabled_indicators, device=None, max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.feature_list = list(feature_list)
        self.enabled_indicators = dict(enabled_indicators)
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing_price = [c for c in ['open', 'high', 'low', 'close'] if c not in df.columns]
        if missing_price:
            raise ValueError(f"Price columns {missing_price} not found in DataFrame.")

        if 'volume' not in df.columns:
            df['volume'] = 0.0
            df['amount'] = 0.0
        if 'amount' not in df.columns and 'volume' in df.columns:
            df['amount'] = df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)

        df = ensure_features(df, self.feature_list, self.enabled_indicators)

        if df[self.feature_list].isnull().values.any():
            raise ValueError(
                "Input DataFrame contains NaN values in feature columns after indicator computation "
                "(likely too few rows for the indicators' warm-up period)."
            )
        return df

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        df = self._prepare_features(df)

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.feature_list].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x, norm_state = normalize_features(x, self.feature_list, clip=self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = denormalize_features(preds, self.feature_list, norm_state)

        pred_df = pd.DataFrame(preds, columns=self.feature_list, index=y_timestamp)
        return pred_df
