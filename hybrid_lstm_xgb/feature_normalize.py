"""
Feature normalization shared by training (CustomKlineDataset) and inference
(KronosPredictorTA).

Groups (only columns present in `feature_list` are touched):

- Open-anchored (divide by window's first open): OHLC, Heikin-Ashi, Bollinger, ATR
- Open-anchored then per-window z-score: MACD / MACDh / MACDs
- Per-window z-score only: KDJ, volume, amount, volume SMA

Anchor is always ``open[0]`` of the array passed in (full window at train time,
lookback history at predict time).

Training applies ``normalize_features_torch`` on GPU after the batch is moved to
device; ``normalize_features`` is a numpy convenience wrapper (same math) used
by inference.
"""
from __future__ import annotations

import numpy as np
import torch

from indicators import (
    ATR_COLUMNS,
    BOLLINGER_COLUMNS,
    HEIKIN_ASHI_COLUMNS,
    KDJ_COLUMNS,
    MACD_COLUMNS,
    VOLUME_SMA_COLUMNS,
)

_PRICE_ANCHOR_COLUMNS = (
    ['open', 'high', 'low', 'close']
    + list(HEIKIN_ASHI_COLUMNS)
    + list(BOLLINGER_COLUMNS)
    + list(ATR_COLUMNS)
)
_ZSCORE_ONLY_COLUMNS = (
    ['volume', 'amount']
    + list(KDJ_COLUMNS)
    + list(VOLUME_SMA_COLUMNS)
)


def _indices(feature_list: list, names: list | tuple) -> np.ndarray:
    name_set = set(names)
    return np.array([i for i, c in enumerate(feature_list) if c in name_set], dtype=np.int64)


def _group_indices(feature_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    price_idx = _indices(feature_list, _PRICE_ANCHOR_COLUMNS)
    macd_idx = _indices(feature_list, MACD_COLUMNS)
    z_idx = _indices(feature_list, _ZSCORE_ONLY_COLUMNS)

    known = set(price_idx.tolist()) | set(macd_idx.tolist()) | set(z_idx.tolist())
    leftover = np.array([i for i in range(len(feature_list)) if i not in known], dtype=np.int64)
    if leftover.size:
        z_idx = np.concatenate([z_idx, leftover]) if z_idx.size else leftover

    return price_idx, macd_idx, z_idx


def _zscore_indices(macd_idx: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
    if macd_idx.size and z_idx.size:
        return np.concatenate([macd_idx, z_idx])
    return macd_idx if macd_idx.size else z_idx


def normalize_features_torch(
    x: torch.Tensor,
    feature_list: list,
    clip: float = 5.0,
    return_state: bool = False,
):
    """Normalize a (T, F) or (B, T, F) feature tensor (device-agnostic / GPU-friendly).

    Returns
    -------
    x_norm : torch.Tensor
        Normalized features, same shape as `x`.
    state : dict, optional
        Only when ``return_state=True``. Parameters needed by
        ``denormalize_features`` (numpy indices / stats when squeezed).
    """
    if 'open' not in feature_list:
        raise ValueError("feature_list must include 'open' for open-anchor normalization")

    squeezed = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeezed = True
    elif x.ndim != 3:
        raise ValueError(f"Expected x shape (T, F) or (B, T, F), got {tuple(x.shape)}")

    if x.shape[-1] != len(feature_list):
        raise ValueError(
            f"Expected last dim {len(feature_list)}, got {x.shape[-1]}"
        )

    open_idx = feature_list.index('open')
    price_idx, macd_idx, z_idx = _group_indices(feature_list)
    zscore_idx = _zscore_indices(macd_idx, z_idx)

    device = x.device
    dtype = x.dtype
    price_t = torch.as_tensor(price_idx, device=device, dtype=torch.long)
    macd_t = torch.as_tensor(macd_idx, device=device, dtype=torch.long)
    zscore_t = torch.as_tensor(zscore_idx, device=device, dtype=torch.long)

    out = x.clone()
    bsz = out.shape[0]

    open_anchor = out[:, 0, open_idx].clone()
    tiny = torch.full_like(open_anchor, 1e-12)
    open_anchor = torch.where(
        open_anchor.abs() < 1e-12,
        torch.where(open_anchor >= 0, tiny, -tiny),
        open_anchor,
    )
    anchor = open_anchor.view(bsz, 1, 1)

    if price_t.numel():
        out[:, :, price_t] = out[:, :, price_t] / anchor
    if macd_t.numel():
        out[:, :, macd_t] = out[:, :, macd_t] / anchor

    mean = torch.zeros(bsz, len(feature_list), device=device, dtype=dtype)
    std = torch.ones(bsz, len(feature_list), device=device, dtype=dtype)

    if zscore_t.numel():
        vals = out[:, :, zscore_t]
        # Match numpy np.std default (ddof=0): population std.
        m = vals.mean(dim=1)
        s = vals.std(dim=1, unbiased=False)
        mean[:, zscore_t] = m
        std[:, zscore_t] = s
        out[:, :, zscore_t] = (vals - m.unsqueeze(1)) / (s.unsqueeze(1) + 1e-5)

    out = torch.clamp(out, -clip, clip)

    if not return_state:
        return out.squeeze(0) if squeezed else out

    if squeezed:
        state = {
            'open_anchor': float(open_anchor[0].item()),
            'mean': mean[0].detach().cpu().numpy().astype(np.float32),
            'std': std[0].detach().cpu().numpy().astype(np.float32),
            'price_idx': price_idx,
            'macd_idx': macd_idx,
            'zscore_idx': zscore_idx,
        }
        return out.squeeze(0), state

    state = {
        'open_anchor': open_anchor,
        'mean': mean,
        'std': std,
        'price_idx': price_idx,
        'macd_idx': macd_idx,
        'zscore_idx': zscore_idx,
    }
    return out, state


def normalize_features(x: np.ndarray, feature_list: list, clip: float = 5.0):
    """Normalize a (T, F) feature matrix (numpy wrapper over the torch path).

    Returns
    -------
    x_norm : np.ndarray
        Normalized features, same shape as `x`.
    state : dict
        Parameters needed by `denormalize_features` (open anchor, z-score stats,
        column index groups).
    """
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[1] != len(feature_list):
        raise ValueError(
            f"Expected x shape (T, {len(feature_list)}), got {getattr(x_np, 'shape', None)}"
        )
    out, state = normalize_features_torch(
        torch.from_numpy(x_np), feature_list, clip=clip, return_state=True
    )
    return out.numpy(), state


def denormalize_features(preds: np.ndarray, feature_list: list, state: dict) -> np.ndarray:
    """Invert `normalize_features` for model outputs of shape (T, F) or (F,)."""
    out = np.asarray(preds, dtype=np.float32).copy()
    squeeze = False
    if out.ndim == 1:
        out = out[np.newaxis, :]
        squeeze = True
    if out.shape[-1] != len(feature_list):
        raise ValueError(
            f"Expected last dim {len(feature_list)}, got {out.shape[-1]}"
        )

    open_anchor = state['open_anchor']
    mean = state['mean']
    std = state['std']
    price_idx = state['price_idx']
    macd_idx = state['macd_idx']
    zscore_idx = state['zscore_idx']

    if zscore_idx.size:
        out[:, zscore_idx] = out[:, zscore_idx] * (std[zscore_idx] + 1e-5) + mean[zscore_idx]
    if macd_idx.size:
        out[:, macd_idx] = out[:, macd_idx] * open_anchor
    if price_idx.size:
        out[:, price_idx] = out[:, price_idx] * open_anchor

    return out.squeeze(0) if squeeze else out
