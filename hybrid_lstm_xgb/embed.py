"""Build the tabular + Sequence Embedding design matrix for XGBoost."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from feature_normalize import normalize_features_torch
from lstm_encoder import LSTMEncoder
from windows import FileWindows, collect_origins, horizon_log_return, slice_history


def tabular_column_names(
    feature_list: Sequence[str],
    last_bar: bool = True,
    realized_vol: bool = True,
    calendar: bool = True,
) -> List[str]:
    names: List[str] = []
    if last_bar:
        names.extend([f"last_{c}" for c in feature_list])
    if realized_vol:
        names.append("realized_vol")
    if calendar:
        names.extend(["hour", "weekday"])
    return names


def embedding_column_names(hidden_size: int) -> List[str]:
    return [f"z_{i}" for i in range(hidden_size)]


def design_column_names(
    feature_list: Sequence[str],
    hidden_size: int,
    last_bar: bool = True,
    realized_vol: bool = True,
    calendar: bool = True,
    include_z: bool = True,
) -> List[str]:
    cols: List[str] = []
    if include_z:
        cols.extend(embedding_column_names(hidden_size))
    cols.extend(tabular_column_names(feature_list, last_bar, realized_vol, calendar))
    return cols


def _realized_vol(closes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.0
    # Guard non-positive prices before log.
    c = np.maximum(closes.astype(np.float64), 1e-12)
    rets = np.diff(np.log(c))
    return float(np.std(rets, ddof=0))


def build_tabular_row(
    hist_features_norm: np.ndarray,
    hist_stamps: np.ndarray,
    hist_closes_raw: np.ndarray,
    feature_list: Sequence[str],
    last_bar: bool,
    realized_vol: bool,
    calendar: bool,
) -> np.ndarray:
    """One row of the Tabular Block from a normalised history window."""
    parts: List[np.ndarray] = []
    if last_bar:
        parts.append(hist_features_norm[-1].astype(np.float32))
    if realized_vol:
        parts.append(np.array([_realized_vol(hist_closes_raw)], dtype=np.float32))
    if calendar:
        # stamps columns: minute, hour, weekday, day, month
        parts.append(hist_stamps[-1, [1, 2]].astype(np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def extract_design_matrix(
    encoder: LSTMEncoder,
    files: Sequence[FileWindows],
    region_name: str,
    lookback: int,
    horizon: int,
    stride: int,
    feature_list: Sequence[str],
    clip: float,
    device: torch.device,
    batch_size: int = 512,
    last_bar: bool = True,
    realized_vol: bool = True,
    calendar: bool = True,
    include_z: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Return (X, y, origin_pairs) for a named region.

    X columns = [z (optional) | tabular block]. y is the raw Horizon Return
    (not standardised). Encoder must be in eval mode.
    """
    if encoder.training:
        raise RuntimeError("extract_design_matrix requires encoder.eval()")

    pairs = collect_origins(files, region_name, lookback, horizon, stride)
    if not pairs:
        z_dim = encoder.hidden_size if include_z else 0
        tab_dim = len(tabular_column_names(feature_list, last_bar, realized_vol, calendar))
        return (
            np.zeros((0, z_dim + tab_dim), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )

    rows: List[np.ndarray] = []
    targets: List[float] = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        hist_batch = []
        stamp_batch = []
        close_batch = []
        y_batch = []
        for fi, origin in batch_pairs:
            fw = files[fi]
            hist, stamps = slice_history(fw, origin, lookback)
            hist_batch.append(hist)
            stamp_batch.append(stamps)
            close_batch.append(fw.closes[origin - lookback:origin].copy())
            y_batch.append(horizon_log_return(fw.closes, origin, horizon))

        x = torch.from_numpy(np.stack(hist_batch, axis=0)).to(device)
        x_norm = normalize_features_torch(x, list(feature_list), clip=clip)
        z = encoder.encode(x_norm).detach().cpu().numpy().astype(np.float32)

        x_norm_np = x_norm.detach().cpu().numpy()
        for bi, (fi, origin) in enumerate(batch_pairs):
            tab = build_tabular_row(
                hist_features_norm=x_norm_np[bi],
                hist_stamps=stamp_batch[bi],
                hist_closes_raw=close_batch[bi],
                feature_list=feature_list,
                last_bar=last_bar,
                realized_vol=realized_vol,
                calendar=calendar,
            )
            if include_z:
                row = np.concatenate([z[bi], tab], axis=0)
            else:
                row = tab
            rows.append(row)
            targets.append(y_batch[bi])

    X = np.stack(rows, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in design matrix for region={region_name}")
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite targets for region={region_name}")
    return X, y, pairs
