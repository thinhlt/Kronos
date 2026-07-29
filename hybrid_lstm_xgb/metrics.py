"""Walk-forward metrics for the hybrid baseline (see research doc D10)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def metrics_from_windows(rows: Sequence[dict], horizon: int) -> Dict:
    """Compute error metrics on all rows; compounding metrics on every-h-th row."""
    empty = {
        "n_windows": 0,
        "n_windows_nonoverlap": 0,
        "mae": None,
        "rmse": None,
        "mape_pct": None,
        "minmax_rmse": None,
        "horizon_dir_acc": None,
        "strategy_total_return": None,
        "buy_hold_total_return": None,
        "avg_pred_return": None,
        "avg_actual_return": None,
    }
    if not rows:
        return empty

    frame = pd.DataFrame(list(rows))
    abs_err = (frame["pred_close_end"] - frame["actual_close_end"]).abs()
    pct_err = abs_err / frame["actual_close_end"].replace(0, np.nan)
    rmse = float(np.sqrt(((frame["pred_close_end"] - frame["actual_close_end"]) ** 2).mean()))
    actual_range = float(frame["actual_close_end"].max() - frame["actual_close_end"].min())
    minmax_rmse = float(rmse / actual_range) if actual_range > 0 else None

    pred_dir = np.sign(frame["pred_return"])
    actual_dir = np.sign(frame["actual_return"])
    dir_mask = actual_dir != 0
    if dir_mask.any():
        dir_acc = float((pred_dir[dir_mask] == actual_dir[dir_mask]).mean())
    else:
        dir_acc = None

    # Non-overlapping subsample for compounding (D7/D10).
    nonoverlap = frame.iloc[:: max(1, horizon)].reset_index(drop=True)
    strat_rets = nonoverlap["actual_return"].where(nonoverlap["pred_return"] > 0, 0.0)
    strategy_total = float((1.0 + strat_rets).prod() - 1.0) if len(nonoverlap) else None
    buy_hold_total = (
        float((1.0 + nonoverlap["actual_return"]).prod() - 1.0) if len(nonoverlap) else None
    )

    return {
        "n_windows": int(len(frame)),
        "n_windows_nonoverlap": int(len(nonoverlap)),
        "mae": float(abs_err.mean()),
        "rmse": rmse,
        "mape_pct": float(pct_err.mean() * 100.0),
        "minmax_rmse": minmax_rmse,
        "horizon_dir_acc": dir_acc,
        "strategy_total_return": strategy_total,
        "buy_hold_total_return": buy_hold_total,
        "avg_pred_return": float(frame["pred_return"].mean()),
        "avg_actual_return": float(frame["actual_return"].mean()),
    }


def reconstruct_close(entry_close: float, log_return: float) -> float:
    return float(entry_close * np.exp(log_return))
