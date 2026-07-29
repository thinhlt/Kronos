"""
Walk-forward backtest for finetuned Kronos TA models on prepared CSV data.

For each data file, runs non-overlapping (by default) forecast windows on the
validation split (last `val_ratio` of bars, matching training), compares
predicted close to actual close, and optionally simulates a simple long/flat
strategy. Reuses `predict.py` model loading and data prep helpers.

Usage:
    python backtest.py --config configs/config_multi_symbol_5m_ta.yaml \\
        --input data/*_kline_5min.csv
    python backtest_multi_symbol_5m_ta.py
    python backtest_multi_symbol_5m_ta_norm.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append("../")

from config_loader import CustomFinetuneConfig
from indicators import BASE_FEATURES
from predict import (
    SCRIPT_DIR,
    build_predictor,
    infer_interval_minutes,
    load_prepared_dataframe,
)
from predictor import KronosPredictorTA

DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "backtests")
DEFAULT_LOCAL_DATA_GLOB = os.path.join(SCRIPT_DIR, "data", "*_kline_5min.csv")


def resolve_default_inputs(explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    matches = sorted(glob.glob(DEFAULT_LOCAL_DATA_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No local CSVs matched {DEFAULT_LOCAL_DATA_GLOB}. "
            "Pass --input or place prepared *_kline_5min.csv files under data/."
        )
    return matches


def validation_start_index(n_rows: int, train_ratio: float) -> int:
    """First index of the validation region (same per-file split as training)."""
    return int(n_rows * train_ratio)


def window_starts(
    n_rows: int,
    lookback: int,
    pred_len: int,
    val_start: int,
    stride: int,
    max_windows: int | None,
) -> list[int]:
    """Forecast origins `i` where history is [i-lookback, i) and target is [i, i+pred_len)."""
    first = max(lookback, val_start)
    last = n_rows - pred_len
    if first > last:
        return []
    starts = list(range(first, last + 1, stride))
    if max_windows is not None and max_windows > 0:
        starts = starts[:max_windows]
    return starts


def filter_starts_last_days(
    df: pd.DataFrame,
    starts: list[int],
    last_days: float | None,
) -> list[int]:
    """Keep origins whose forecast origin timestamp falls in the last `last_days`."""
    if last_days is None or last_days <= 0 or not starts:
        return starts
    stamps = pd.to_datetime(df["timestamps"])
    cutoff = stamps.iloc[-1] - pd.Timedelta(days=last_days)
    return [i for i in starts if stamps.iloc[i - 1] >= cutoff]


def _format_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")


def predict_window(
    predictor: KronosPredictorTA,
    df: pd.DataFrame,
    start_idx: int,
    lookback: int,
    pred_len: int,
    feature_list: list[str],
    T: float,
    top_p: float,
    sample_count: int,
) -> pd.DataFrame:
    hist = df.iloc[start_idx - lookback:start_idx].reset_index(drop=True)
    future = df.iloc[start_idx:start_idx + pred_len].reset_index(drop=True)

    x_df = hist[BASE_FEATURES].copy()
    for col in feature_list:
        if col not in x_df.columns and col in hist.columns:
            x_df[col] = hist[col].values

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=hist["timestamps"],
        y_timestamp=future["timestamps"],
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    return pred_df


def metrics_from_windows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_windows": 0,
            "mae": None,
            "rmse": None,
            "mape_pct": None,
            "horizon_dir_acc": None,
            "strategy_total_return": None,
            "buy_hold_total_return": None,
            "avg_pred_return": None,
            "avg_actual_return": None,
        }

    frame = pd.DataFrame(rows)
    abs_err = (frame["pred_close_end"] - frame["actual_close_end"]).abs()
    pct_err = abs_err / frame["actual_close_end"].replace(0, np.nan)

    pred_dir = np.sign(frame["pred_return"])
    actual_dir = np.sign(frame["actual_return"])
    # Flat actual moves do not count as hits or misses.
    dir_mask = actual_dir != 0
    if dir_mask.any():
        dir_acc = float((pred_dir[dir_mask] == actual_dir[dir_mask]).mean())
    else:
        dir_acc = None

    # Long when model forecasts positive close return over the horizon; else flat.
    strat_rets = frame["actual_return"].where(frame["pred_return"] > 0, 0.0)
    strategy_total = float((1.0 + strat_rets).prod() - 1.0)
    buy_hold_total = float((1.0 + frame["actual_return"]).prod() - 1.0)

    return {
        "n_windows": int(len(frame)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(((frame["pred_close_end"] - frame["actual_close_end"]) ** 2).mean())),
        "mape_pct": float(pct_err.mean() * 100.0),
        "horizon_dir_acc": dir_acc,
        "strategy_total_return": strategy_total,
        "buy_hold_total_return": buy_hold_total,
        "avg_pred_return": float(frame["pred_return"].mean()),
        "avg_actual_return": float(frame["actual_return"].mean()),
    }


def plot_symbol_backtest(
    windows: pd.DataFrame,
    symbol: str,
    output_dir: str,
    actual_close: pd.Series | None = None,
) -> str | None:
    if windows.empty:
        return None

    plot_df = windows.copy()
    plot_df["origin_time"] = pd.to_datetime(plot_df["origin_time"])
    plot_df["horizon_end_time"] = pd.to_datetime(plot_df["horizon_end_time"])
    plot_df = plot_df.sort_values("horizon_end_time")

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("white")

    ax1 = axes[0]
    # Continuous actual close (bar-by-bar) so the price path is visible.
    if actual_close is not None and len(actual_close) > 0:
        series = actual_close.copy()
        series.index = pd.to_datetime(series.index)
        t0 = plot_df["origin_time"].min()
        t1 = plot_df["horizon_end_time"].max()
        series = series.loc[(series.index >= t0) & (series.index <= t1)]
        ax1.plot(
            series.index,
            series.values,
            label="Actual close",
            color="#1f77b4",
            linewidth=1.8,
            zorder=3,
        )
    else:
        ax1.plot(
            plot_df["horizon_end_time"],
            plot_df["actual_close_end"],
            label="Actual close (horizon end)",
            color="#1f77b4",
            linewidth=1.8,
            zorder=3,
        )
    ax1.plot(
        plot_df["horizon_end_time"],
        plot_df["pred_close_end"],
        label="Predicted close (horizon end)",
        color="#ff7f0e",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    ax1.set_ylabel("Close")
    ax1.set_title(f"{symbol} walk-forward close (actual vs predicted)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    cum_strat = (1.0 + plot_df["actual_return"].where(plot_df["pred_return"] > 0, 0.0)).cumprod()
    cum_bh = (1.0 + plot_df["actual_return"]).cumprod()
    ax2.plot(plot_df["origin_time"], cum_bh, label="Buy & hold (per window)", color="#1f77b4")
    ax2.plot(plot_df["origin_time"], cum_strat, label="Long-if-pred-up", color="#2ca02c")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cumulative growth")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    _format_time_axis(ax2)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{symbol}_backtest.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def backtest_one_file(
    data_path: str,
    predictor: KronosPredictorTA,
    config: CustomFinetuneConfig,
    output_root: str,
    lookback: int,
    pred_len: int,
    stride: int,
    max_windows: int | None,
    T: float,
    top_p: float,
    sample_count: int,
    last_days: float | None = 3.0,
) -> dict:
    symbol = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(output_root, symbol)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Backtesting: {symbol}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 60}")

    df = load_prepared_dataframe(data_path, config.feature_list, config.enabled_indicators)
    n = len(df)
    val_start = validation_start_index(n, config.train_ratio)
    starts = window_starts(n, lookback, pred_len, val_start, stride, max_windows)
    starts = filter_starts_last_days(df, starts, last_days)

    if not starts:
        raise ValueError(
            f"{symbol}: no usable windows (rows={n}, lookback={lookback}, "
            f"pred_len={pred_len}, val_start={val_start}, stride={stride}, "
            f"last_days={last_days})"
        )

    interval_minutes = infer_interval_minutes(df["timestamps"])
    print(
        f"rows={n} | val_start={val_start} ({df['timestamps'].iloc[val_start]}) | "
        f"windows={len(starts)} | lookback={lookback} pred_len={pred_len} stride={stride} | "
        f"interval={interval_minutes}m | last_days={last_days}"
    )

    rows = []
    t0 = time.time()
    for w_i, start_idx in enumerate(starts):
        pred_df = predict_window(
            predictor=predictor,
            df=df,
            start_idx=start_idx,
            lookback=lookback,
            pred_len=pred_len,
            feature_list=config.feature_list,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )
        actual = df.iloc[start_idx:start_idx + pred_len]
        entry_close = float(df["close"].iloc[start_idx - 1])
        actual_end = float(actual["close"].iloc[-1])
        pred_end = float(pred_df["close"].iloc[-1])

        row = {
            "symbol": symbol,
            "window_idx": w_i,
            "origin_idx": start_idx,
            "origin_time": str(df["timestamps"].iloc[start_idx - 1]),
            "horizon_end_time": str(actual["timestamps"].iloc[-1]),
            "entry_close": entry_close,
            "actual_close_end": actual_end,
            "pred_close_end": pred_end,
            "actual_return": actual_end / entry_close - 1.0,
            "pred_return": pred_end / entry_close - 1.0,
        }
        rows.append(row)

        elapsed = time.time() - t0
        done = w_i + 1
        eta = elapsed / done * (len(starts) - done)
        print(
            f"  [{done}/{len(starts)}] origin={row['origin_time']} "
            f"pred_ret={row['pred_return']:+.3%} actual_ret={row['actual_return']:+.3%} "
            f"ETA {eta / 60:.1f}m",
            flush=True,
        )

    windows_df = pd.DataFrame(rows)
    metrics = metrics_from_windows(rows)
    actual_close = pd.Series(
        df["close"].to_numpy(),
        index=pd.to_datetime(df["timestamps"]),
        name="close",
    )
    chart_path = plot_symbol_backtest(windows_df, symbol, output_dir, actual_close=actual_close)

    windows_csv = os.path.join(output_dir, f"{symbol}_windows.csv")
    windows_df.to_csv(windows_csv, index=False, encoding="utf-8-sig")

    report = {
        "symbol": symbol,
        "data_file": os.path.abspath(data_path),
        "generated_at": datetime.now().isoformat(),
        "lookback": lookback,
        "pred_len": pred_len,
        "stride": stride,
        "last_days": last_days,
        "train_ratio": config.train_ratio,
        "val_start_index": val_start,
        "val_start_time": str(df["timestamps"].iloc[val_start]),
        "interval_minutes": interval_minutes,
        "metrics": metrics,
        "windows_csv": windows_csv,
        "chart": chart_path,
    }
    report_path = os.path.join(output_dir, f"{symbol}_backtest_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"  mae={metrics['mae']:.6g} mape={metrics['mape_pct']:.3f}% "
        f"dir_acc={metrics['horizon_dir_acc']} "
        f"strat={metrics['strategy_total_return']:+.3%} bh={metrics['buy_hold_total_return']:+.3%}"
    )
    print(f"  windows -> {windows_csv}")
    print(f"  report  -> {report_path}")
    return report


def aggregate_summary(reports: list[dict], output_root: str, exp_name: str, model_paths: dict) -> str:
    rows = []
    for r in reports:
        m = r["metrics"]
        rows.append({
            "symbol": r["symbol"],
            "n_windows": m["n_windows"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mape_pct": m["mape_pct"],
            "horizon_dir_acc": m["horizon_dir_acc"],
            "strategy_total_return": m["strategy_total_return"],
            "buy_hold_total_return": m["buy_hold_total_return"],
        })
    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(output_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    numeric = summary_df.select_dtypes(include=[np.number])
    means = {col: float(numeric[col].mean()) for col in numeric.columns} if not numeric.empty else {}

    summary = {
        "exp_name": exp_name,
        "generated_at": datetime.now().isoformat(),
        "tokenizer_path": model_paths["tokenizer"],
        "basemodel_path": model_paths["basemodel"],
        "n_symbols": len(reports),
        "per_symbol_mean": means,
        "symbols": rows,
        "summary_csv": summary_csv,
    }
    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Backtest summary ({exp_name})")
    print(f"{'=' * 60}")
    if means:
        print(
            f"mean mape={means.get('mape_pct'):.3f}% | "
            f"mean dir_acc={means.get('horizon_dir_acc')} | "
            f"mean strat={means.get('strategy_total_return'):+.3%} | "
            f"mean bh={means.get('buy_hold_total_return'):+.3%}"
        )
    print(f"summary csv  -> {summary_csv}")
    print(f"summary json -> {summary_path}")
    return summary_path


def build_arg_parser(default_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for finetuned Kronos TA models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        required=default_config is None,
        help="Training YAML config (feature list, windows, exp_name)",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="CSV paths / globs (default: local data/*_kline_5min.csv)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Results root (default: backtests/<exp_name>/)",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer dir")
    parser.add_argument("--basemodel", type=str, default=None, help="Override basemodel dir")
    parser.add_argument("--lookback", type=int, default=None, help="Override lookback_window")
    parser.add_argument("--pred-len", type=int, default=None, help="Override predict_window")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Step between forecast origins (default: pred_len, non-overlapping)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap windows per symbol (useful for a quick smoke run)",
    )
    parser.add_argument("--T", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="AR samples averaged per window (1 is fastest; predict.py uses 5)",
    )
    parser.add_argument(
        "--last-days",
        type=float,
        default=3.0,
        help="Only backtest/plot origins in the last N days of each CSV (0 = full val region)",
    )
    return parser


def run_backtest(args: argparse.Namespace) -> str:
    raw_inputs = resolve_default_inputs(args.input)
    # Expand any globs the caller passed explicitly.
    data_paths: list[str] = []
    for entry in raw_inputs:
        if any(ch in entry for ch in "*?["):
            data_paths.extend(sorted(glob.glob(entry)))
        else:
            data_paths.append(entry)
    data_paths = sorted(dict.fromkeys(data_paths))
    if not data_paths:
        raise ValueError("No data files resolved from --input / default local glob")

    config = CustomFinetuneConfig(args.config, data_path_override=data_paths)
    lookback = args.lookback if args.lookback is not None else config.lookback_window
    pred_len = args.pred_len if args.pred_len is not None else config.predict_window
    stride = args.stride if args.stride is not None else pred_len

    output_root = args.output_root or os.path.join(DEFAULT_OUTPUT_ROOT, config.exp_name)
    os.makedirs(output_root, exist_ok=True)

    print(f"Config      : {args.config}")
    print(f"Experiment  : {config.exp_name}")
    print(f"Output root : {output_root}")
    print(
        f"lookback={lookback} pred_len={pred_len} stride={stride} "
        f"sample_count={args.sample_count} last_days={args.last_days}"
    )
    print(f"Symbols ({len(data_paths)}):")
    for path in data_paths:
        print(f"  - {path}")

    predictor, model_paths = build_predictor(
        config,
        tokenizer_override=args.tokenizer,
        basemodel_override=args.basemodel,
    )
    predictor.model.eval()
    predictor.tokenizer.eval()

    reports = []
    with torch.inference_mode():
        for data_path in data_paths:
            reports.append(
                backtest_one_file(
                    data_path=data_path,
                    predictor=predictor,
                    config=config,
                    output_root=output_root,
                    lookback=lookback,
                    pred_len=pred_len,
                    stride=stride,
                    max_windows=args.max_windows,
                    T=args.T,
                    top_p=args.top_p,
                    sample_count=args.sample_count,
                    last_days=args.last_days,
                )
            )

    return aggregate_summary(reports, output_root, config.exp_name, model_paths)


def main_for_config(
    default_config: str,
    argv: list[str] | None = None,
    extra_defaults: list[str] | None = None,
) -> None:
    """Entry point used by the per-experiment wrapper scripts.

    `extra_defaults` are prepended CLI flags (e.g. a shorter --pred-len for
    practical runtimes). Explicit user argv always wins over them.
    """
    if not os.path.isabs(default_config):
        default_config = os.path.join(SCRIPT_DIR, default_config)
    parser = build_arg_parser(default_config=default_config)
    merged = list(extra_defaults or [])
    if argv is None:
        merged.extend(sys.argv[1:])
    else:
        merged.extend(argv)
    args = parser.parse_args(merged)
    run_backtest(args)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
