"""Walk-forward backtest for the hybrid LSTM + XGBoost baseline.

Val-region origins at backtest_stride (default 1). Error metrics use every origin;
compounding metrics use the non-overlapping subsample (every horizon-th origin).
"""
from __future__ import annotations

import omp_compat  # noqa: F401  — must precede torch/xgboost on macOS

import argparse
import glob
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config_loader import HybridConfig, SCRIPT_DIR
from metrics import metrics_from_windows
from predictor import HybridPredictor
from windows import (
    FileWindows,
    build_all_files,
    collect_origins,
    horizon_log_return,
    slice_history,
)

DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "backtests")


def filter_origins_last_days(
    fw: FileWindows,
    origins: List[int],
    last_days: float | None,
) -> List[int]:
    if last_days is None or last_days <= 0 or not origins:
        return origins
    stamps = pd.to_datetime(fw.timestamps)
    cutoff = stamps.iloc[-1] - pd.Timedelta(days=last_days)
    return [o for o in origins if stamps.iloc[o - 1] >= cutoff]


def _format_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")


def plot_symbol_backtest(
    windows: pd.DataFrame,
    symbol: str,
    output_dir: str,
    model_name: str,
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
            label="Actual close",
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
    ax1.set_title(f"{symbol} [{model_name}] walk-forward close (actual vs predicted)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    # Plot non-overlapping cumulative for readability.
    horizon = max(1, int(plot_df["horizon"].iloc[0]) if "horizon" in plot_df.columns else 12)
    sub = plot_df.iloc[::horizon].reset_index(drop=True)
    cum_bh = (1.0 + sub["actual_return"]).cumprod()
    cum_strat = (1.0 + sub["actual_return"].where(sub["pred_return"] > 0, 0.0)).cumprod()
    ax2.plot(sub["origin_time"], cum_bh, label="Buy & hold (non-overlap)", color="#1f77b4")
    ax2.plot(sub["origin_time"], cum_strat, label="Long-if-pred-up (non-overlap)", color="#2ca02c")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cumulative growth")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    _format_time_axis(ax2)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{symbol}_{model_name}_backtest.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


@torch.no_grad()
def backtest_one_file(
    fw: FileWindows,
    predictor: HybridPredictor,
    config: HybridConfig,
    output_root: str,
    model_names: Sequence[str],
    last_days: float | None = 3.0,
) -> Dict:
    symbol = os.path.splitext(os.path.basename(fw.path))[0]
    output_dir = os.path.join(output_root, symbol)
    os.makedirs(output_dir, exist_ok=True)

    pairs = collect_origins(
        [fw],
        "val",
        config.lookback_window,
        config.horizon,
        config.backtest_stride,
    )
    # pairs are (0, origin) because we passed a single-file list
    origins = [origin for _, origin in pairs]
    origins = filter_origins_last_days(fw, origins, last_days)
    print(
        f"\n{'=' * 60}\nBacktesting: {symbol}\n"
        f"val origins={len(origins)} stride={config.backtest_stride} "
        f"lookback={config.lookback_window} horizon={config.horizon} "
        f"last_days={last_days}\n"
        f"{'=' * 60}"
    )
    if not origins:
        raise ValueError(f"{symbol}: no val-region origins (last_days={last_days})")

    per_model_rows: Dict[str, List[dict]] = {m: [] for m in model_names}
    t0 = time.time()
    batch_size = config.lstm_batch_size

    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        for oi, origin in enumerate(batch_origins):
            hist, stamps = slice_history(fw, origin, config.lookback_window)
            hist_closes = fw.closes[origin - config.lookback_window:origin].copy()
            entry_close = float(fw.closes[origin - 1])
            actual_end = float(fw.closes[origin + config.horizon - 1])
            actual_logret = horizon_log_return(fw.closes, origin, config.horizon)
            actual_return = float(np.exp(actual_logret) - 1.0)

            preds = predictor.predict_close_end(
                hist_features=hist,
                hist_stamps=stamps,
                hist_closes=hist_closes,
                entry_close=entry_close,
                feature_list=config.feature_list,
                models=model_names,
            )
            for name, p in preds.items():
                per_model_rows[name].append({
                    "symbol": symbol,
                    "window_idx": start + oi,
                    "origin_idx": origin,
                    "origin_time": str(fw.timestamps.iloc[origin - 1]),
                    "horizon_end_time": str(fw.timestamps.iloc[origin + config.horizon - 1]),
                    "horizon": config.horizon,
                    "entry_close": entry_close,
                    "actual_close_end": actual_end,
                    "pred_close_end": p["pred_close_end"],
                    "actual_return": actual_return,
                    "pred_return": p["pred_return"],
                    "pred_log_return": p["pred_log_return"],
                })

        done = min(start + batch_size, len(origins))
        elapsed = time.time() - t0
        eta = elapsed / done * (len(origins) - done) if done else 0.0
        print(f"  [{done}/{len(origins)}] ETA {eta / 60:.1f}m", flush=True)

    actual_close = pd.Series(
        fw.closes,
        index=pd.to_datetime(fw.timestamps),
        name="close",
    )
    reports = {}
    for name, rows in per_model_rows.items():
        windows_df = pd.DataFrame(rows)
        metrics = metrics_from_windows(rows, horizon=config.horizon)
        chart = plot_symbol_backtest(
            windows_df, symbol, output_dir, name, actual_close=actual_close
        )
        windows_csv = os.path.join(output_dir, f"{symbol}_{name}_windows.csv")
        windows_df.to_csv(windows_csv, index=False, encoding="utf-8-sig")
        report = {
            "symbol": symbol,
            "model": name,
            "data_file": os.path.abspath(fw.path),
            "generated_at": datetime.now().isoformat(),
            "lookback": config.lookback_window,
            "horizon": config.horizon,
            "stride": config.backtest_stride,
            "last_days": last_days,
            "train_ratio": config.train_ratio,
            "metrics": metrics,
            "windows_csv": windows_csv,
            "chart": chart,
        }
        report_path = os.path.join(output_dir, f"{symbol}_{name}_backtest_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        m = metrics
        print(
            f"  [{name}] mape={m['mape_pct']:.3f}% dir_acc={m['horizon_dir_acc']} "
            f"minmax_rmse={m['minmax_rmse']} "
            f"strat={m['strategy_total_return']} bh={m['buy_hold_total_return']} "
            f"n={m['n_windows']}/{m['n_windows_nonoverlap']}"
        )
        reports[name] = report
    return {"symbol": symbol, "models": reports}


def aggregate_summary(file_reports: List[dict], output_root: str, exp_name: str) -> str:
    rows = []
    for fr in file_reports:
        for name, report in fr["models"].items():
            m = report["metrics"]
            rows.append({
                "symbol": fr["symbol"],
                "model": name,
                "n_windows": m["n_windows"],
                "n_windows_nonoverlap": m["n_windows_nonoverlap"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape_pct": m["mape_pct"],
                "minmax_rmse": m["minmax_rmse"],
                "horizon_dir_acc": m["horizon_dir_acc"],
                "strategy_total_return": m["strategy_total_return"],
                "buy_hold_total_return": m["buy_hold_total_return"],
            })
    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(output_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    by_model = {}
    for name, group in summary_df.groupby("model"):
        numeric = group.select_dtypes(include=[np.number])
        by_model[name] = {col: float(numeric[col].mean()) for col in numeric.columns}

    summary = {
        "exp_name": exp_name,
        "generated_at": datetime.now().isoformat(),
        "n_symbols": len(file_reports),
        "per_model_mean": by_model,
        "rows": rows,
        "summary_csv": summary_csv,
    }
    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}\nBacktest summary ({exp_name})\n{'=' * 60}")
    for name, means in by_model.items():
        print(
            f"[{name}] mean mape={means.get('mape_pct'):.3f}% | "
            f"dir_acc={means.get('horizon_dir_acc')} | "
            f"minmax_rmse={means.get('minmax_rmse')} | "
            f"strat={means.get('strategy_total_return')} | "
            f"bh={means.get('buy_hold_total_return')}"
        )
    print(f"summary csv  -> {summary_csv}")
    print(f"summary json -> {summary_path}")
    return summary_path


def run_backtest(
    config: HybridConfig,
    output_root: str | None = None,
    last_days: float | None = 3.0,
) -> str:
    output_root = output_root or os.path.join(DEFAULT_OUTPUT_ROOT, config.exp_name)
    os.makedirs(output_root, exist_ok=True)

    model_names = ["hybrid"]
    if config.ablation_lstm_only:
        model_names.append("lstm_only")
    if config.ablation_xgb_only and os.path.exists(config.artifact_paths()["xgb_only_model"]):
        model_names.append("xgb_only")

    print(f"Config     : {config.config_path}")
    print(f"Experiment : {config.exp_name}")
    print(f"Save dir   : {config.save_dir}")
    print(f"Output     : {output_root}")
    print(f"Models     : {model_names}")
    print(f"last_days  : {last_days}")

    predictor = HybridPredictor(config.save_dir, device=config.device(), load_ablations=True)
    files = build_all_files(config)
    reports = []
    for fw in files:
        reports.append(
            backtest_one_file(
                fw, predictor, config, output_root, model_names, last_days=last_days
            )
        )
    return aggregate_summary(reports, output_root, config.exp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest hybrid LSTM+XGBoost baseline")
    parser.add_argument("--config", type=str, default="configs/config_hybrid_5m_h12.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--input", type=str, nargs="+", default=None)
    parser.add_argument(
        "--last-days",
        type=float,
        default=3.0,
        help="Only backtest/plot origins in the last N days (0 = full val region)",
    )
    args = parser.parse_args()

    data_override = None
    if args.input:
        paths: List[str] = []
        for entry in args.input:
            if any(ch in entry for ch in "*?["):
                paths.extend(sorted(glob.glob(entry)))
            else:
                paths.append(entry)
        data_override = sorted(dict.fromkeys(paths))

    config = HybridConfig(args.config, data_path_override=data_override)
    run_backtest(config, output_root=args.output_root, last_days=args.last_days)


if __name__ == "__main__":
    main()
