"""
Predict the horizon-end close from each CSV's last bar and export the full
1-hour (12 × 5m) forecast window.

Uses the trained hybrid LSTM+XGBoost artifacts. Default config
``config_hybrid_5m_h12`` forecasts a **scalar** Horizon Return over 12 bars.
The per-bar CSV path compounds that return evenly across the hour so the final
bar matches ``pred_close_end`` (intermediate bars are not independently scored).

Usage:
    python predict.py --config configs/config_hybrid_5m_h12.yaml
    python predict.py --config configs/config_hybrid_5m_h12.yaml --input data/BTCUSDT_kline_5min.csv
    python predict.py --config configs/config_hybrid_5m_h12.yaml --models hybrid xgb_only
"""
from __future__ import annotations

import omp_compat  # noqa: F401  — must precede torch/xgboost on macOS

import argparse
import glob
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Sequence

import pandas as pd

from config_loader import SCRIPT_DIR, HybridConfig, resolve_data_paths
from predictor import HybridPredictor
from windows import load_file

DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "predictions")
INTERVAL_MINUTES = 5


def build_hour_path(
    origin_time: pd.Timestamp,
    entry_close: float,
    pred_log_return: float,
    horizon: int,
    interval_minutes: int = INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Build one row per future 5m bar for the full horizon window.

    Close path: entry * exp(log_return * k / horizon) for k=1..horizon so the
    last bar equals the model's horizon-end close.
    """
    rows = []
    for k in range(1, horizon + 1):
        frac = k / horizon
        log_ret_k = pred_log_return * frac
        close_k = float(entry_close * math.exp(log_ret_k))
        ts = origin_time + pd.Timedelta(minutes=interval_minutes * k)
        rows.append({
            "bar": k,
            "timestamps": ts,
            "pred_close": close_k,
            "pred_log_return_cum": log_ret_k,
            "pred_return_cum_pct": (math.exp(log_ret_k) - 1.0) * 100.0,
            "is_horizon_end": k == horizon,
        })
    return pd.DataFrame(rows)


def predict_one_file(
    data_path: str,
    predictor: HybridPredictor,
    config: HybridConfig,
    model_names: Sequence[str],
    primary_model: str = "hybrid",
) -> Dict:
    symbol = os.path.splitext(os.path.basename(data_path))[0]
    features, stamps, closes, timestamps = load_file(
        data_path, config.feature_list, config.enabled_indicators
    )
    n = len(features)
    lookback = predictor.lookback
    horizon = predictor.horizon
    if n < lookback:
        raise ValueError(
            f"{symbol}: need >= {lookback} bars after warm-up, got {n}"
        )

    # Origin is one past the last bar: history = last `lookback` rows,
    # entry close = last close (same convention as backtest.py).
    origin = n
    hist = features[origin - lookback:origin].copy()
    hist_stamps = stamps[origin - lookback:origin].copy()
    hist_closes = closes[origin - lookback:origin].copy()
    entry_close = float(closes[origin - 1])
    origin_time = pd.Timestamp(timestamps.iloc[origin - 1])
    horizon_end_time = origin_time + pd.Timedelta(minutes=INTERVAL_MINUTES * horizon)
    horizon_hours = (INTERVAL_MINUTES * horizon) / 60.0

    preds = predictor.predict_close_end(
        hist_features=hist,
        hist_stamps=hist_stamps,
        hist_closes=hist_closes,
        entry_close=entry_close,
        feature_list=config.feature_list,
        models=model_names,
    )

    models_out = {}
    for name, p in preds.items():
        direction = "up" if p["pred_return"] > 0 else ("down" if p["pred_return"] < 0 else "flat")
        path_df = build_hour_path(
            origin_time=origin_time,
            entry_close=entry_close,
            pred_log_return=p["pred_log_return"],
            horizon=horizon,
        )
        models_out[name] = {
            "pred_log_return": p["pred_log_return"],
            "pred_return": p["pred_return"],
            "pred_return_pct": p["pred_return"] * 100.0,
            "pred_close_end": p["pred_close_end"],
            "direction": direction,
            "path": path_df,
        }

    primary = primary_model if primary_model in models_out else next(iter(models_out))
    return {
        "symbol": symbol,
        "data_file": os.path.abspath(data_path),
        "generated_at": datetime.now().isoformat(),
        "lookback": lookback,
        "horizon": horizon,
        "horizon_hours": horizon_hours,
        "interval_minutes": INTERVAL_MINUTES,
        "origin_idx": origin - 1,
        "origin_time": str(origin_time),
        "horizon_end_time": str(horizon_end_time),
        "entry_close": entry_close,
        "primary_model": primary,
        "models": models_out,
    }


def print_report(report: Dict, primary_model: str = "hybrid") -> None:
    print(f"\n{'=' * 60}")
    print(f"Predict: {report['symbol']}")
    print(
        f"origin={report['origin_time']} -> "
        f"horizon_end={report['horizon_end_time']} "
        f"({report['horizon']} bars / {report['horizon_hours']:g}h)"
    )
    print(f"entry_close={report['entry_close']:.8g}")
    for name, m in report["models"].items():
        mark = " *" if name == report.get("primary_model", primary_model) else ""
        print(
            f"  [{name}]{mark} pred_close={m['pred_close_end']:.8g} "
            f"return={m['pred_return_pct']:+.4f}% dir={m['direction']}"
        )


def _models_for_json(models_out: Dict) -> Dict:
    """Drop DataFrame paths; keep scalar fields for JSON reports."""
    out = {}
    for name, m in models_out.items():
        out[name] = {
            "pred_log_return": m["pred_log_return"],
            "pred_return": m["pred_return"],
            "pred_return_pct": m["pred_return_pct"],
            "pred_close_end": m["pred_close_end"],
            "direction": m["direction"],
        }
    return out


def write_forecast_csvs(report: Dict, output_dir: str) -> Dict[str, str]:
    """Write one 12-bar CSV per model plus a primary-model alias."""
    symbol = report["symbol"]
    paths: Dict[str, str] = {}
    for name, m in report["models"].items():
        path_df = m["path"].copy()
        path_df.insert(0, "symbol", symbol)
        path_df.insert(1, "model", name)
        path_df.insert(2, "origin_time", report["origin_time"])
        path_df.insert(3, "entry_close", report["entry_close"])
        path_df["pred_close_end"] = m["pred_close_end"]
        path_df["direction"] = m["direction"]
        out_path = os.path.join(output_dir, f"{symbol}_{name}_1h_forecast.csv")
        path_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        paths[name] = out_path

    primary = report["primary_model"]
    if primary in paths:
        alias = os.path.join(output_dir, f"{symbol}_1h_forecast.csv")
        # Copy primary path CSV under a stable name.
        pd.read_csv(paths[primary]).to_csv(alias, index=False, encoding="utf-8-sig")
        paths["primary"] = alias
    return paths


def run_predict(
    config: HybridConfig,
    output_root: str | None = None,
    models: Sequence[str] | None = None,
) -> str:
    output_root = output_root or os.path.join(DEFAULT_OUTPUT_ROOT, config.exp_name)
    os.makedirs(output_root, exist_ok=True)

    if models is None:
        model_names = ["hybrid"]
        if config.ablation_lstm_only:
            model_names.append("lstm_only")
        if config.ablation_xgb_only and os.path.exists(
            config.artifact_paths()["xgb_only_model"]
        ):
            model_names.append("xgb_only")
    else:
        model_names = list(models)

    print(f"Config     : {config.config_path}")
    print(f"Experiment : {config.exp_name}")
    print(f"Save dir   : {config.save_dir}")
    print(f"Output     : {output_root}")
    print(f"Models     : {model_names}")
    print(
        f"Horizon    : {config.horizon} bars × {INTERVAL_MINUTES}m = "
        f"{(config.horizon * INTERVAL_MINUTES) / 60.0:g}h"
    )
    print(f"Files      : {len(config.data_paths)}")

    predictor = HybridPredictor(
        config.save_dir, device=config.device(), load_ablations=True
    )
    if predictor.horizon != config.horizon or predictor.lookback != config.lookback_window:
        raise ValueError(
            f"feature_spec mismatch: model lookback/horizon="
            f"{predictor.lookback}/{predictor.horizon} vs config "
            f"{config.lookback_window}/{config.horizon}"
        )

    reports: List[Dict] = []
    summary_rows = []
    path_rows = []
    for path in config.data_paths:
        report = predict_one_file(path, predictor, config, model_names)
        print_report(report)
        csv_paths = write_forecast_csvs(report, output_root)

        json_report = {
            **{k: v for k, v in report.items() if k != "models"},
            "models": _models_for_json(report["models"]),
            "forecast_csv": csv_paths.get("primary"),
            "forecast_csvs": {k: v for k, v in csv_paths.items() if k != "primary"},
        }
        report_path = os.path.join(output_root, f"{report['symbol']}_forecast.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        print(f"  wrote {csv_paths.get('primary')}")
        print(f"  wrote {report_path}")

        reports.append(json_report)
        for name, m in report["models"].items():
            summary_rows.append({
                "symbol": report["symbol"],
                "model": name,
                "origin_time": report["origin_time"],
                "horizon_end_time": report["horizon_end_time"],
                "horizon": report["horizon"],
                "horizon_hours": report["horizon_hours"],
                "entry_close": report["entry_close"],
                "pred_close_end": m["pred_close_end"],
                "pred_return_pct": m["pred_return_pct"],
                "direction": m["direction"],
                "forecast_csv": csv_paths.get(name),
                "data_file": report["data_file"],
            })
            path_df = m["path"].copy()
            path_df.insert(0, "symbol", report["symbol"])
            path_df.insert(1, "model", name)
            path_df.insert(2, "origin_time", report["origin_time"])
            path_df.insert(3, "entry_close", report["entry_close"])
            path_rows.append(path_df)

    summary_csv = os.path.join(output_root, "summary.csv")
    summary_json = os.path.join(output_root, "summary.json")
    all_paths_csv = os.path.join(output_root, "all_1h_forecasts.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if path_rows:
        pd.concat(path_rows, ignore_index=True).to_csv(
            all_paths_csv, index=False, encoding="utf-8-sig"
        )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "exp_name": config.exp_name,
                "horizon": config.horizon,
                "horizon_hours": (config.horizon * INTERVAL_MINUTES) / 60.0,
                "lookback": config.lookback_window,
                "models": model_names,
                "note": (
                    "Per-bar closes compound the scalar horizon log-return evenly; "
                    "only the final bar is the model output."
                ),
                "predictions": reports,
            },
            f,
            indent=2,
        )

    print(f"\nDone: {len(reports)} file(s)")
    print(f"summary csv       -> {summary_csv}")
    print(f"all 1h forecasts  -> {all_paths_csv}")
    print(f"summary json      -> {summary_json}")
    return summary_json


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict horizon-end close from each CSV's last bar "
            "(default hybrid_5m_h12 = 12×5m = 1h ahead)"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_hybrid_5m_h12.yaml",
        help="Hybrid YAML (model paths, lookback/horizon, indicators)",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="CSV path(s) / globs (overrides config data_path)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=f"Output directory (default: predictions/<exp_name>/ under {SCRIPT_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["hybrid", "lstm_only", "xgb_only"],
        help="Which models to score (default: hybrid + enabled ablations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_override = None
    if args.input:
        paths: List[str] = []
        for entry in args.input:
            if any(ch in entry for ch in "*?["):
                matches = sorted(glob.glob(entry))
                if not matches:
                    # Also try relative to script dir / cwd via resolve helper.
                    matches = resolve_data_paths(entry, base_dir=SCRIPT_DIR)
                paths.extend(matches)
            else:
                path = entry if os.path.isabs(entry) else os.path.normpath(
                    os.path.join(os.getcwd(), entry)
                )
                if not os.path.exists(path):
                    alt = os.path.normpath(os.path.join(SCRIPT_DIR, entry))
                    path = alt if os.path.exists(alt) else path
                paths.append(path)
        data_override = sorted(dict.fromkeys(paths))

    config = HybridConfig(args.config, data_path_override=data_override)
    run_predict(config, output_root=args.output_root, models=args.models)


if __name__ == "__main__":
    main()
