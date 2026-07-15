"""
Run Kronos TA inference on prepared CSV data files.

Loads the finetuned tokenizer + basemodel from a training config, reads one or
more OHLCV (or prepare_dataset-augmented) CSVs, and writes forecast outputs
into a directory named after each data file stem, e.g.:

    data/ADAUSDT_kline_5min.csv  ->  predictions/ADAUSDT_kline_5min/

Usage:
    python predict.py --config configs/config_multi_symbol_5m_ta.yaml
    python predict.py --config configs/config_multi_symbol_5m_ta.yaml \\
        --input data/ADAUSDT_kline_5min.csv data/ZECUSDT_kline_5min_2026_06.csv
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import torch

warnings.filterwarnings("ignore")

sys.path.append("../")
from model import Kronos, KronosTokenizer

from config_loader import CustomFinetuneConfig
from indicators import BASE_FEATURES, ensure_features
from predictor import KronosPredictorTA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "predictions")


def resolve_pretrained_dir(*candidates: str) -> str:
    """Return the first directory that contains config.json.

    Training saves under `.../best_model/`; some exported runs place weights
    directly under `tokenizer/` or `basemodel/`, so callers should pass both.
    """
    tried = []
    for path in candidates:
        if not path:
            continue
        tried.append(path)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
        parent = os.path.dirname(path.rstrip(os.sep))
        if parent and parent not in tried:
            tried.append(parent)
            if os.path.isfile(os.path.join(parent, "config.json")):
                return parent
    raise FileNotFoundError(
        "Could not find a pretrained directory with config.json. Tried:\n  "
        + "\n  ".join(tried)
    )


def local_exp_model_dirs(exp_name: str) -> tuple[str, str]:
    """Default on-disk layout under finetune_ta/finetuned/<exp_name>/."""
    root = os.path.join(SCRIPT_DIR, "finetuned", exp_name)
    return (
        os.path.join(root, "tokenizer", "best_model"),
        os.path.join(root, "basemodel", "best_model"),
    )


def resolve_device(config: CustomFinetuneConfig) -> str:
    if config.use_cuda and torch.cuda.is_available():
        return f"cuda:{config.device_id}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_interval_minutes(timestamps: pd.Series) -> int:
    diffs = timestamps.diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return 5
    median = float(diffs.median())
    if median <= 0:
        return 5
    return max(1, int(round(median)))


def output_dir_for_data_file(data_path: str, output_root: str) -> str:
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(output_root, stem)


def load_prepared_dataframe(
    data_path: str,
    feature_list: list,
    enabled_indicators: dict,
) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    required = ["timestamps", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    for col in BASE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    df = df.dropna(subset=["timestamps", "open", "high", "low", "close"]).reset_index(drop=True)
    df = ensure_features(df, feature_list, enabled_indicators)

    before = len(df)
    df = df.dropna(subset=feature_list).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} warm-up / NaN rows; {len(df)} usable bars remain")

    if df.empty:
        raise ValueError(f"No usable rows after indicator warm-up in {data_path}")

    print(f"Loaded {len(df)} rows | {df['timestamps'].min()} -> {df['timestamps'].max()}")
    print(f"Latest close: {df['close'].iloc[-1]}")
    return df


def generate_future_timestamps(
    last_timestamp: pd.Timestamp,
    pred_len: int,
    interval_minutes: int,
) -> list:
    future = []
    current = last_timestamp + timedelta(minutes=interval_minutes)
    for _ in range(pred_len):
        future.append(current)
        current += timedelta(minutes=interval_minutes)
    print(f"Forecast timestamps: {future[0]} -> {future[-1]} ({len(future)} bars)")
    return future


def plot_prediction(
    historical_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    future_timestamps: list,
    symbol: str,
    output_dir: str,
    interval_minutes: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    historical_prices = historical_df.set_index("timestamps")["close"]
    prediction_prices = pred_df.set_index(pd.DatetimeIndex(future_timestamps))["close"]
    historical_volume = historical_df.set_index("timestamps")["volume"]
    prediction_volume = pred_df.set_index(pd.DatetimeIndex(future_timestamps))["volume"]

    current_price = historical_prices.iloc[-1]
    colors = {"historical": "#1f77b4", "prediction": "#ff7f0e", "grid": "#e9ecef"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("white")

    ax1.plot(historical_prices.index, historical_prices.values, color=colors["historical"], linewidth=2, label="History")
    if len(prediction_prices) > 0:
        ax1.plot(
            [historical_prices.index[-1], prediction_prices.index[0]],
            [historical_prices.iloc[-1], prediction_prices.iloc[0]],
            color=colors["prediction"],
            linewidth=2,
        )
        ax1.plot(
            prediction_prices.index,
            prediction_prices.values,
            color=colors["prediction"],
            linewidth=2,
            label="Forecast",
        )
        ax1.axvline(x=historical_prices.index[-1], color="red", linestyle="--", alpha=0.6)
    ax1.set_ylabel("Close")
    ax1.set_title(f"{symbol} {interval_minutes}m Kronos-TA forecast | current: {current_price}")
    ax1.legend(loc="upper left")
    ax1.grid(True, color=colors["grid"], alpha=0.7)

    vol_max = historical_volume.max()
    if vol_max and vol_max > 0:
        hist_vol_norm = historical_volume / vol_max
        pred_vol_norm = prediction_volume / vol_max
    else:
        hist_vol_norm = historical_volume
        pred_vol_norm = prediction_volume
    ax2.bar(historical_volume.index, hist_vol_norm.values, alpha=0.6, color=colors["historical"], label="History")
    ax2.bar(prediction_volume.index, pred_vol_norm.values, alpha=0.6, color=colors["prediction"], label="Forecast")
    ax2.set_ylabel("Relative volume")
    ax2.legend(loc="upper left")
    ax2.grid(True, color=colors["grid"], alpha=0.7)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"{symbol}_prediction.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Chart saved: {chart_path}")


def predict_one_file(
    data_path: str,
    predictor: KronosPredictorTA,
    config: CustomFinetuneConfig,
    output_root: str,
    model_paths: dict,
    lookback: int | None = None,
    pred_len: int | None = None,
) -> str:
    lookback = lookback if lookback is not None else config.lookback_window
    pred_len = pred_len if pred_len is not None else config.predict_window
    symbol = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = output_dir_for_data_file(data_path, output_root)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Predicting: {symbol}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 60}")

    df = load_prepared_dataframe(data_path, config.feature_list, config.enabled_indicators)
    if len(df) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} usable bars; got {len(df)} in {data_path}"
        )

    interval_minutes = infer_interval_minutes(df["timestamps"])
    historical_df = df.iloc[-lookback:].reset_index(drop=True)
    x_df = historical_df[BASE_FEATURES].reset_index(drop=True)
    # Prefer precomputed indicator columns when present so ensure_features is a no-op.
    for col in config.feature_list:
        if col not in x_df.columns and col in historical_df.columns:
            x_df[col] = historical_df[col].values

    x_timestamp = historical_df["timestamps"].reset_index(drop=True)
    last_timestamp = df["timestamps"].iloc[-1]
    future_timestamps = generate_future_timestamps(last_timestamp, pred_len, interval_minutes)

    print(f"Running prediction (lookback={lookback}, pred_len={pred_len}, interval={interval_minutes}m)...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=pd.Series(future_timestamps),
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=5,
        verbose=True,
    )
    print("Prediction complete.")

    plot_prediction(historical_df, pred_df, future_timestamps, symbol, output_dir, interval_minutes)

    forecast_csv = os.path.join(output_dir, f"{symbol}_forecast.csv")
    out = pred_df.copy()
    out.insert(0, "timestamp", future_timestamps)
    out.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    current_price = float(historical_df["close"].iloc[-1])
    forecast_price = float(pred_df["close"].iloc[-1])
    change_pct = (forecast_price / current_price - 1.0) * 100.0

    report = {
        "symbol": symbol,
        "data_file": os.path.abspath(data_path),
        "generated_at": datetime.now().isoformat(),
        "lookback": lookback,
        "pred_len": pred_len,
        "interval_minutes": interval_minutes,
        "feature_list": config.feature_list,
        "enabled_indicators": config.enabled_indicators,
        "tokenizer_path": model_paths["tokenizer"],
        "basemodel_path": model_paths["basemodel"],
        "current_price": current_price,
        "forecast_price": forecast_price,
        "change_pct": change_pct,
        "forecast_start": str(future_timestamps[0]),
        "forecast_end": str(future_timestamps[-1]),
        "forecast_csv": forecast_csv,
    }
    report_path = os.path.join(output_dir, f"{symbol}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Forecast CSV : {forecast_csv}")
    print(f"Report JSON  : {report_path}")
    print(f"Current      : {current_price}")
    print(f"Forecast     : {forecast_price} ({change_pct:+.2f}%)")
    return output_dir


def build_predictor(
    config: CustomFinetuneConfig,
    tokenizer_override: str | None = None,
    basemodel_override: str | None = None,
) -> tuple[KronosPredictorTA, dict]:
    local_tok, local_base = local_exp_model_dirs(config.exp_name)
    tokenizer_path = resolve_pretrained_dir(
        tokenizer_override,
        config.finetuned_tokenizer_path,
        config.tokenizer_best_model_path,
        local_tok,
    )
    basemodel_path = resolve_pretrained_dir(
        basemodel_override,
        config.basemodel_best_model_path,
        local_base,
    )

    device = resolve_device(config)
    print(f"Loading tokenizer from: {tokenizer_path}")
    print(f"Loading basemodel from: {basemodel_path}")
    print(f"Device: {device}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(basemodel_path)
    predictor = KronosPredictorTA(
        model,
        tokenizer,
        feature_list=config.feature_list,
        enabled_indicators=config.enabled_indicators,
        device=device,
        max_context=config.max_context,
        clip=config.clip,
    )
    return predictor, {"tokenizer": tokenizer_path, "basemodel": basemodel_path}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict with finetuned Kronos TA tokenizer + basemodel on prepared CSV data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training YAML config (model paths, feature list, lookback/predict windows)",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="One or more CSV paths / globs (overrides config data_path)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent directory for per-file result folders (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Override finetuned tokenizer directory (best_model or parent with config.json)",
    )
    parser.add_argument(
        "--basemodel",
        type=str,
        default=None,
        help="Override finetuned basemodel directory (best_model or parent with config.json)",
    )
    parser.add_argument("--lookback", type=int, default=None, help="Override config lookback_window")
    parser.add_argument("--pred-len", type=int, default=None, help="Override config predict_window")
    return parser.parse_args()


def main():
    args = parse_args()
    config = CustomFinetuneConfig(args.config, data_path_override=args.input)
    config.print_config_summary()

    data_paths = list(config.data_paths)
    if not data_paths:
        raise ValueError("No data files to predict on. Pass --input or set data.data_path in the config.")

    print(f"Will predict on {len(data_paths)} file(s)")
    for path in data_paths:
        print(f"  - {path} -> {output_dir_for_data_file(path, args.output_root)}")

    predictor, model_paths = build_predictor(
        config,
        tokenizer_override=args.tokenizer,
        basemodel_override=args.basemodel,
    )

    results = []
    for data_path in data_paths:
        out_dir = predict_one_file(
            data_path=data_path,
            predictor=predictor,
            config=config,
            output_root=args.output_root,
            model_paths=model_paths,
            lookback=args.lookback,
            pred_len=args.pred_len,
        )
        results.append(out_dir)

    print(f"\nDone. Wrote {len(results)} result director{'y' if len(results) == 1 else 'ies'}:")
    for out_dir in results:
        print(f"  {out_dir}")


if __name__ == "__main__":
    main()
