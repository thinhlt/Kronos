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
import shutil
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

sys.path.append("../")
from model import Kronos, KronosTokenizer

from config_loader import CustomFinetuneConfig
from indicators import (
    ATR_COLUMNS,
    BASE_FEATURES,
    BOLLINGER_COLUMNS,
    HEIKIN_ASHI_COLUMNS,
    MACD_COLUMNS,
    ensure_features,
)
from predictor import KronosPredictorTA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "predictions")

# Open-anchored columns: rescale these when re-anchoring preds to live price.
_REANCHOR_COLUMNS = (
    ["open", "high", "low", "close"]
    + list(HEIKIN_ASHI_COLUMNS)
    + list(BOLLINGER_COLUMNS)
    + list(ATR_COLUMNS)
    + list(MACD_COLUMNS)
)


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
    verbose: bool = True,
) -> list:
    future = []
    current = last_timestamp + timedelta(minutes=interval_minutes)
    for _ in range(pred_len):
        future.append(current)
        current += timedelta(minutes=interval_minutes)
    if verbose:
        print(f"Forecast timestamps: {future[0]} -> {future[-1]} ({len(future)} bars)")
    return future


def overlap_mape_pct(pred_close, actual_close) -> float:
    """Mean absolute percentage error (%) of overlap closes vs known actuals."""
    pred = np.asarray(pred_close, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_close, dtype=np.float64).reshape(-1)
    if pred.shape != actual.shape:
        raise ValueError(f"pred/actual shape mismatch: {pred.shape} vs {actual.shape}")
    if pred.size == 0:
        raise ValueError("Cannot compute MAPE on empty overlap")
    denom = np.where(np.abs(actual) < 1e-12, np.nan, np.abs(actual))
    mape = float(np.nanmean(np.abs(pred - actual) / denom) * 100.0)
    if np.isnan(mape):
        raise ValueError("MAPE is NaN (actual closes near zero?)")
    return mape


def reanchor_predictions(
    pred_df: pd.DataFrame,
    current_price: float,
    origin_offset: int,
) -> tuple[pd.DataFrame, float]:
    """Drop the first `origin_offset` bars and scale open-anchored cols to live price.

    Model history ends at T-`origin_offset`, so pred[origin_offset-1] is the model's
    estimate of "now" (T). Scale by current_price / that close so usable pred[0]
    (raw pred[origin_offset]) is consistent with the real ticker.
    """
    if origin_offset <= 0:
        return pred_df.copy(), float(pred_df["close"].iloc[0])

    if len(pred_df) <= origin_offset:
        raise ValueError(
            f"Need more than origin_offset={origin_offset} predicted bars; got {len(pred_df)}"
        )

    pred_now = float(pred_df["close"].iloc[origin_offset - 1])
    if pred_now == 0.0:
        raise ValueError("Model close at T (pred[origin_offset-1]) is zero; cannot re-anchor")

    scale = current_price / pred_now
    usable = pred_df.iloc[origin_offset:].copy().reset_index(drop=True)
    for col in _REANCHOR_COLUMNS:
        if col in usable.columns:
            usable[col] = usable[col] * scale

    real_first = float(usable["close"].iloc[0])
    print(
        f"Re-anchored: current={current_price}, model_T={pred_now}, "
        f"scale={scale:.6f}, first_future={real_first}"
    )
    return usable, real_first


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
    origin_offset: int = 3,
    max_attempts: int = 20,
    mape_threshold: float = 0.5,
) -> str:
    lookback = lookback if lookback is not None else config.lookback_window
    pred_len = pred_len if pred_len is not None else config.predict_window
    if origin_offset < 0:
        raise ValueError(f"origin_offset must be >= 0, got {origin_offset}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    symbol = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = output_dir_for_data_file(data_path, output_root)
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Predicting: {symbol}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 60}")

    df = load_prepared_dataframe(data_path, config.feature_list, config.enabled_indicators)
    min_rows = lookback + origin_offset
    if len(df) < min_rows:
        raise ValueError(
            f"Need at least lookback+origin_offset={min_rows} usable bars; "
            f"got {len(df)} in {data_path}"
        )

    interval_minutes = infer_interval_minutes(df["timestamps"])
    # Model context ends at T-origin_offset; chart history still ends at live T.
    if origin_offset > 0:
        model_hist = df.iloc[-(lookback + origin_offset):-origin_offset].reset_index(drop=True)
        actual_overlap = df["close"].iloc[-origin_offset:].to_numpy(dtype=np.float64)
    else:
        model_hist = df.iloc[-lookback:].reset_index(drop=True)
        actual_overlap = np.array([], dtype=np.float64)
    chart_hist = df.iloc[-lookback:].reset_index(drop=True)

    x_df = model_hist[BASE_FEATURES].reset_index(drop=True)
    # Prefer precomputed indicator columns when present so ensure_features is a no-op.
    for col in config.feature_list:
        if col not in x_df.columns and col in model_hist.columns:
            x_df[col] = model_hist[col].values

    x_timestamp = model_hist["timestamps"].reset_index(drop=True)
    current_price = float(df["close"].iloc[-1])
    last_live_timestamp = df["timestamps"].iloc[-1]
    model_origin_timestamp = model_hist["timestamps"].iloc[-1]

    pred_len_raw = pred_len + origin_offset
    # Raw AR stamps start after T-offset; usable stamps start after live T.
    raw_timestamps = generate_future_timestamps(
        model_origin_timestamp, pred_len_raw, interval_minutes, verbose=False
    )
    future_timestamps = generate_future_timestamps(
        last_live_timestamp, pred_len, interval_minutes, verbose=True
    )

    print(
        f"Running prediction (lookback={lookback}, pred_len={pred_len}, "
        f"origin_offset={origin_offset}, max_attempts={max_attempts}, "
        f"mape_threshold={mape_threshold}%, interval={interval_minutes}m)..."
    )
    if origin_offset > 0:
        print(
            f"Model origin T-{origin_offset}: {model_origin_timestamp} | "
            f"live T: {last_live_timestamp} close={current_price}"
        )

    best_raw = None
    best_mape = float("inf")
    best_attempt = 0
    attempts_run = 0
    early_exit = False

    for attempt in range(1, max_attempts + 1):
        attempts_run = attempt
        raw_pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(raw_timestamps),
            pred_len=pred_len_raw,
            T=0.2,
            top_p=0.9,
            top_k=20,
            sample_count=1,
            verbose=False,
        )

        if origin_offset > 0:
            mape_pct = overlap_mape_pct(
                raw_pred_df["close"].iloc[:origin_offset].values,
                actual_overlap,
            )
        else:
            mape_pct = 0.0

        raw_csv = os.path.join(temp_dir, f"attempt_{attempt:02d}_raw.csv")
        meta_path = os.path.join(temp_dir, f"attempt_{attempt:02d}_meta.json")
        raw_out = raw_pred_df.copy()
        raw_out.insert(0, "timestamp", raw_timestamps)
        raw_out.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        meta = {
            "attempt": attempt,
            "mape_pct": mape_pct,
            "mape_threshold_pct": mape_threshold,
            "origin_offset": origin_offset,
            "raw_csv": raw_csv,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"  attempt {attempt:02d}/{max_attempts}: MAPE={mape_pct:.4f}%")

        if mape_pct < best_mape:
            best_mape = mape_pct
            best_raw = raw_pred_df
            best_attempt = attempt

        if mape_pct < mape_threshold:
            early_exit = True
            print(f"  Early exit: MAPE {mape_pct:.4f}% < {mape_threshold}%")
            break

    if best_raw is None:
        raise RuntimeError("No prediction attempts produced a usable sample")

    selected_attempt = best_attempt
    selected_mape = best_mape
    if not early_exit:
        print(
            f"  No attempt under {mape_threshold}%; "
            f"keeping best MAPE={best_mape:.4f}% (attempt {best_attempt:02d})"
        )

    print("Prediction complete.")

    pred_df, real_first = reanchor_predictions(best_raw, current_price, origin_offset)
    if len(pred_df) != len(future_timestamps):
        raise RuntimeError(
            f"Usable forecast length {len(pred_df)} != pred_len timestamps {len(future_timestamps)}"
        )

    plot_prediction(chart_hist, pred_df, future_timestamps, symbol, output_dir, interval_minutes)

    forecast_csv = os.path.join(output_dir, f"{symbol}_forecast.csv")
    out = pred_df.copy()
    out.insert(0, "timestamp", future_timestamps)
    out.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    forecast_price = float(pred_df["close"].iloc[-1])
    change_pct = (forecast_price / current_price - 1.0) * 100.0
    next_bar_change_pct = (real_first / current_price - 1.0) * 100.0

    report = {
        "symbol": symbol,
        "data_file": os.path.abspath(data_path),
        "generated_at": datetime.now().isoformat(),
        "lookback": lookback,
        "pred_len": pred_len,
        "origin_offset": origin_offset,
        "max_attempts": max_attempts,
        "attempts_run": attempts_run,
        "selected_attempt": selected_attempt,
        "early_exit": early_exit,
        "mape_pct": selected_mape,
        "mape_threshold_pct": mape_threshold,
        "interval_minutes": interval_minutes,
        "feature_list": config.feature_list,
        "enabled_indicators": config.enabled_indicators,
        "tokenizer_path": model_paths["tokenizer"],
        "basemodel_path": model_paths["basemodel"],
        "current_price": current_price,
        "next_bar_price": real_first,
        "next_bar_change_pct": next_bar_change_pct,
        "forecast_price": forecast_price,
        "change_pct": change_pct,
        "model_origin_timestamp": str(model_origin_timestamp),
        "forecast_start": str(future_timestamps[0]),
        "forecast_end": str(future_timestamps[-1]),
        "forecast_csv": forecast_csv,
        "temp_dir": temp_dir,
    }
    report_path = os.path.join(output_dir, f"{symbol}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Forecast CSV : {forecast_csv}")
    print(f"Report JSON  : {report_path}")
    print(f"Selected     : attempt {selected_attempt:02d} MAPE={selected_mape:.4f}% early_exit={early_exit}")
    print(f"Current      : {current_price}")
    print(f"Next bar     : {real_first} ({next_bar_change_pct:+.2f}%)")
    print(f"Forecast end : {forecast_price} ({change_pct:+.2f}%)")
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
    parser.add_argument(
        "--origin-offset",
        type=int,
        default=3,
        help=(
            "Start model history at T-N (default: 3). Predicts N extra bars, "
            "scores MAPE on those N known bars, drops them, and re-anchors to live price."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Max sample_count=1 draws; stop early if overlap MAPE is under threshold (default: 20)",
    )
    parser.add_argument(
        "--mape-threshold",
        type=float,
        default=0.5,
        help="Early-exit overlap MAPE threshold in percent (default: 0.5)",
    )
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
            origin_offset=args.origin_offset,
            max_attempts=args.max_attempts,
            mape_threshold=args.mape_threshold,
        )
        results.append(out_dir)

    print(f"\nDone. Wrote {len(results)} result director{'y' if len(results) == 1 else 'ies'}:")
    for out_dir in results:
        print(f"  {out_dir}")


if __name__ == "__main__":
    main()
