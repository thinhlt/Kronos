"""
Build prediction-ready CSVs for one or more symbols.

Window (UTC): start of last calendar month through end of yesterday.

Steps per symbol:
  1. Download Binance Vision monthly 5m archives for every fully-finished
     month in the window (fall back to daily ZIPs if a monthly file is missing).
  2. Download Binance Vision daily 5m archives for the remaining days
     (typically the current month up through yesterday).
  3. Convert to Kronos OHLCV, merge monthly + daily, drop duplicates.
  4. Compute technical-indicator features and write one final CSV.

Output default:  finetune_ta/data/{SYMBOL}_kline_5min.csv

Usage:
    python download_input_predict_data.py --symbols BTCUSDT ADAUSDT
    python download_input_predict_data.py --symbols ETHUSDT --force
    python download_input_predict_data.py --symbols BTCUSDT \\
        --config configs/config_multi_symbol_5m_ta.yaml
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests

from clean_merge_data import KRONOS_COLUMNS, load_raw_month
from download_data import DATA_DIR, DEFAULT_SYMBOLS, INTERVAL, RAW_DIR
from indicators import (
    BASE_FEATURES,
    DEFAULT_ENABLED_INDICATORS,
    INDICATOR_ORDER,
    build_feature_list,
    compute_indicators,
)

MONTHLY_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DAILY_BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
REQUEST_TIMEOUT = 60


def prediction_window(today: date | None = None) -> tuple[date, date]:
    """Return [start_of_last_month, yesterday] inclusive (UTC calendar dates)."""
    today = today or datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    first_of_this_month = today.replace(day=1)
    start = (first_of_this_month - timedelta(days=1)).replace(day=1)
    if yesterday < start:
        raise ValueError(
            f"Empty window: start={start}, yesterday={yesterday}. "
            "Cannot build prediction data before the 2nd of a month."
        )
    return start, yesterday


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def complete_months_in_window(start: date, end: date) -> list[tuple[int, int]]:
    """Months whose full calendar span sits inside [start, end]."""
    months = []
    year, month = start.year, start.month
    while True:
        month_start = date(year, month, 1)
        if month == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month + 1, 1)
        month_end = next_month_start - timedelta(days=1)
        if month_start > end:
            break
        if month_start >= start and month_end <= end:
            months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def days_not_covered_by_months(
    start: date,
    end: date,
    months: list[tuple[int, int]],
) -> list[date]:
    covered = set()
    for year, month in months:
        month_start = date(year, month, 1)
        if month == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month + 1, 1)
        covered.update(daterange(month_start, next_month_start - timedelta(days=1)))
    return [d for d in daterange(start, end) if d not in covered]


def monthly_zip_url(symbol: str, year: int, month: int) -> str:
    return f"{MONTHLY_BASE_URL}/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"


def daily_zip_url(symbol: str, day: date) -> str:
    return (
        f"{DAILY_BASE_URL}/{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-{day.isoformat()}.zip"
    )


def monthly_raw_path(symbol: str, year: int, month: int) -> str:
    return os.path.join(RAW_DIR, symbol, f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.csv")


def daily_raw_path(symbol: str, day: date) -> str:
    return os.path.join(RAW_DIR, symbol, "daily", f"{symbol}-{INTERVAL}-{day.isoformat()}.csv")


def _download_zip_to_csv(url: str, dest_path: str, force: bool = False) -> str:
    """Download a Vision ZIP and extract its CSV. Returns ok/skipped/missing."""
    if os.path.exists(dest_path) and not force:
        return "skipped"

    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code == 404:
        return "missing"
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        inner_name = zf.namelist()[0]
        csv_bytes = zf.read(inner_name)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(csv_bytes)
    return "ok"


def download_month(symbol: str, year: int, month: int, force: bool = False) -> str:
    return _download_zip_to_csv(
        monthly_zip_url(symbol, year, month),
        monthly_raw_path(symbol, year, month),
        force=force,
    )


def download_day(symbol: str, day: date, force: bool = False) -> str:
    return _download_zip_to_csv(
        daily_zip_url(symbol, day),
        daily_raw_path(symbol, day),
        force=force,
    )


def download_symbol_window(
    symbol: str,
    start: date,
    end: date,
    force: bool = False,
) -> tuple[list[str], list[str]]:
    """Download monthly + daily archives. Returns (monthly_paths, daily_paths)."""
    months = complete_months_in_window(start, end)
    monthly_paths: list[str] = []
    daily_days: list[date] = []

    print(f"\n{symbol}: window {start} -> {end}")
    print(f"  complete months to fetch as monthly: "
          f"{[f'{y:04d}-{m:02d}' for y, m in months] or '(none)'}")

    for year, month in months:
        status = download_month(symbol, year, month, force=force)
        path = monthly_raw_path(symbol, year, month)
        if status == "missing":
            print(f"  monthly {year:04d}-{month:02d}: missing on Vision, falling back to daily")
            month_start = date(year, month, 1)
            if month == 12:
                month_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, month + 1, 1) - timedelta(days=1)
            month_end = min(month_end, end)
            daily_days.extend(daterange(month_start, month_end))
        else:
            print(f"  monthly {year:04d}-{month:02d}: {status}")
            monthly_paths.append(path)

    daily_days.extend(days_not_covered_by_months(start, end, months))
    # de-dupe while preserving order (fallback months may overlap the partial tail)
    seen = set()
    unique_days = []
    for day in daily_days:
        if day not in seen:
            seen.add(day)
            unique_days.append(day)

    print(f"  daily files to fetch: {len(unique_days)}")
    daily_paths: list[str] = []
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    missing_days = []
    for day in unique_days:
        status = download_day(symbol, day, force=force)
        counts[status] += 1
        if status == "missing":
            missing_days.append(day.isoformat())
        else:
            daily_paths.append(daily_raw_path(symbol, day))

    print(f"  daily: {counts['ok']} downloaded, {counts['skipped']} already present, "
          f"{counts['missing']} missing")
    if missing_days:
        print(f"  missing days: {', '.join(missing_days)}")

    return monthly_paths, daily_paths


def merge_raw_files(paths: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=KRONOS_COLUMNS)

    frames = [load_raw_month(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.sort_values("timestamps")
        .drop_duplicates(subset=["timestamps"])
        .reset_index(drop=True)
    )
    return merged


def filter_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Keep bars whose open time falls within [start 00:00, end 23:59:59]."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (df["timestamps"] >= start_ts) & (df["timestamps"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


def add_features(df: pd.DataFrame, enabled_indicators: dict) -> tuple[pd.DataFrame, list]:
    feature_list = build_feature_list(enabled_indicators)
    if df[BASE_FEATURES].isnull().any().any():
        print("  warning: missing base OHLCV values, forward-filling")
        df = df.copy()
        df[BASE_FEATURES] = df[BASE_FEATURES].ffill()

    df = compute_indicators(df, enabled_indicators)
    before = len(df)
    warmup_mask = df[feature_list].notna().all(axis=1)
    df = df.loc[warmup_mask].reset_index(drop=True)
    print(f"  features: dropped {before - len(df)} warm-up rows, {len(df)} usable remain")
    return df, feature_list


def final_output_path(symbol: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"{symbol}_kline_5min.csv")


def process_symbol(
    symbol: str,
    start: date,
    end: date,
    output_dir: str,
    enabled_indicators: dict,
    force: bool = False,
) -> str | None:
    monthly_paths, daily_paths = download_symbol_window(symbol, start, end, force=force)
    all_paths = monthly_paths + daily_paths
    if not all_paths:
        print(f"{symbol}: no raw files downloaded, skipping")
        return None

    print(f"  merging {len(monthly_paths)} monthly + {len(daily_paths)} daily file(s)")
    merged = merge_raw_files(all_paths)
    merged = filter_window(merged, start, end)
    if merged.empty:
        print(f"{symbol}: merged frame empty after window filter, skipping")
        return None

    print(f"  merged OHLCV: {len(merged)} rows, "
          f"{merged['timestamps'].min()} -> {merged['timestamps'].max()}")

    featured, feature_list = add_features(merged, enabled_indicators)
    if featured.empty:
        print(f"{symbol}: no rows left after indicator warm-up, skipping")
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = final_output_path(symbol, output_dir)
    featured.to_csv(out_path, index=False)
    print(f"  wrote {len(featured)} rows x {len(feature_list)} features -> {out_path}")
    return out_path


def load_enabled_indicators(config_path: str | None) -> dict:
    if not config_path:
        return dict(DEFAULT_ENABLED_INDICATORS)

    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    indicators = (cfg.get("features") or {}).get("indicators") or {}
    return {
        name: indicators.get(name, DEFAULT_ENABLED_INDICATORS[name])
        for name in INDICATOR_ORDER
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download last-month→yesterday Binance 5m data (monthly+daily), "
            "merge, compute TA features, write one CSV per symbol"
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help=f"Symbols to fetch (e.g. BTCUSDT ADAUSDT). Common set: {DEFAULT_SYMBOLS}",
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help=f"Directory for final {{SYMBOL}}_kline_5min.csv files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training YAML whose features.indicators block is reused",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives that already exist on disk",
    )
    for name in INDICATOR_ORDER:
        flag = f"--no-{name.replace('_', '-')}"
        parser.add_argument(
            flag,
            dest=name,
            action="store_false",
            help=f"Disable {name} (ignored when --config is set)",
        )
        parser.set_defaults(**{name: True})
    return parser.parse_args()


def main():
    args = parse_args()
    start, end = prediction_window()

    if args.config:
        enabled_indicators = load_enabled_indicators(args.config)
    else:
        enabled_indicators = {name: getattr(args, name) for name in INDICATOR_ORDER}

    feature_list = build_feature_list(enabled_indicators)
    print(f"Prediction input window (UTC): {start} -> {end}")
    print(f"Symbols: {args.symbols}")
    print(f"Enabled indicators: {enabled_indicators}")
    print(f"Feature list ({len(feature_list)}): {feature_list}")
    print(f"Raw cache: {RAW_DIR}/<SYMBOL>/")
    print(f"Final output dir: {args.output_dir}")

    written = []
    for symbol in args.symbols:
        symbol = symbol.upper()
        out_path = process_symbol(
            symbol=symbol,
            start=start,
            end=end,
            output_dir=args.output_dir,
            enabled_indicators=enabled_indicators,
            force=args.force,
        )
        if out_path:
            written.append(out_path)

    print(f"\nDone: {len(written)}/{len(args.symbols)} file(s) written")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
