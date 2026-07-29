"""
Build prediction-ready CSVs for one or more symbols.

Default window (UTC): start of last calendar month through end of yesterday.

Steps per symbol:
  1. Download Binance Vision monthly 5m archives for every fully-finished
     month in the window (fall back to daily ZIPs if a monthly file is missing).
  2. Download Binance Vision daily 5m archives for the remaining days
     (typically the current month up through yesterday).
  3. Convert to Kronos OHLCV, merge monthly + daily, drop duplicates.
  4. Optionally (--live) fill gaps / today via Binance REST klines API so the
     series reaches the last closed 5m bar even when yesterday's Vision ZIP
     is not published yet.
  5. Optionally (--max-bars) keep only the most recent N bars (default off in
     archive mode; 2000 when --live).
  6. Compute technical-indicator features and write one final CSV.

Output default:  finetune_ta/data/{SYMBOL}_kline_5min.csv

Usage:
    python download_input_predict_data.py --symbols BTCUSDT ADAUSDT
    python download_input_predict_data.py --symbols ETHUSDT --force
    python download_input_predict_data.py --symbols BTCUSDT --live
    python download_input_predict_data.py --symbols BTCUSDT --live --max-bars 2000
    python download_input_predict_data.py --symbols BTCUSDT \\
        --config configs/config_multi_symbol_5m_ta.yaml
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

from clean_merge_data import KRONOS_COLUMNS, load_raw_month
from download_data import DATA_DIR, DEFAULT_SYMBOLS, INTERVAL, RAW_DIR
from indicators import (
    BASE_FEATURES,
    DEFAULT_ENABLED_INDICATORS,
    INDICATOR_ORDER,
    VOLUME_SMA_LENGTH,
    build_feature_list,
    compute_indicators,
)

MONTHLY_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DAILY_BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
REQUEST_TIMEOUT = 60
INTERVAL_MS = 300_000
API_MAX_LIMIT = 1000
# Extra raw bars so indicator warm-up (VOL_SMA_50 is longest) still leaves
# roughly --max-bars usable rows after NaN drop.
INDICATOR_WARMUP_BUFFER = VOLUME_SMA_LENGTH + 30
DEFAULT_LIVE_MAX_BARS = 2000
BARS_PER_DAY = 24 * 60 // 5  # 288 at 5m


def prediction_window(today: date | None = None) -> tuple[date, date]:
    """Return [start_of_last_month, yesterday] inclusive (UTC calendar dates)."""
    today = today or datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    first_of_this_month = today.replace(day=1)
    start = (first_of_this_month - timedelta(days=1)).replace(day=1)
    if yesterday < start:
        raise ValueError(
            f"Empty window: start={start}, yesterday={yesterday}. "
            "Cannot build prediction data before the 2nd of a month."
        )
    return start, yesterday


def live_prediction_window(
    max_bars: int = DEFAULT_LIVE_MAX_BARS,
    today: date | None = None,
) -> tuple[date, date]:
    """Short window sized for --max-bars, ending today (API fills through now)."""
    today = today or datetime.now(timezone.utc).date()
    # Cover max_bars + indicator warm-up, plus 2 days for late/missing Vision ZIPs.
    raw_needed = max_bars + INDICATOR_WARMUP_BUFFER
    days_needed = max(1, (raw_needed + BARS_PER_DAY - 1) // BARS_PER_DAY) + 2
    start = today - timedelta(days=days_needed - 1)
    return start, today


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


def filter_window(
    df: pd.DataFrame,
    start: date,
    end: date,
    end_inclusive_now: bool = False,
) -> pd.DataFrame:
    """Keep bars whose open time falls within [start 00:00, end of day or now]."""
    start_ts = pd.Timestamp(start)
    if end_inclusive_now:
        end_ts = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    else:
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (df["timestamps"] >= start_ts) & (df["timestamps"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


def _api_klines_to_dataframe(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=KRONOS_COLUMNS)

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["amount"] = df["quote_volume"].astype(float)
    return (
        df[KRONOS_COLUMNS]
        .sort_values("timestamps")
        .drop_duplicates(subset=["timestamps"])
        .reset_index(drop=True)
    )


def fetch_api_klines(
    symbol: str,
    start_time: datetime,
    end_time: datetime | None = None,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    """Fetch closed 5m spot klines via Binance REST, paginating forward."""
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": API_MAX_LIMIT,
        }
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        last_open_ms = batch[-1][0]
        next_cursor = last_open_ms + INTERVAL_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(batch) < API_MAX_LIMIT:
            break
        time.sleep(sleep_seconds)

    # Drop the in-progress candle (close_time still in the future).
    if rows and rows[-1][0] + INTERVAL_MS > now_ms:
        rows = rows[:-1]

    return _api_klines_to_dataframe(rows)


def fill_with_api(
    symbol: str,
    merged: pd.DataFrame,
    window_start: date,
) -> pd.DataFrame:
    """Append REST klines from after the last Vision bar (or window start) to now."""
    now = datetime.now(timezone.utc)
    if merged.empty:
        api_start = datetime.combine(window_start, datetime.min.time(), tzinfo=timezone.utc)
    else:
        last_ts = merged["timestamps"].max()
        api_start = pd.Timestamp(last_ts).to_pydatetime().replace(tzinfo=timezone.utc)
        api_start = api_start + timedelta(milliseconds=INTERVAL_MS)

    if api_start >= now:
        print("  api fill: already up to date, skipping")
        return merged

    print(f"  api fill: {api_start.isoformat()} -> {now.isoformat()} UTC")
    api_df = fetch_api_klines(symbol, start_time=api_start, end_time=now)
    if api_df.empty:
        print("  api fill: no new bars returned")
        return merged

    print(f"  api fill: got {len(api_df)} closed bars "
          f"({api_df['timestamps'].min()} -> {api_df['timestamps'].max()})")
    combined = pd.concat([merged, api_df], ignore_index=True)
    return (
        combined.sort_values("timestamps")
        .drop_duplicates(subset=["timestamps"], keep="last")
        .reset_index(drop=True)
    )


def trim_to_max_bars(df: pd.DataFrame, max_bars: int | None) -> pd.DataFrame:
    if not max_bars or max_bars <= 0 or len(df) <= max_bars:
        return df
    return df.iloc[-max_bars:].reset_index(drop=True)


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
    live: bool = False,
    max_bars: int | None = None,
) -> str | None:
    monthly_paths, daily_paths = download_symbol_window(symbol, start, end, force=force)
    all_paths = monthly_paths + daily_paths

    if all_paths:
        print(f"  merging {len(monthly_paths)} monthly + {len(daily_paths)} daily file(s)")
        merged = merge_raw_files(all_paths)
        merged = filter_window(merged, start, end, end_inclusive_now=live)
    else:
        merged = pd.DataFrame(columns=KRONOS_COLUMNS)
        print(f"  no Vision archives available for window")

    if live:
        merged = fill_with_api(symbol, merged, window_start=start)
        merged = filter_window(merged, start, end, end_inclusive_now=True)

    if merged.empty:
        print(f"{symbol}: no OHLCV rows after Vision/API merge, skipping")
        return None

    print(f"  merged OHLCV: {len(merged)} rows, "
          f"{merged['timestamps'].min()} -> {merged['timestamps'].max()}")

    # Keep a warm-up cushion for indicators, then trim again after feature drop.
    if max_bars and max_bars > 0:
        raw_keep = max_bars + INDICATOR_WARMUP_BUFFER
        before = len(merged)
        merged = trim_to_max_bars(merged, raw_keep)
        if before != len(merged):
            print(f"  trimmed raw to last {len(merged)} bars "
                  f"(max_bars={max_bars} + warmup buffer {INDICATOR_WARMUP_BUFFER})")

    featured, feature_list = add_features(merged, enabled_indicators)
    if featured.empty:
        print(f"{symbol}: no rows left after indicator warm-up, skipping")
        return None

    if max_bars and max_bars > 0:
        before = len(featured)
        featured = trim_to_max_bars(featured, max_bars)
        if before != len(featured):
            print(f"  trimmed features to last {len(featured)} bars (max_bars={max_bars})")

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
            "Download Binance 5m data (Vision monthly+daily, optional REST live fill), "
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
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Fill from last Vision bar through the latest closed 5m candle via "
            "Binance REST API (covers late/missing yesterday ZIPs and today). "
            "Uses a short ~max-bars window instead of last-month→yesterday."
        ),
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help=(
            f"Keep at most N most-recent 5m bars in the final CSV after indicators. "
            f"Default: {DEFAULT_LIVE_MAX_BARS} with --live, unlimited otherwise. "
            f"Pass 0 to disable the cap."
        ),
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

    if args.max_bars is None:
        max_bars = DEFAULT_LIVE_MAX_BARS if args.live else None
    elif args.max_bars <= 0:
        max_bars = None
    else:
        max_bars = args.max_bars

    if args.live:
        start, end = live_prediction_window(max_bars=max_bars or DEFAULT_LIVE_MAX_BARS)
    else:
        start, end = prediction_window()

    if args.config:
        enabled_indicators = load_enabled_indicators(args.config)
    else:
        enabled_indicators = {name: getattr(args, name) for name in INDICATOR_ORDER}

    feature_list = build_feature_list(enabled_indicators)
    mode = "live (Vision + REST API)" if args.live else "archive (Vision only)"
    print(f"Mode: {mode}")
    print(f"Prediction input window (UTC): {start} -> {end}"
          + (" (through now via API)" if args.live else ""))
    print(f"Max bars: {max_bars if max_bars else 'unlimited'}")
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
            live=args.live,
            max_bars=max_bars,
        )
        if out_path:
            written.append(out_path)

    print(f"\nDone: {len(written)}/{len(args.symbols)} file(s) written")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
