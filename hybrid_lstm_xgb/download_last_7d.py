"""
Download the last 7 days of 5m history, refine with TA features, save to data/.

Self-contained under hybrid_lstm_xgb/ (uses local indicators.py). Fetches via
Binance REST through the latest closed 5m bar, computes TA features, then
trims to ~7 days of usable rows.

Output default:  hybrid_lstm_xgb/data/{SYMBOL}_kline_5min.csv

Usage:
    python download_last_7d.py
    python download_last_7d.py --symbols BTCUSDT ETHUSDT
    python download_last_7d.py --days 7 --output-dir data
    python download_last_7d.py --config configs/config_hybrid_5m_h12.yaml
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yaml

from indicators import (
    BASE_FEATURES,
    DEFAULT_ENABLED_INDICATORS,
    INDICATOR_ORDER,
    VOLUME_SMA_LENGTH,
    build_feature_list,
    compute_indicators,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DEFAULT_DAYS = 7
DEFAULT_SYMBOLS = [
    "ADAUSDT",
    "ATOMUSDT",
    "BATUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DASHUSDT",
    "ETCUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "RVNUSDT",
    "SANDUSDT",
    "SOLUSDT",
    "TRXUSDT",
    "XRPUSDT",
    "XTZUSDT",
]

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
REQUEST_TIMEOUT = 60
INTERVAL = "5m"
INTERVAL_MS = 300_000
API_MAX_LIMIT = 1000
BARS_PER_DAY = 24 * 60 // 5  # 288 at 5m
INDICATOR_WARMUP_BUFFER = VOLUME_SMA_LENGTH + 30

KRONOS_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


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


def load_enabled_indicators(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    indicators = (cfg.get("features") or {}).get("indicators") or {}
    return {
        name: indicators.get(name, DEFAULT_ENABLED_INDICATORS[name])
        for name in INDICATOR_ORDER
    }


def process_symbol(
    symbol: str,
    days: float,
    output_dir: str,
    enabled_indicators: dict,
) -> str | None:
    now = datetime.now(timezone.utc)
    # Fetch warm-up cushion before the keep-window so indicators are valid
    # from the first bar we retain.
    raw_days = days + (INDICATOR_WARMUP_BUFFER / BARS_PER_DAY) + 0.25
    start_time = now - timedelta(days=raw_days)
    keep_bars = max(1, int(round(days * BARS_PER_DAY)))

    print(f"\n{symbol}: fetching ~{raw_days:.2f}d raw "
          f"({start_time.isoformat()} -> {now.isoformat()} UTC)")
    merged = fetch_api_klines(symbol, start_time=start_time, end_time=now)
    if merged.empty:
        print(f"{symbol}: no OHLCV rows from API, skipping")
        return None

    print(f"  raw OHLCV: {len(merged)} rows, "
          f"{merged['timestamps'].min()} -> {merged['timestamps'].max()}")

    featured, feature_list = add_features(merged, enabled_indicators)
    if featured.empty:
        print(f"{symbol}: no rows left after indicator warm-up, skipping")
        return None

    before = len(featured)
    featured = trim_to_max_bars(featured, keep_bars)
    if before != len(featured):
        print(f"  trimmed to last {len(featured)} bars (~{days:g}d)")

    os.makedirs(output_dir, exist_ok=True)
    out_path = final_output_path(symbol, output_dir)
    featured.to_csv(out_path, index=False)
    print(f"  wrote {len(featured)} rows x {len(feature_list)} features -> {out_path}")
    print(f"  range: {featured['timestamps'].min()} -> {featured['timestamps'].max()}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download last N days of Binance 5m klines via REST, compute TA "
            "features, write one CSV per symbol under data/"
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help=f"Symbols to fetch (default: {DEFAULT_SYMBOLS})",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=DEFAULT_DAYS,
        help=f"How many days of history to keep in the final CSV (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help=f"Directory for final {{SYMBOL}}_kline_5min.csv files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional hybrid YAML whose features.indicators block is reused",
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
    if args.days <= 0:
        raise SystemExit("--days must be > 0")

    if args.config:
        enabled_indicators = load_enabled_indicators(args.config)
    else:
        enabled_indicators = {name: getattr(args, name) for name in INDICATOR_ORDER}

    feature_list = build_feature_list(enabled_indicators)
    print(f"Window: last {args.days:g} day(s) through latest closed 5m bar (UTC)")
    print(f"Symbols: {args.symbols}")
    print(f"Enabled indicators: {enabled_indicators}")
    print(f"Feature list ({len(feature_list)}): {feature_list}")
    print(f"Final output dir: {args.output_dir}")

    written = []
    for symbol in args.symbols:
        out_path = process_symbol(
            symbol=symbol.upper(),
            days=args.days,
            output_dir=args.output_dir,
            enabled_indicators=enabled_indicators,
        )
        if out_path:
            written.append(out_path)

    print(f"\nDone: {len(written)}/{len(args.symbols)} file(s) written")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
