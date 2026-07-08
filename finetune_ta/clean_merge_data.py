"""
Clean up raw Binance monthly kline archives (downloaded by download_data.py)
into Kronos-format CSVs, merging all months for each symbol into a single
file.

Raw input:  finetune_ta/data/raw/{SYMBOL}/{SYMBOL}-5m-YYYY-MM.csv
            (Binance's own 12-column, headerless kline format; open_time may
            be in milliseconds, microseconds, or nanoseconds depending on
            when the archive was generated -- detected per file.)

Kronos output: finetune_ta/data/{SYMBOL}_5m.csv
            columns: timestamps, open, high, low, close, volume, amount
            (amount = quote-asset volume, matching examples/get_binance_btcusdt.py)

Raw files are left untouched on disk after merging, so this script can be
re-run any time (e.g. after downloading more months) without re-fetching.

Usage:
    python clean_merge_data.py
    python clean_merge_data.py --symbols BTCUSDT ETHUSDT
"""
import argparse
import os

import pandas as pd

from download_data import DEFAULT_SYMBOLS, INTERVAL, RAW_DIR, DATA_DIR

RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
KRONOS_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def detect_time_unit(value: int) -> str:
    """Binance Vision archives switched from ms to us/ns timestamps over time."""
    if value > 1e17:
        return "ns"
    if value > 1e14:
        return "us"
    return "ms"


def load_raw_month(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=RAW_COLUMNS)
    unit = detect_time_unit(int(df["open_time"].iloc[0]))
    df["timestamps"] = pd.to_datetime(df["open_time"], unit=unit, utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["amount"] = df["quote_volume"].astype(float)
    return df[KRONOS_COLUMNS]


def find_raw_months(symbol: str) -> list:
    """Return sorted list of (year, month, path) for all raw files on disk."""
    symbol_dir = os.path.join(RAW_DIR, symbol)
    if not os.path.isdir(symbol_dir):
        return []

    prefix = f"{symbol}-{INTERVAL}-"
    months = []
    for name in os.listdir(symbol_dir):
        if not (name.startswith(prefix) and name.endswith(".csv")):
            continue
        ym = name[len(prefix):-len(".csv")]
        year, month = (int(p) for p in ym.split("-"))
        months.append((year, month, os.path.join(symbol_dir, name)))

    return sorted(months)


def missing_months_within(months: list) -> list:
    """Given sorted (year, month, path) tuples, find gaps between first and last."""
    if len(months) < 2:
        return []

    present = {(y, m) for y, m, _ in months}
    missing = []
    year, month = months[0][0], months[0][1]
    end_year, end_month = months[-1][0], months[-1][1]
    while (year, month) <= (end_year, end_month):
        if (year, month) not in present:
            missing.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return missing


def merge_symbol(symbol: str, output_dir: str) -> bool:
    months = find_raw_months(symbol)
    if not months:
        print(f"{symbol}: no raw files found under {RAW_DIR}/{symbol}/, skipping")
        return False

    gaps = missing_months_within(months)
    if gaps:
        print(f"{symbol}: warning, missing months in range (merging anyway): {', '.join(gaps)}")

    frames = [load_raw_month(path) for _, _, path in months]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)

    output_path = os.path.join(output_dir, f"{symbol}_{INTERVAL}.csv")
    merged.to_csv(output_path, index=False)

    span = f"{merged['timestamps'].min()} -> {merged['timestamps'].max()}"
    print(f"{symbol}: merged {len(months)} month file(s), {len(merged)} rows, {span} -> {output_path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw Binance monthly klines to Kronos-format CSV, merged per symbol"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                         help=f"Symbols to merge (default: {DEFAULT_SYMBOLS})")
    parser.add_argument("--output-dir", default=DATA_DIR,
                         help=f"Where to write merged per-symbol CSVs (default: {DATA_DIR})")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ok = 0
    for symbol in args.symbols:
        if merge_symbol(symbol, args.output_dir):
            ok += 1

    print(f"\nDone: {ok}/{len(args.symbols)} symbols merged")


if __name__ == "__main__":
    main()
