"""
Download historical 5m klines for a fixed set of USDT spot pairs from
Binance's public bulk-data archive (data.binance.vision), one monthly
ZIP per symbol per month.

Binance Vision only publishes a month's ZIP once that month is fully
complete, so this script never requests the current, still-in-progress
month -- it stops at the last fully-finished month automatically.

Raw files are saved uncombined, one row-for-row copy of Binance's own
CSV per symbol per month, under:

    finetune_ta/data/raw/{SYMBOL}/{SYMBOL}-5m-YYYY-MM.csv

Run clean_merge_data.py afterwards to convert them to Kronos format and
merge each symbol's months into a single training-ready CSV.

Usage:
    python download_data.py
    python download_data.py --symbols BTCUSDT ETHUSDT --start 2024-01
    python download_data.py --end 2025-06
"""
import argparse
import io
import os
import zipfile
from datetime import date

import requests

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "ZECUSDT",
    "AVAXUSDT",
]

INTERVAL = "5m"
DEFAULT_START = "2020-09"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def month_range(start: str, end: str):
    """Yield (year, month) tuples from start to end (inclusive), both 'YYYY-MM'."""
    start_year, start_month = (int(p) for p in start.split("-"))
    end_year, end_month = (int(p) for p in end.split("-"))

    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def last_complete_month() -> str:
    """Return the most recent fully-finished month as 'YYYY-MM' (UTC today)."""
    today = date.today()
    year, month = today.year, today.month - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def monthly_zip_url(symbol: str, year: int, month: int) -> str:
    return f"{BASE_URL}/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"


def raw_csv_path(symbol: str, year: int, month: int) -> str:
    return os.path.join(RAW_DIR, symbol, f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.csv")


def download_month(symbol: str, year: int, month: int, force: bool = False) -> str:
    """Download and extract one symbol-month. Returns 'ok', 'skipped', or 'missing'."""
    dest_path = raw_csv_path(symbol, year, month)
    if os.path.exists(dest_path) and not force:
        return "skipped"

    url = monthly_zip_url(symbol, year, month)
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        # Symbol didn't exist yet, or wasn't listed, for this month.
        return "missing"
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        inner_name = zf.namelist()[0]
        csv_bytes = zf.read(inner_name)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(csv_bytes)
    return "ok"


def download_symbol(symbol: str, start: str, end: str, force: bool = False) -> dict:
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    missing_months = []
    for year, month in month_range(start, end):
        status = download_month(symbol, year, month, force=force)
        counts[status] += 1
        if status == "missing":
            missing_months.append(f"{year:04d}-{month:02d}")

    print(f"{symbol}: {counts['ok']} downloaded, {counts['skipped']} already present, "
          f"{counts['missing']} missing")
    if missing_months:
        print(f"  missing months (no listing on Binance Vision): {', '.join(missing_months)}")
    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bulk-download Binance monthly 5m kline archives for a fixed pair list"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                         help=f"Symbols to fetch (default: {DEFAULT_SYMBOLS})")
    parser.add_argument("--start", default=DEFAULT_START,
                         help=f"Start month, YYYY-MM (default: {DEFAULT_START})")
    parser.add_argument("--end", default=None,
                         help="End month, YYYY-MM (default: last fully-completed month)")
    parser.add_argument("--force", action="store_true",
                         help="Re-download months that already exist on disk")
    return parser.parse_args()


def main():
    args = parse_args()
    end = args.end or last_complete_month()

    print(f"Fetching {INTERVAL} klines for {len(args.symbols)} symbols, "
          f"{args.start} -> {end} (last complete month)")
    print(f"Raw files saved under: {RAW_DIR}/<SYMBOL>/")

    for symbol in args.symbols:
        download_symbol(symbol, args.start, end, force=args.force)


if __name__ == "__main__":
    main()
