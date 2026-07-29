"""Per-file Stage Segments, origin enumeration, and lazy window slicing.

See docs/research/hybrid-lstm-xgboost-baseline.md D5/D7 and ADR 0002 / 0004.
A window's history and target must lie entirely within one region of one file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from indicators import BASE_FEATURES, ensure_features


TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]


@dataclass(frozen=True)
class RegionBounds:
    """Half-open bar index range [start, end) usable for windows."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass
class FileWindows:
    """One Data File after indicator warm-up drop and region carving."""

    path: str
    features: np.ndarray  # (T, F) float32
    stamps: np.ndarray  # (T, 5) float32 minute/hour/weekday/day/month
    closes: np.ndarray  # (T,) float32
    timestamps: pd.Series
    regions: Dict[str, RegionBounds]


def _valid_origins(region: RegionBounds, lookback: int, horizon: int, stride: int) -> List[int]:
    """Origins i such that history [i-lookback, i) and target [i, i+horizon) ⊆ region."""
    if region.length <= 0 or stride < 1:
        return []
    first = region.start + lookback
    last = region.end - horizon  # inclusive
    if first > last:
        return []
    return list(range(first, last + 1, stride))


def _carve_fit_dev(
    seg_start: int,
    seg_end: int,
    dev_fraction: float,
    purge_bars: int,
    min_region_bars: int,
) -> Tuple[RegionBounds, RegionBounds]:
    """Split a Stage Segment into fit + Dev Tail with a Purge Gap between them.

    `min_region_bars` is the minimum length so that lookback+horizon windows fit
    (typically lookback + horizon). Both fit and Dev Tail are forced to at least
    that size when the segment is long enough; otherwise they may still be short
    and origin enumeration will simply return empty.
    """
    length = seg_end - seg_start
    empty_fit = RegionBounds(seg_start, seg_start)
    empty_dev = RegionBounds(seg_end, seg_end)
    if length <= 0:
        return empty_fit, empty_dev

    # Ideal Dev Tail from fraction, then floor at min_region_bars when possible.
    dev_len = max(1, int(round(length * dev_fraction)))
    if length >= 2 * min_region_bars + purge_bars:
        dev_len = max(dev_len, min_region_bars)
        # Leave room for fit + purge.
        dev_len = min(dev_len, length - min_region_bars - purge_bars)
    else:
        # Segment too short to guarantee both sides; still prefer a usable Dev Tail.
        dev_len = min(dev_len, max(0, length - 1))

    dev_start = seg_end - dev_len
    fit_end = max(seg_start, dev_start - purge_bars)
    return RegionBounds(seg_start, fit_end), RegionBounds(dev_start, seg_end)


def carve_regions(
    n_bars: int,
    train_ratio: float,
    lstm_fraction: float,
    purge_bars: int,
    lstm_dev_fraction: float,
    xgb_dev_fraction: float,
    lookback: int = 128,
    horizon: int = 12,
) -> Dict[str, RegionBounds]:
    """Carve train/val and the two Stage Segments (fit + Dev Tail each)."""
    train_end = int(n_bars * train_ratio)
    stage1_end = int(train_end * lstm_fraction)
    min_region = lookback + horizon

    # Purge between Stage Segments eats from the front of stage-2.
    stage2_start = min(train_end, stage1_end + purge_bars)

    s1_fit, s1_dev = _carve_fit_dev(
        0, stage1_end, lstm_dev_fraction, purge_bars, min_region
    )
    s2_fit, s2_dev = _carve_fit_dev(
        stage2_start, train_end, xgb_dev_fraction, purge_bars, min_region
    )
    val = RegionBounds(train_end, n_bars)

    return {
        "s1_fit": s1_fit,
        "s1_dev": s1_dev,
        "s2_fit": s2_fit,
        "s2_dev": s2_dev,
        "val": val,
        "train": RegionBounds(0, train_end),
        "stage1": RegionBounds(0, stage1_end),
        "stage2": RegionBounds(stage1_end, train_end),
    }


def load_file(
    data_path: str,
    feature_list: Sequence[str],
    enabled_indicators: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Load one CSV, compute indicators, drop warm-up NaNs."""
    df = pd.read_csv(data_path)
    if "timestamps" not in df.columns:
        raise ValueError(f"{data_path}: missing 'timestamps' column")

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    if df[BASE_FEATURES].isnull().any().any():
        print(f"Warning: Missing base OHLCV in {data_path}, forward filling")
        df[BASE_FEATURES] = df[BASE_FEATURES].ffill()

    df = ensure_features(df, list(feature_list), enabled_indicators)

    df["minute"] = df["timestamps"].dt.minute
    df["hour"] = df["timestamps"].dt.hour
    df["weekday"] = df["timestamps"].dt.weekday
    df["day"] = df["timestamps"].dt.day
    df["month"] = df["timestamps"].dt.month

    valid = df[list(feature_list)].notna().all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        print(f"Dropped {dropped} leading NaN warm-up rows ({os.path.basename(data_path)})")
    df = df.loc[valid].reset_index(drop=True)

    features = df[list(feature_list)].to_numpy(dtype=np.float32, copy=True)
    stamps = df[TIME_FEATURES].to_numpy(dtype=np.float32, copy=True)
    closes = df["close"].to_numpy(dtype=np.float32, copy=True)
    return features, stamps, closes, df["timestamps"].copy()


def build_file_windows(
    data_path: str,
    feature_list: Sequence[str],
    enabled_indicators: dict,
    train_ratio: float,
    lstm_fraction: float,
    purge_bars: int,
    lstm_dev_fraction: float,
    xgb_dev_fraction: float,
    lookback: int,
    horizon: int,
) -> FileWindows:
    features, stamps, closes, timestamps = load_file(data_path, feature_list, enabled_indicators)
    regions = carve_regions(
        n_bars=len(features),
        train_ratio=train_ratio,
        lstm_fraction=lstm_fraction,
        purge_bars=purge_bars,
        lstm_dev_fraction=lstm_dev_fraction,
        xgb_dev_fraction=xgb_dev_fraction,
        lookback=lookback,
        horizon=horizon,
    )
    return FileWindows(
        path=data_path,
        features=features,
        stamps=stamps,
        closes=closes,
        timestamps=timestamps,
        regions=regions,
    )


def collect_origins(
    files: Sequence[FileWindows],
    region_name: str,
    lookback: int,
    horizon: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Return list of (file_idx, origin_idx) for a named region."""
    pairs: List[Tuple[int, int]] = []
    for fi, fw in enumerate(files):
        region = fw.regions[region_name]
        for origin in _valid_origins(region, lookback, horizon, stride):
            pairs.append((fi, origin))
    return pairs


def horizon_log_return(closes: np.ndarray, origin: int, horizon: int) -> float:
    """y = log(close[origin + horizon - 1] / close[origin - 1])."""
    entry = float(closes[origin - 1])
    end = float(closes[origin + horizon - 1])
    if entry <= 0.0 or end <= 0.0:
        return float("nan")
    return float(np.log(end / entry))


def slice_history(
    fw: FileWindows,
    origin: int,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lookback, F) features and (lookback, 5) stamps for history ending at origin."""
    start = origin - lookback
    return fw.features[start:origin].copy(), fw.stamps[start:origin].copy()


def build_all_files(config) -> List[FileWindows]:
    files = []
    for path in config.data_paths:
        fw = build_file_windows(
            data_path=path,
            feature_list=config.feature_list,
            enabled_indicators=config.enabled_indicators,
            train_ratio=config.train_ratio,
            lstm_fraction=config.lstm_fraction,
            purge_bars=config.purge_bars,
            lstm_dev_fraction=config.lstm_dev_fraction,
            xgb_dev_fraction=config.xgb_dev_fraction,
            lookback=config.lookback_window,
            horizon=config.horizon,
        )
        files.append(fw)
        print(
            f"[{os.path.basename(path)}] bars={len(fw.features)} "
            f"s1_fit={fw.regions['s1_fit'].length} s1_dev={fw.regions['s1_dev'].length} "
            f"s2_fit={fw.regions['s2_fit'].length} s2_dev={fw.regions['s2_dev'].length} "
            f"val={fw.regions['val'].length}"
        )
    return files


def summarize_origin_counts(
    files: Sequence[FileWindows],
    lookback: int,
    horizon: int,
    stride: int,
) -> Dict[str, int]:
    names = ["s1_fit", "s1_dev", "s2_fit", "s2_dev", "val"]
    return {
        name: len(collect_origins(files, name, lookback, horizon, stride))
        for name in names
    }
