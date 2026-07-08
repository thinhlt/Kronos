"""
Standalone utility: compute technical indicators for one or more raw OHLCV
CSVs and save the augmented result(s) to disk, for inspecting/sanity-checking
the indicator math before (or without) running any training.

This step is optional -- CustomKlineDataset computes indicators inline on
load if they're not already present. Use this script to eyeball the output,
diff against a reference implementation, or cache the augmented CSV(s) so
subsequent dataset loads skip recomputation.

Usage:
    python prepare_dataset.py --config configs/config_ali09988_candle-5min_ta.yaml
    python prepare_dataset.py --input data/raw.csv --output data/raw_with_indicators.csv
    python prepare_dataset.py --input data/*.csv   # each file processed independently
"""
import argparse
import os
import sys

import pandas as pd

from indicators import BASE_FEATURES, INDICATOR_ORDER, build_feature_list, compute_indicators


def load_enabled_indicators_from_config(config_path):
    from config_loader import CustomFinetuneConfig
    config = CustomFinetuneConfig(config_path)
    return config.data_paths, config.enabled_indicators


def process_one_file(data_path, output_path, enabled_indicators, feature_list):
    print(f"\nReading raw data from: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    if df[BASE_FEATURES].isnull().any().any():
        print("Warning: missing values found in base OHLCV columns, forward-filling")
        df[BASE_FEATURES] = df[BASE_FEATURES].fillna(method='ffill')

    df = compute_indicators(df, enabled_indicators)

    n_rows = len(df)
    warmup_mask = df[feature_list].notna().all(axis=1)
    n_valid = int(warmup_mask.sum())
    print(f"Total rows: {n_rows}, usable after warm-up NaN drop: {n_valid} (dropped {n_rows - n_valid})")

    df = df.loc[warmup_mask].reset_index(drop=True)

    print("Per-column summary stats (post warm-up):")
    print(df[feature_list].describe().T[['mean', 'std', 'min', 'max']])

    df.to_csv(output_path, index=False)
    print(f"Saved augmented CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute and save technical-indicator-augmented OHLCV data')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML to read data_path(s) and enabled indicators from')
    parser.add_argument('--input', type=str, nargs='+', default=None,
                        help='One or more raw OHLCV CSV paths, or glob pattern(s) (overrides --config data_path)')
    parser.add_argument('--output', type=str, default=None,
                        help='Where to write the augmented CSV when a single --input file is given '
                             '(default: <input>_with_indicators.csv). Ignored for multiple files, '
                             'which always use the per-file default naming.')
    for name in INDICATOR_ORDER:
        parser.add_argument(f'--no-{name.replace("_", "-")}', dest=name, action='store_false',
                            help=f'Disable {name} (only used without --config)')
        parser.set_defaults(**{name: True})
    args = parser.parse_args()

    if args.config:
        from config_loader import resolve_data_paths
        data_paths, enabled_indicators = load_enabled_indicators_from_config(args.config)
        if args.input:
            data_paths = resolve_data_paths(args.input)
    elif args.input:
        from config_loader import resolve_data_paths
        data_paths = resolve_data_paths(args.input)
        enabled_indicators = {name: getattr(args, name) for name in INDICATOR_ORDER}
    else:
        parser.error('Either --config or --input must be provided')
        return

    feature_list = build_feature_list(enabled_indicators)
    print(f"Enabled indicators: {enabled_indicators}")
    print(f"Feature list ({len(feature_list)} dims): {feature_list}")
    print(f"Processing {len(data_paths)} file(s)")

    if len(data_paths) == 1 and args.output:
        output_paths = [args.output]
    else:
        output_paths = []
        for data_path in data_paths:
            base, ext = os.path.splitext(data_path)
            output_paths.append(f"{base}_with_indicators{ext}")

    for data_path, output_path in zip(data_paths, output_paths):
        process_one_file(data_path, output_path, enabled_indicators, feature_list)


if __name__ == '__main__':
    main()
