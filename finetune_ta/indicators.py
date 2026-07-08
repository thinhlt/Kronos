"""
Technical-indicator feature engineering, shared by both the dataset (inline,
computed lazily if columns are missing) and `prepare_dataset.py` (standalone,
for sanity-checking the computation).

Indicator parameters are fixed to industry-standard defaults and are not
config-driven -- only *which* indicators are enabled is configurable (see
`config_loader.CustomFinetuneConfig.enabled_indicators`).
"""
import pandas as pd
import pandas_ta_classic as ta

BASE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']

KDJ_LENGTH, KDJ_SIGNAL = 9, 3
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLLINGER_LENGTH, BOLLINGER_STD = 20, 2.0

HEIKIN_ASHI_COLUMNS = ['HA_open', 'HA_high', 'HA_low', 'HA_close']
KDJ_COLUMNS = [f'K_{KDJ_LENGTH}_{KDJ_SIGNAL}', f'D_{KDJ_LENGTH}_{KDJ_SIGNAL}', f'J_{KDJ_LENGTH}_{KDJ_SIGNAL}']
MACD_COLUMNS = [
    f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
    f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
    f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
]
# Only the scale-free derivatives (%B, bandwidth) are kept -- the raw
# upper/middle/lower bands are dropped since %B/bandwidth already normalize
# the band's relationship to price.
BOLLINGER_COLUMNS = [f'BBB_{BOLLINGER_LENGTH}_{BOLLINGER_STD}', f'BBP_{BOLLINGER_LENGTH}_{BOLLINGER_STD}']

# Order matters: this defines the tokenizer's d_in layout.
INDICATOR_ORDER = ['heikin_ashi', 'kdj', 'macd', 'bollinger']
INDICATOR_COLUMNS = {
    'heikin_ashi': HEIKIN_ASHI_COLUMNS,
    'kdj': KDJ_COLUMNS,
    'macd': MACD_COLUMNS,
    'bollinger': BOLLINGER_COLUMNS,
}

DEFAULT_ENABLED_INDICATORS = {name: True for name in INDICATOR_ORDER}


def build_feature_list(enabled_indicators: dict) -> list:
    """Base OHLCV+amount columns, followed by enabled indicator columns in a fixed order."""
    feature_list = list(BASE_FEATURES)
    for name in INDICATOR_ORDER:
        if enabled_indicators.get(name, False):
            feature_list.extend(INDICATOR_COLUMNS[name])
    return feature_list


def compute_indicators(df: pd.DataFrame, enabled_indicators: dict) -> pd.DataFrame:
    """Compute and append the enabled technical-indicator columns to `df`.

    `df` must already contain clean (non-NaN) `open, high, low, close` columns.
    Indicators that need a warm-up period (MACD, Bollinger, KDJ) will produce
    leading NaNs -- callers are expected to drop those rows before use.
    """
    df = df.copy()
    required_ohlc = {'open', 'high', 'low', 'close'}
    missing = required_ohlc - set(df.columns)
    if missing:
        raise ValueError(f"Cannot compute technical indicators, missing columns: {sorted(missing)}")

    if enabled_indicators.get('heikin_ashi', False) and not set(HEIKIN_ASHI_COLUMNS).issubset(df.columns):
        ha = df.ta.ha()
        df[HEIKIN_ASHI_COLUMNS] = ha[HEIKIN_ASHI_COLUMNS]

    if enabled_indicators.get('kdj', False) and not set(KDJ_COLUMNS).issubset(df.columns):
        kdj = df.ta.kdj(length=KDJ_LENGTH, signal=KDJ_SIGNAL)
        df[KDJ_COLUMNS] = kdj[KDJ_COLUMNS]

    if enabled_indicators.get('macd', False) and not set(MACD_COLUMNS).issubset(df.columns):
        macd = df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        df[MACD_COLUMNS] = macd[MACD_COLUMNS]

    if enabled_indicators.get('bollinger', False) and not set(BOLLINGER_COLUMNS).issubset(df.columns):
        bb = df.ta.bbands(length=BOLLINGER_LENGTH, std=BOLLINGER_STD)
        df[BOLLINGER_COLUMNS] = bb[BOLLINGER_COLUMNS]

    return df


def ensure_features(df: pd.DataFrame, feature_list: list, enabled_indicators: dict) -> pd.DataFrame:
    """Return `df` with every column in `feature_list` present, computing
    whichever indicator columns are missing. If the CSV was produced by
    `prepare_dataset.py` (columns already present), this is a no-op."""
    missing_cols = [c for c in feature_list if c not in df.columns]
    if missing_cols:
        df = compute_indicators(df, enabled_indicators)

    still_missing = [c for c in feature_list if c not in df.columns]
    if still_missing:
        raise ValueError(f"Feature columns still missing after computing indicators: {still_missing}")
    return df
