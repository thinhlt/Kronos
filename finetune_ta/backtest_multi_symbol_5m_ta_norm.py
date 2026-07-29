"""
Walk-forward backtest for the finetuned `multi_symbol_5m_ta_norm` experiment.

Loads tokenizer + basemodel from finetuned/multi_symbol_5m_ta_norm/ (via the
training config + local fallback), runs validation-split forecasts on local
data/*_kline_5min.csv, and writes results under backtests/multi_symbol_5m_ta_norm/.

Defaults to --pred-len 48 (~4h on 5m bars). Full training horizon is 288
(~1 day) but each window is much slower (~30m AR decode on CPU/MPS).

Usage:
    python backtest_multi_symbol_5m_ta_norm.py
    python backtest_multi_symbol_5m_ta_norm.py --max-windows 2
    python backtest_multi_symbol_5m_ta_norm.py --pred-len 288 --input data/BTCUSDT_kline_5min.csv
"""
from backtest import main_for_config

# Shorter horizon than training's 288; stride keeps ~4 windows/symbol on the
# local ~14k-bar CSVs so a full multi-symbol pass stays tractable.
_DEFAULTS = ["--pred-len", "48", "--stride", "288"]

if __name__ == "__main__":
    main_for_config(
        "configs/config_multi_symbol_5m_ta_norm.yaml",
        extra_defaults=_DEFAULTS,
    )
