# Hybrid LSTM + XGBoost baseline

Isolated **Baseline Pipeline** implementing the two-stage architecture from
[arXiv 2506.22055](https://arxiv.org/abs/2506.22055), adapted to this repo's
5-minute multi-symbol CSVs and feature stack. Design decisions live in
[`docs/research/hybrid-lstm-xgboost-baseline.md`](../docs/research/hybrid-lstm-xgboost-baseline.md).

## What it does

1. **Stage 1 — LSTM.** Supervised training on the 12-bar Horizon Return
   (`log(close[t+h-1] / close[t-1])`). The linear head is discarded at inference;
   only the final hidden state `z = h_n ∈ R^64` is kept.
2. **Stage 2 — XGBoost.** Maps `[z | Tabular Block]` → Horizon Return with squared
   error loss. The Tabular Block is last-bar normalised features + realised vol +
   calendar (hour, weekday).

Stages fit on **disjoint Stage Segments** of each file's train region (75% / 25%)
with Purge Gaps and Dev Tails for early stopping — see ADR 0004.

## Layout

```
hybrid_lstm_xgb/
├── configs/config_hybrid_5m_h12.yaml
├── indicators.py / feature_normalize.py   # copied from finetune_ta (isolation rule)
├── config_loader.py
├── windows.py
├── lstm_encoder.py / train_lstm.py
├── embed.py / train_xgb.py
├── predictor.py / metrics.py / backtest.py
└── train_hybrid.py
```

## Quick start

```bash
# from repo root, with the project venv active
# macOS: brew install libomp  (required by xgboost)
cd hybrid_lstm_xgb

# optional: refresh last-7-days prediction CSVs (uses local indicators.py)
python download_last_7d.py --symbols BTCUSDT ETHUSDT
# -> data/{SYMBOL}_kline_5min.csv

# forecast from each CSV's last bar (default model: 12×5m = 1h ahead)
python predict.py --config configs/config_hybrid_5m_h12.yaml
python predict.py --config configs/config_hybrid_5m_h12.yaml --input data/BTCUSDT_kline_5min.csv

python train_hybrid.py --config configs/config_hybrid_5m_h12.yaml
python backtest.py --config configs/config_hybrid_5m_h12.yaml
```

For a fast local smoke run on the short sample CSVs:

```bash
python train_hybrid.py --config configs/config_hybrid_smoke.yaml --force-lstm
python backtest.py --config configs/config_hybrid_smoke.yaml
```

On macOS, torch and xgboost both ship OpenMP runtimes. Entry points import
`omp_compat.py` first, which sets `KMP_DUPLICATE_LIB_OK=TRUE` and caps OpenMP /
BLAS threads to 1. Without that, stage-2 `xgboost.train` can segfault after the
LSTM stage. If you import both libraries yourself, import `omp_compat` first.

Artifacts land under `finetuned/<exp_name>/`:

- `lstm_best.pt` — Best Model for stage 1
- `xgb_model.json` — hybrid booster
- `xgb_only_model.json` — tabular-only ablation
- `feature_spec.json` — feature list, horizon, strides, `target_std`, …

If `lstm_best.pt` already exists, `train_hybrid.py` skips stage 1 and goes straight
to XGBoost. Pass `--force-lstm` to retrain.

## Stride caveat (D7)

- **Training** enumerates windows at `train_stride: 12` (non-overlapping targets).
- **Backtest** enumerates at `backtest_stride: 1` (every origin).

Error metrics (MAE / RMSE / MAPE / MinMax RMSE / directional accuracy) use every
origin. Compounding metrics (`strategy_total_return`, `buy_hold_total_return`) use
the non-overlapping subsample so overlapping horizons are not counted 12×. Reports
carry both `n_windows` and `n_windows_nonoverlap`.

## Ablations

Backtest scores three models on the same origins:

| Model | Meaning |
|---|---|
| `hybrid` | LSTM embedding + Tabular Block → XGBoost |
| `lstm_only` | Stage-1 linear head (no XGBoost) |
| `xgb_only` | Tabular Block → XGBoost (no `z`) |

Headline success criterion is **directional accuracy**, not MAPE (see research doc D10).
Pre-registered bar: >52% val directional accuracy, consistent across symbols.

## Feature-code drift (D9)

`indicators.py` and `feature_normalize.py` are copies of `finetune_ta/`. If a
benchmark comparison looks wrong, diff these two files against `finetune_ta/` first.

## Lookback note

Default `lookback_window: 128` (not Kronos's 1024). An LSTM cannot usefully carry
gradients across 1024 steps at hidden size 64; state that difference when comparing.
