# Kronos Finetuning with Technical Indicators (`finetune_ta`)

A finetuning pipeline for Kronos on a single OHLCV CSV, augmented with
technical indicators as extra feature channels: **KDJ**, **MACD**,
**Heikin-Ashi**, **Bollinger Bands**, **ATR**, and **volume SMA**. Fully
isolated from `finetune_csv/` and `finetune/` -- no shared config, data, or
training code -- so it never interferes with those pipelines.

See [`CONTEXT.md`](../CONTEXT.md) for terminology (Base Features vs.
Technical Indicator Features, `d_in`, ...) and
[`docs/adr/0001-technical-indicator-tokenizer-from-scratch.md`](../docs/adr/0001-technical-indicator-tokenizer-from-scratch.md)
for why the tokenizer is trained from scratch here.

## Feature set

| Group | Columns | Notes |
|---|---|---|
| Base | `open, high, low, close, volume, amount` | Raw CSV columns |
| Heikin-Ashi | `HA_open, HA_high, HA_low, HA_close` | Smoothed synthetic candles, alongside (not replacing) raw OHLC |
| KDJ | `K_9_3, D_9_3, J_9_3` | length=9, signal=3 |
| MACD | `MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9` | fast=12, slow=26, signal=9 |
| Bollinger | `BBM_20_2.0, BBU_20_2.0, BBL_20_2.0` | middle, upper, lower; length=20, std=2.0 |
| ATR | `ATRr_14` | Average True Range (Wilder); length=14 |
| Volume SMA | `VOL_SMA_50` | rolling mean of volume over last 50 bars |

All 21 columns are normalized on GPU after each training batch is transferred
(not on disk, and not in ``Dataset.__getitem__``): OHLC / Heikin-Ashi /
Bollinger / ATR are divided by the window's first open; MACD is `(value / open)`
then per-window z-score; KDJ, volume, amount, and volume SMA are per-window
z-score only; then clip (see `data.clip` and `feature_normalize.py`).
Toggle indicators on/off via the `features.indicators` block in the YAML
config; each indicator's internal parameters are fixed (not config-driven)
-- see `indicators.py`.

## Data acquisition

Two scripts build training data from scratch for a fixed list of 10 USDT
spot pairs (`BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT,
TRXUSDT, ZECUSDT, AVAXUSDT`), 5m interval, from 2024-01-01 onward:

```bash
python download_data.py       # bulk-downloads monthly kline archives from data.binance.vision
python clean_merge_data.py    # converts to Kronos format, merges months into one CSV per pair
```

For **prediction input** (last calendar month through yesterday, monthly +
daily archives, indicators included in one final file):

```bash
python download_input_predict_data.py --symbols BTCUSDT ADAUSDT
# -> data/{SYMBOL}_kline_5min.csv

# When yesterday's Vision ZIP is late / you need bars through the last closed
# 5m candle: merge a short daily history with REST API fill, cap at 2000 bars.
python download_input_predict_data.py --symbols BTCUSDT ADAUSDT --live
python download_input_predict_data.py --symbols BTCUSDT --live --max-bars 1500
```

For a short **last-7-days** prediction window (REST API only, TA features
included):

```bash
python download_last_7d.py
python download_last_7d.py --symbols BTCUSDT ETHUSDT
# -> data/{SYMBOL}_kline_5min.csv
```

- `download_data.py` fetches Binance's public monthly ZIP archives (not the
  REST API -- far fewer requests for multi-year history) into
  `data/raw/{SYMBOL}/{SYMBOL}-5m-YYYY-MM.csv`. It only requests months that
  are fully complete on Binance Vision, is resumable (skips months already on
  disk), and warns (without failing) on any symbol-month with no listing.
- `clean_merge_data.py` reads those raw files -- auto-detecting whether each
  file's `open_time` is in ms, us, or ns, since Binance Vision has changed
  this over time -- normalizes to the Kronos columns (`timestamps, open,
  high, low, close, volume, amount`), and merges all months for a symbol into
  `data/{SYMBOL}_5m.csv`. Raw per-month files are kept on disk so re-running
  after downloading more months doesn't require re-fetching. Missing months
  are reported as warnings but don't block the merge.

## Training on multiple files / symbols at once

`data.data_path` in the config accepts a single file (as before), a glob
pattern, or a YAML list of paths/patterns:

```yaml
data:
  data_path: "data/*_kline_5min.csv"   # every matching file trains jointly
```

Each file is loaded, indicator-augmented, and time-split (train/val/test)
**independently** -- a training window is never built from more than one
file, so a sample is never partly BTCUSDT and partly ETHUSDT. `train_ratio`/
`val_ratio` apply per file (each symbol contributes its own trailing slice to
validation), and the tokenizer/predictor both train on the union of all
files' windows. See
[`docs/adr/0002-multi-file-windows-never-cross-files.md`](../docs/adr/0002-multi-file-windows-never-cross-files.md).

`prepare_dataset.py` also accepts multiple/glob inputs (`--input data/*.csv`),
processing each file independently and writing one `*_with_indicators.csv`
per input.

## Checkpointing and resuming

Two different things get saved during training, for two different purposes:

| | `best_model/` | `checkpoint_last.pt` |
|---|---|---|
| Written | only when validation loss improves | every epoch (and every `checkpoint_every_n_steps` steps, if set) |
| Contains | model weights only (HF `safetensors` + `config.json`) | model + optimizer + LR scheduler + epoch + best-val-loss + RNG state |
| Used for | inference, and handing the tokenizer to phase 2 | resuming a training run that stopped partway through |

Resuming is automatic: if `training.resume: true` (the default) and a
`checkpoint_last.pt` exists in a phase's save directory, `finetune_tokenizer.py`,
`finetune_base_model.py`, and `train_sequential.py` all pick it up and
continue from the next epoch. Re-running the exact same command after an
interruption is enough:

```bash
python train_sequential.py --config configs/config_ali09988_candle-5min_ta.yaml
```

Pass `--no-resume` to ignore an existing checkpoint and start a phase from
epoch 0 (the `best_model/` from a previous run, if any, is left alone).

Mid-epoch checkpoints (`training.checkpoint_every_n_steps`) record the
*previous* completed epoch as the resume point, not partial progress within
the interrupted epoch -- resuming just re-runs that epoch from its start
rather than attempting exact mid-epoch resumption. See
[`docs/adr/0003-resumable-checkpoints-for-kaggle-sessions.md`](../docs/adr/0003-resumable-checkpoints-for-kaggle-sessions.md).

## Mixed precision (AMP) training

Set `training.use_amp: true` to run most ops in a lower-precision dtype on
CUDA GPUs, which typically speeds up training and cuts memory use (letting
you raise `tokenizer_batch_size` / `basemodel_batch_size`). It has no effect when training on CPU.

```yaml
training:
  tokenizer_batch_size: 256   # tokenizer is lighter; raise if VRAM allows
  basemodel_batch_size: 32    # predictor is heavier; keep smaller to avoid OOM
  use_amp: true
  amp_dtype: "fp16"   # or "bf16" on Ampere-or-newer GPUs (RTX 30xx+/A100)
```

`fp16` works on most CUDA GPUs (including Kaggle's T4/P100) and uses
gradient scaling (`torch.cuda.amp.GradScaler`) to avoid gradient underflow;
`bf16` needs an Ampere-or-newer GPU but needs no scaling. The loss-scaler
state is included in `checkpoint_last.pt`, so resuming an AMP run doesn't
reset its warmup. You can also override the config from the command line
with `--amp` / `--no-amp` on `train_sequential.py`.

Legacy single `training.batch_size` still works as a fallback for both phases
when the phase-specific keys are omitted.

## Predictor candle logic loss

Optional self-consistency penalty on the **basemodel only** (tokenizer train
unchanged). Soft-decodes s1/s2 logits through the frozen tokenizer decoder and
penalizes invalid OHLC envelopes and Heikin-Ashi formula mismatches. No gap
term; no supervised continuous MSE for these penalties.

```yaml
training:
  logic_loss:
    enabled: true              # CE + logic on every epoch when true
    weight: 0.1
    ohlc_weight: 1.0
    ha_weight: 1.0
    # VRAM mitigations (soft-decode is otherwise very heavy at lookback=1024):
    max_timesteps: 128         # contiguous window subsample before decode
    use_checkpoint: true       # checkpoint frozen tokenizer decoder layers
```

Validation / best-model selection stays CE-only.

## Running on Kaggle

Kaggle Notebook sessions are capped (roughly 9-12h interactive, ~30 GPU-hours/
week on the free tier) and `/kaggle/working` starts empty every session --
so a multi-epoch, multi-symbol run will usually need to resume **across**
sessions, not just across a crash within one. See
[`configs/config_kaggle_multi_symbol_5m_ta.yaml`](configs/config_kaggle_multi_symbol_5m_ta.yaml)
for a starting-point config. Workflow:

1. Clone/upload the repo into `/kaggle/working/Kronos` and run
   `train_sequential.py` as usual. Checkpoints accumulate in
   `/kaggle/working/Kronos/finetune_ta/finetuned/<exp_name>/`.
2. Before the session ends, **Save Version** (commit) the notebook so
   `/kaggle/working` is preserved as that version's output.
3. In your next session, add the previous version's output as a Notebook
   input (Kaggle mounts it read-only under `/kaggle/input/<slug>/...`), and
   set `model_paths.resume_from_path` (or `--resume-from-path`) to the
   experiment directory inside it, e.g.
   `/kaggle/input/<slug>/Kronos/finetune_ta/finetuned/kaggle_multi_symbol_5m_ta`.
4. Run the same `train_sequential.py` command again. On startup, if
   `base_save_path` has no progress yet, its contents are copied in once from
   `resume_from_path`, and training resumes from `checkpoint_last.pt` as
   normal -- new checkpoints are written to the writable `/kaggle/working`.

Repeat steps 2-4 across sessions until `tokenizer_epochs`/`basemodel_epochs`
are reached.

## Setup

```bash
pip install -r ../requirements.txt   # includes pandas-ta-classic
```

`pandas-ta-classic` (import name `pandas_ta_classic`) is the actively
maintained community fork of the original `pandas-ta`, which was pulled from
PyPI/GitHub in 2025.

## Usage

1. (Optional) Sanity-check the indicator computation and inspect the
   augmented data before training:

   ```bash
   python prepare_dataset.py --config configs/config_ali09988_candle-5min_ta.yaml
   ```

   This is purely a diagnostic step -- the dataset class computes indicators
   inline automatically if they're not already present in the CSV.

2. Run the full tokenizer -> predictor pipeline:

   ```bash
   python train_sequential.py --config configs/config_ali09988_candle-5min_ta.yaml
   ```

   For multi-symbol training (see "Training on multiple files / symbols at
   once" below), use a config whose `data.data_path` is a glob/list, e.g.
   [`configs/config_kaggle_multi_symbol_5m_ta.yaml`](configs/config_kaggle_multi_symbol_5m_ta.yaml).

   Or run phases individually:

   ```bash
   python finetune_tokenizer.py --config configs/config_ali09988_candle-5min_ta.yaml
   python finetune_base_model.py --config configs/config_ali09988_candle-5min_ta.yaml
   ```

## Why the tokenizer trains from scratch

Enabling any indicator changes `d_in` away from 6, so the pretrained
`NeoQuasar/Kronos-Tokenizer-base` checkpoint's `embed`/`head` layers (sized
for `d_in=6`) can't be loaded. `experiment.pre_trained_tokenizer: false` in
the config trains a randomly-initialized tokenizer (same architecture
hyperparameters as the pretrained one, just with `d_in` matching your enabled
feature set) from scratch.

The **predictor**, by contrast, never sees raw features directly -- it only
consumes the tokenizer's quantized tokens plus calendar features -- so
`experiment.pre_trained_predictor: true` still works regardless of `d_in`.

If you change which indicators are enabled, you must retrain both the
tokenizer (phase 1) and the predictor (phase 2), since the predictor is
trained against a specific tokenizer's token vocabulary.

## Inference

`predictor.py: KronosPredictorTA` mirrors `model.kronos.KronosPredictor`'s
`predict()` API but computes the same indicator feature set at inference
time and is generalized to any `feature_list`/`d_in`. It's kept separate
from the shared `KronosPredictor` so `finetune_csv`/`finetune` inference is
never affected by indicator-related changes.

CLI (loads finetuned tokenizer + basemodel from the training config, writes
results under `predictions/<data_file_stem>/`):

```bash
python predict.py --config configs/config_multi_symbol_5m_ta.yaml \
  --input data/ADAUSDT_kline_5min.csv
```

Or programmatically:

```python
import torch
from model import Kronos, KronosTokenizer
from predictor import KronosPredictorTA
from config_loader import CustomFinetuneConfig

config = CustomFinetuneConfig('configs/config_ali09988_candle-5min_ta.yaml')
tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
model = Kronos.from_pretrained(config.basemodel_best_model_path)

predictor = KronosPredictorTA(
    model, tokenizer,
    feature_list=config.feature_list,
    enabled_indicators=config.enabled_indicators,
)
pred_df = predictor.predict(df, x_timestamp, y_timestamp, pred_len=48)
```

`pred_df` contains predictions for all 18 feature columns (not just
OHLCV) -- the model was trained to jointly forecast raw prices and their
derived indicators.

## Walk-forward backtest

`backtest.py` runs forecast windows on each file's validation split
(last `val_ratio` of bars), scores horizon-end close error / direction
accuracy, and simulates a simple long-if-pred-up strategy. Per-experiment
wrappers pin the two multi-symbol runs (default `--pred-len 48` with
`--stride 288` for tractable runtime; training used 288):

```bash
python backtest_multi_symbol_5m_ta.py
python backtest_multi_symbol_5m_ta_norm.py

# quick smoke (2 windows on one symbol)
python backtest_multi_symbol_5m_ta.py \
  --input data/BTCUSDT_kline_5min.csv --max-windows 2

# full training horizon (slow: ~30m/window on CPU/MPS)
python backtest_multi_symbol_5m_ta.py --pred-len 288 --stride 288 \
  --input data/BTCUSDT_kline_5min.csv --max-windows 1
```

Results land under `backtests/<exp_name>/` (`summary.json`, per-symbol
`*_windows.csv`, charts).
