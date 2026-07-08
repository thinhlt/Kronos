import os
import sys
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from indicators import BASE_FEATURES, ensure_features
from model_factory import load_finetuned_tokenizer


class CustomKlineDataset(Dataset):
    """Loads one or more OHLCV(+indicator) CSVs as a single dataset.

    Each file is loaded, indicator-augmented, and time-split independently --
    a training window is never built from more than one file, so a sample is
    never partly one symbol and partly another. See
    docs/adr/0002-multi-file-windows-never-cross-files.md.
    """

    def __init__(self, data_paths, data_type='train', lookback_window=90, predict_window=10,
                 clip=5.0, seed=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 feature_list=None, enabled_indicators=None):
        self.data_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        if not self.data_paths:
            raise ValueError("CustomKlineDataset requires at least one data path")
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.feature_list = list(feature_list) if feature_list else list(BASE_FEATURES)
        self.enabled_indicators = enabled_indicators or {}
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        self.py_rng = random.Random(seed)

        # Per-file post-split data/timestamps and each file's own (non-overlapping)
        # window count -- the global sample index below is just the concatenation
        # of these per-file counts.
        self.file_data = []
        self.file_timestamps = []
        self.file_n_samples = []
        for data_path in self.data_paths:
            data, timestamps = self._load_and_preprocess_one(data_path)
            data, timestamps = self._split_one_by_time(data, timestamps)
            n_samples = max(0, len(data) - self.window + 1)
            self.file_data.append(data)
            self.file_timestamps.append(timestamps)
            self.file_n_samples.append(n_samples)

        self.cumulative_samples = np.cumsum([0] + self.file_n_samples)
        self.n_samples = int(self.cumulative_samples[-1])

        total_length = sum(len(d) for d in self.file_data)
        print(f"[{data_type.upper()}] {len(self.data_paths)} file(s), total data length: "
              f"{total_length}, available samples: {self.n_samples}")
        if self.n_samples == 0:
            raise ValueError(
                f"[{data_type.upper()}] No usable samples across {len(self.data_paths)} file(s) -- "
                f"lookback_window+predict_window ({self.window}) may be larger than the "
                f"{data_type} split of at least one file."
            )

    def _load_and_preprocess_one(self, data_path):
        df = pd.read_csv(data_path)

        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        # Forward-fill stray gaps in the base OHLCV+amount columns before deriving
        # indicators from them, so a single missing tick doesn't NaN-out an entire
        # indicator warm-up window downstream.
        if df[BASE_FEATURES].isnull().any().any():
            print(f"Warning: Missing values found in base OHLCV data ({data_path}), performing forward fill")
            df[BASE_FEATURES] = df[BASE_FEATURES].fillna(method='ffill')

        df = ensure_features(df, self.feature_list, self.enabled_indicators)

        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month

        data = df[self.feature_list + self.time_feature_list].copy()
        timestamps = df['timestamps'].copy()

        # Indicators like MACD/Bollinger/KDJ need a warm-up period and produce
        # leading NaNs; drop those rows rather than fabricate warm-up values.
        before = len(data)
        valid_mask = data[self.feature_list].notna().all(axis=1)
        data = data.loc[valid_mask].reset_index(drop=True)
        timestamps = timestamps.loc[valid_mask].reset_index(drop=True)
        dropped = before - len(data)
        if dropped > 0:
            print(f"Dropped {dropped} leading rows with NaN technical-indicator warm-up values ({data_path})")

        print(f"[{os.path.basename(data_path)}] time range: {timestamps.min()} to {timestamps.max()}, "
              f"usable length: {len(data)} records")

        return data, timestamps

    def _split_one_by_time(self, data, timestamps):
        total_length = len(data)

        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))

        if self.data_type == 'train':
            data = data.iloc[:train_end]
            timestamps = timestamps.iloc[:train_end]
        elif self.data_type == 'val':
            data = data.iloc[train_end:val_end]
            timestamps = timestamps.iloc[train_end:val_end]
        elif self.data_type == 'test':
            data = data.iloc[val_end:]
            timestamps = timestamps.iloc[val_end:]

        return data.reset_index(drop=True), timestamps.reset_index(drop=True)

    def set_epoch_seed(self, epoch):
        epoch_seed = self.seed + epoch
        self.py_rng.seed(epoch_seed)
        self.current_epoch = epoch

    def __len__(self):
        return self.n_samples

    def _locate(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_samples, idx, side='right') - 1)
        local_idx = idx - int(self.cumulative_samples[file_idx])
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        data = self.file_data[file_idx]
        max_start = len(data) - self.window
        if max_start <= 0:
            raise ValueError(f"Data length insufficient to create samples in file: {self.data_paths[file_idx]}")

        if self.data_type == 'train':
            epoch = getattr(self, 'current_epoch', 0)
            start_idx = (local_idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)
        else:
            start_idx = local_idx % (max_start + 1)

        end_idx = start_idx + self.window

        window_data = data.iloc[start_idx:end_idx]

        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_file = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)

    logger.info(f"=== Basemodel Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")

    train_dataset = CustomKlineDataset(
        data_paths=config.data_paths,
        data_type='train',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        feature_list=config.feature_list,
        enabled_indicators=config.enabled_indicators
    )

    val_dataset = CustomKlineDataset(
        data_paths=config.data_paths,
        data_type='val',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        feature_list=config.feature_list,
        enabled_indicators=config.enabled_indicators
    )

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler
    )

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def train_model(model, tokenizer, device, config, save_dir, logger):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1

    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.basemodel_epochs,
        pct_start=0.03,
        div_factor=10
    )

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float('inf')
    batch_idx_global = 0

    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        train_batches = 0

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            train_batches += 1

            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {loss.item():.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)

            batch_idx_global += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                val_loss += loss.item()
                val_batches += 1

        if use_ddp:
            tensor_sum = torch.tensor([epoch_train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_batches_all = int(tensor_sum[1].item())
            val_loss_all = tensor_sum[2].item()
            val_batches_all = int(tensor_sum[3].item())
            avg_train_loss = (epoch_train_loss_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_batches_all) if val_batches_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
                       f"Training Loss: {avg_train_loss:.4f}\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Epoch Time: {epoch_time:.2f} seconds\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)

    return best_val_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Kronos Basemodel Fine-tuning Training (technical-indicator features)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from config_loader import CustomFinetuneConfig
    from model_factory import build_predictor
    config = CustomFinetuneConfig(args.config)

    os.makedirs(config.basemodel_save_path, exist_ok=True)

    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    logger.info("Loading finetuned tokenizer and predictor...")
    print("Loading finetuned tokenizer and predictor...")
    tokenizer = load_finetuned_tokenizer(config)
    model = build_predictor(config)

    tokenizer = tokenizer.to(device)
    model = model.to(device)

    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_size:,}")
    print(f"Model parameters: {model_size:,}")

    logger.info("=== Training Configuration ===")
    logger.info(f"Data files ({len(config.data_paths)}): {config.data_paths}")
    logger.info(f"Feature list ({config.d_in} dims): {config.feature_list}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.predictor_learning_rate}")
    logger.info(f"Training epochs: {config.basemodel_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Tokenizer path: {config.finetuned_tokenizer_path}")
    logger.info(f"Pretrained model path: {config.pretrained_predictor_path}")

    logger.info("Starting fine-tuning training...")
    print("Starting fine-tuning training...")
    best_val_loss = train_model(model, tokenizer, device, config, config.basemodel_save_path, logger)

    final_msg = f"Training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.basemodel_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    import time
    main()
else:
    import time
