"""Shared dataset + DataLoader construction for tokenizer and basemodel training."""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from indicators import BASE_FEATURES, ensure_features


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

        # Per-file post-split feature/stamp arrays (float32 numpy) and each
        # file's own (non-overlapping) window count -- the global sample index
        # below is just the concatenation of these per-file counts.
        # Normalization is deferred to GPU (see normalize_features_torch in the
        # training loops) so __getitem__ stays a cheap slice + tensor wrap.
        self.file_data = []
        self.file_stamps = []
        self.file_n_samples = []
        for data_path in self.data_paths:
            data, timestamps = self._load_and_preprocess_one(data_path)
            data, timestamps = self._split_one_by_time(data, timestamps)
            features = data[self.feature_list].to_numpy(dtype=np.float32, copy=True)
            stamps = data[self.time_feature_list].to_numpy(dtype=np.float32, copy=True)
            n_samples = max(0, len(features) - self.window + 1)
            self.file_data.append(features)
            self.file_stamps.append(stamps)
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
        stamps = self.file_stamps[file_idx]
        max_start = len(data) - self.window
        if max_start <= 0:
            raise ValueError(f"Data length insufficient to create samples in file: {self.data_paths[file_idx]}")

        if self.data_type == 'train':
            epoch = getattr(self, 'current_epoch', 0)
            start_idx = (local_idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)
        else:
            start_idx = local_idx % (max_start + 1)

        end_idx = start_idx + self.window

        # Copy so torch tensors own their storage (safe with DataLoader workers).
        x_tensor = torch.from_numpy(data[start_idx:end_idx].copy())
        x_stamp_tensor = torch.from_numpy(stamps[start_idx:end_idx].copy())

        return x_tensor, x_stamp_tensor


def create_dataloaders(config, batch_size=None):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")

    batch_size = config.batch_size if batch_size is None else batch_size

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
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
        prefetch_factor=4,
    )

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler
