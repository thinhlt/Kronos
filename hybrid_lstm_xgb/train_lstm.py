"""Stage-1 trainer: supervised LSTM on Horizon Return, then keep h_n as embedding."""
from __future__ import annotations

import json
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from feature_normalize import normalize_features_torch
from lstm_encoder import LSTMEncoder
from windows import (
    FileWindows,
    build_all_files,
    collect_origins,
    horizon_log_return,
    slice_history,
    summarize_origin_counts,
)


class OriginDataset(Dataset):
    """Lazy history slices + Horizon Return targets for a list of (file, origin)."""

    def __init__(
        self,
        files: Sequence[FileWindows],
        pairs: Sequence[Tuple[int, int]],
        lookback: int,
        horizon: int,
    ):
        self.files = files
        self.pairs = list(pairs)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        fi, origin = self.pairs[idx]
        fw = self.files[fi]
        hist, _ = slice_history(fw, origin, self.lookback)
        y = horizon_log_return(fw.closes, origin, self.horizon)
        return torch.from_numpy(hist), torch.tensor(y, dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_std(files: Sequence[FileWindows], pairs: Sequence[Tuple[int, int]], horizon: int) -> float:
    ys = [horizon_log_return(files[fi].closes, origin, horizon) for fi, origin in pairs]
    ys = np.asarray(ys, dtype=np.float64)
    ys = ys[np.isfinite(ys)]
    if len(ys) == 0:
        raise ValueError("No finite targets to compute target_std")
    std = float(ys.std(ddof=0))
    return max(std, 1e-8)


def _run_epoch(
    model: LSTMEncoder,
    loader: DataLoader,
    feature_list: list,
    clip: float,
    target_std: float,
    device: torch.device,
    optimizer=None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    for hist, y in loader:
        hist = hist.to(device)
        y = y.to(device)
        x_norm = normalize_features_torch(hist, feature_list, clip=clip)
        y_std = y / target_std
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x_norm)
        loss = torch.mean((pred - y_std) ** 2)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * hist.size(0)
        n += hist.size(0)
    return total_loss / max(n, 1)


def train_lstm(config, files: List[FileWindows] | None = None) -> dict:
    """Fit stage-1 LSTM on s1_fit, early-stop on s1_dev, write lstm_best.pt + feature_spec seed."""
    set_seed(config.seed)
    device = config.device()
    os.makedirs(config.save_dir, exist_ok=True)
    paths = config.artifact_paths()

    if files is None:
        files = build_all_files(config)

    counts = summarize_origin_counts(
        files, config.lookback_window, config.horizon, config.train_stride
    )
    print(f"Origin counts (train_stride={config.train_stride}): {counts}")
    if counts["s1_fit"] == 0 or counts["s1_dev"] == 0:
        raise ValueError(
            f"Stage-1 regions empty after carving: s1_fit={counts['s1_fit']} "
            f"s1_dev={counts['s1_dev']}. Check lookback/horizon/purge/dev_fraction."
        )

    fit_pairs = collect_origins(
        files, "s1_fit", config.lookback_window, config.horizon, config.train_stride
    )
    dev_pairs = collect_origins(
        files, "s1_dev", config.lookback_window, config.horizon, config.train_stride
    )
    target_std = _target_std(files, fit_pairs, config.horizon)
    print(f"target_std (s1_fit) = {target_std:.6g}")

    train_ds = OriginDataset(files, fit_pairs, config.lookback_window, config.horizon)
    dev_ds = OriginDataset(files, dev_pairs, config.lookback_window, config.horizon)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.lstm_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config.lstm_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = LSTMEncoder(
        d_in=config.d_in,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm_learning_rate,
        weight_decay=config.lstm_weight_decay,
    )

    best_dev = float("inf")
    best_epoch = -1
    patience_left = config.lstm_early_stopping_patience
    history = []

    for epoch in range(1, config.lstm_epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, config.feature_list, config.clip, target_std, device, optimizer
        )
        with torch.no_grad():
            dev_loss = _run_epoch(
                model, dev_loader, config.feature_list, config.clip, target_std, device, None
            )
        history.append({"epoch": epoch, "train_mse": train_loss, "dev_mse": dev_loss})
        print(f"[LSTM] epoch {epoch}/{config.lstm_epochs} train_mse={train_loss:.6g} "
              f"dev_mse={dev_loss:.6g}")

        if dev_loss < best_dev:
            best_dev = dev_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "d_in": config.d_in,
                    "hidden_size": config.lstm_hidden_size,
                    "num_layers": config.lstm_num_layers,
                    "dropout": config.lstm_dropout,
                    "target_std": target_std,
                    "feature_list": list(config.feature_list),
                    "lookback_window": config.lookback_window,
                    "horizon": config.horizon,
                    "clip": config.clip,
                    "epoch": epoch,
                    "dev_mse": best_dev,
                },
                paths["lstm_best"],
            )
            print(f"  -> saved {paths['lstm_best']}")

    return {
        "lstm_best": paths["lstm_best"],
        "target_std": target_std,
        "best_epoch": best_epoch,
        "best_dev_mse": best_dev,
        "history": history,
        "origin_counts": counts,
    }


def load_lstm_encoder(lstm_path: str, device: torch.device) -> Tuple[LSTMEncoder, dict]:
    ckpt = torch.load(lstm_path, map_location=device, weights_only=False)
    model = LSTMEncoder(
        d_in=int(ckpt["d_in"]),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt.get("num_layers", 1)),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


if __name__ == "__main__":
    import argparse

    from config_loader import HybridConfig

    parser = argparse.ArgumentParser(description="Train stage-1 LSTM encoder")
    parser.add_argument("--config", type=str, default="configs/config_hybrid_5m_h12.yaml")
    args = parser.parse_args()
    cfg = HybridConfig(args.config)
    result = train_lstm(cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
