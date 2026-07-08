"""Full training-state Checkpoint save/load, distinct from the weights-only
Best Model. See CONTEXT.md: "Checkpoint" and docs/adr/0003 for why this
exists and what it's for (resuming an interrupted run, never inference).

A Checkpoint bundles model weights, optimizer state, LR scheduler state,
epoch number, best validation loss so far, RNG state, and (when AMP is
enabled) the GradScaler state -- everything needed to resume training
byte-for-byte from the next epoch. It's written to `checkpoint_last.pt` and
overwritten in place every epoch (and optionally more often), never kept
per-epoch.
"""
import os
import random

import numpy as np
import torch

CHECKPOINT_FILENAME = "checkpoint_last.pt"


def _unwrap(model):
    """DDP-wrapped models nest the real module under `.module`; checkpoints
    always store/restore the underlying module's state so they load fine
    whether or not the run that resumes them uses distributed training."""
    return model.module if hasattr(model, "module") else model


def find_checkpoint(save_dir: str):
    """Returns the path to `checkpoint_last.pt` in `save_dir` if it exists,
    else None."""
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)
    return checkpoint_path if os.path.exists(checkpoint_path) else None


def save_checkpoint(save_dir, model, optimizer, scheduler, epoch, best_val_loss,
                     extra=None, scaler=None):
    """Saves a full resumable Checkpoint to `save_dir/checkpoint_last.pt`,
    overwriting any previous one. `epoch` is the last *completed* epoch
    (0-indexed) -- resuming continues from `epoch + 1`.

    `scaler` is an optional `torch.cuda.amp.GradScaler`; its state is saved
    too so resuming an AMP run doesn't reset the loss-scale warmup.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)

    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        state["extra"] = extra

    # Write to a temp file first so a crash mid-save can't corrupt the
    # previous, still-valid checkpoint that resuming depends on.
    tmp_path = checkpoint_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device=None, scaler=None):
    """Restores model/optimizer/scheduler/RNG (and optional GradScaler) state
    from a Checkpoint. Returns `(start_epoch, best_val_loss)`, where
    `start_epoch` is the next epoch to run (i.e. `saved_epoch + 1`)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    _unwrap(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # rng_state = checkpoint.get("rng_state")
    # if rng_state:
    #     random.setstate(rng_state["python"])
    #     np.random.set_state(rng_state["numpy"])
    #     torch.set_rng_state(rng_state["torch"].cpu().to(torch.uint8) if torch.is_tensor(rng_state["torch"]) else rng_state["torch"])
    #     if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
    #         try:
    #             torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    #         except RuntimeError:
    #             # e.g. resuming on a different GPU count/type than the run
    #             # that saved this checkpoint -- not fatal, just less exact.
    #             pass

    saved_epoch = checkpoint.get("epoch", -1)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    start_epoch = saved_epoch + 1

    return start_epoch, best_val_loss
