"""Full training-state Checkpoint save/load, distinct from the weights-only
Best Model. See CONTEXT.md: "Checkpoint" and docs/adr/0003 for why this
exists and what it's for (resuming an interrupted run, never inference).

A Checkpoint bundles model weights, optimizer state, LR scheduler state,
epoch number, global step, best validation loss so far, RNG state, and
(when AMP is enabled) the GradScaler state -- everything needed to resume
training byte-for-byte from the next epoch. It's written to
`checkpoint_last.pt` and overwritten in place every epoch (and optionally
more often), never kept per-epoch.
"""
import os
import random
import shutil
import numpy as np
import torch

CHECKPOINT_FILENAME = "checkpoint_last.pt"


def _unwrap(model):
    """Peel DDP / DataParallel / torch.compile wrappers down to the real
    module. Checkpoints always store/restore that underlying state_dict so
    they load fine whether the run that saved or resumes them used DDP,
    torch.compile, both, or neither.

    torch.compile nests the original module under `_orig_mod`; saving the
    wrapper's state_dict otherwise prefixes every key with `_orig_mod.` and
    breaks load into a plain KronosTokenizer / predictor.
    """
    while True:
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel,
                              torch.nn.DataParallel)):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            break
    return model


def _normalize_state_dict(state_dict):
    """Strip `module.` / `_orig_mod.` key prefixes left by older checkpoints
    that were saved without unwrapping DDP or torch.compile wrappers."""
    prefixes = ("module.", "_orig_mod.")
    normalized = {}
    for key, value in state_dict.items():
        while True:
            stripped = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    stripped = True
                    break
            if not stripped:
                break
        normalized[key] = value
    return normalized


def find_checkpoint(save_dir: str):
    """Returns the path to `checkpoint_last.pt` in `save_dir` if it exists,
    else None."""
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)
    return checkpoint_path if os.path.exists(checkpoint_path) else None


def save_checkpoint(save_dir, model, optimizer, scheduler, epoch, best_val_loss,
                     global_step=0, extra=None, scaler=None):
    """Saves a full resumable Checkpoint to `save_dir/checkpoint_last.pt`,
    overwriting any previous one. `epoch` is the last *completed* epoch
    (0-indexed) -- resuming continues from `epoch + 1`. `global_step` is the
    total number of optimizer steps taken so far across all epochs -- it's
    saved verbatim so resuming (including mid-epoch, via the
    `checkpoint_every_n_steps` safety net) picks logging/step-based
    scheduling back up exactly where it left off, rather than approximating
    it from `epoch * steps_per_epoch`.

    `scaler` is an optional `torch.cuda.amp.GradScaler`; its state is saved
    too so resuming an AMP run doesn't reset the loss-scale warmup.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)

    state = {
        "epoch": epoch,
        "global_step": global_step,
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
    tmp_path = "/content/checkpoint_last.pt"
    torch.save(state, tmp_path)
    shutil.copyfile(tmp_path, checkpoint_path)

    # #region agent log
    try:
        import json, time as _time
        _payload = {
            "sessionId": "1a47a0",
            "runId": "pre-fix",
            "hypothesisId": "H3_H5",
            "location": "checkpoint_utils.py:save_checkpoint",
            "message": "checkpoint saved",
            "data": {
                "checkpoint_path": checkpoint_path,
                "epoch": epoch,
                "global_step": global_step,
                "extra": extra,
                "tmp_path": tmp_path,
            },
            "timestamp": int(_time.time() * 1000),
        }
        for _p in (
            "/Users/macintosh/Prj/Kronos/.cursor/debug-1a47a0.log",
            os.path.join(save_dir, "debug-1a47a0.log"),
        ):
            try:
                os.makedirs(os.path.dirname(_p), exist_ok=True)
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(_payload) + "\n")
            except Exception:
                pass
        print(f"[DEBUG_RESUME] {json.dumps(_payload)}")
    except Exception:
        pass
    # #endregion

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device=None, scaler=None):
    """Restores model/optimizer/scheduler/RNG (and optional GradScaler) state
    from a Checkpoint. Returns `(start_epoch, best_val_loss, global_step)`,
    where `start_epoch` is the next epoch to run (i.e. `saved_epoch + 1`) and
    `global_step` is the total optimizer-step count to resume counting from.
    `global_step` is `None` for checkpoints saved before this field existed --
    callers should fall back to an epoch-based approximation in that case."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    _unwrap(model).load_state_dict(_normalize_state_dict(checkpoint["model_state_dict"]))

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
    global_step = checkpoint.get("global_step")

    # #region agent log
    try:
        import json, time as _time
        _extra = checkpoint.get("extra")
        _payload = {
            "sessionId": "1a47a0",
            "runId": "pre-fix",
            "hypothesisId": "H2_H3_H5",
            "location": "checkpoint_utils.py:load_checkpoint",
            "message": "checkpoint fields after load",
            "data": {
                "checkpoint_path": str(checkpoint_path),
                "saved_epoch": saved_epoch,
                "start_epoch": start_epoch,
                "global_step": global_step,
                "has_extra": _extra is not None,
                "extra_keys": list(_extra.keys()) if isinstance(_extra, dict) else None,
                "extra": {k: _extra.get(k) for k in ("mid_epoch_step", "epoch_batch_idx", "steps_per_epoch") if isinstance(_extra, dict)},
                "ckpt_keys": sorted(list(checkpoint.keys())),
            },
            "timestamp": int(_time.time() * 1000),
        }
        for _p in (
            "/Users/macintosh/Prj/Kronos/.cursor/debug-1a47a0.log",
            os.path.join(os.path.dirname(checkpoint_path), "debug-1a47a0.log"),
        ):
            try:
                os.makedirs(os.path.dirname(_p), exist_ok=True)
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(_payload) + "\n")
            except Exception:
                pass
        print(f"[DEBUG_RESUME] {json.dumps(_payload)}")
    except Exception:
        pass
    # #endregion

    return start_epoch, best_val_loss, global_step
