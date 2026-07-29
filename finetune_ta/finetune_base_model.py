import os
import sys
import time
import random
import numpy as np
import torch
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model_factory import load_finetuned_tokenizer
from checkpoint_utils import find_checkpoint, load_checkpoint, save_checkpoint
from dataloaders import create_dataloaders
from feature_normalize import normalize_features_torch
from logic_loss import (
    build_bit_tables,
    freeze_tokenizer,
    logic_loss_from_logits,
)


def _make_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )


def _make_scheduler(optimizer, config, steps_per_epoch, epochs):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        div_factor=10,
    )


def _make_grad_scaler(use_amp, amp_dtype):
    return torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))


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


def train_model(model, tokenizer, device, config, save_dir, logger):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(
        config, batch_size=config.basemodel_batch_size
    )

    # Tokenizer stays frozen; soft-decode still propagates grads into logits.
    freeze_tokenizer(tokenizer)

    logic_enabled = bool(getattr(config, 'logic_loss_enabled', False))
    logic_weight = float(getattr(config, 'logic_loss_weight', 0.1))
    logic_ohlc_weight = float(getattr(config, 'logic_loss_ohlc_weight', 1.0))
    logic_ha_weight = float(getattr(config, 'logic_loss_ha_weight', 1.0))
    logic_max_timesteps = int(getattr(config, 'logic_loss_max_timesteps', 128))
    logic_use_checkpoint = bool(getattr(config, 'logic_loss_use_checkpoint', True))

    s1_bit_table = s2_bit_table = None
    if logic_enabled:
        tok = tokenizer.module if hasattr(tokenizer, 'module') else tokenizer
        s1_bit_table, s2_bit_table = build_bit_tables(
            tok.s1_bits, tok.s2_bits, tok.codebook_dim, device=device
        )
        msg = (
            f"Logic loss configured: enabled={logic_enabled}, "
            f"weight={logic_weight}, ohlc_weight={logic_ohlc_weight}, ha_weight={logic_ha_weight}, "
            f"max_timesteps={logic_max_timesteps}, use_checkpoint={logic_use_checkpoint}"
        )
        logger.info(msg)
        if rank == 0:
            print(msg)

    accumulation_steps = max(1, int(getattr(config, 'accumulation_steps', 1)))
    # OneCycle steps once per optimizer update (after accumulation), not per micro-batch.
    steps_per_epoch = max(1, len(train_loader) // accumulation_steps)
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config, steps_per_epoch, config.basemodel_epochs)

    # Mixed precision: only meaningful on CUDA. fp16 needs a GradScaler to
    # avoid gradient underflow; bf16 has fp32's exponent range so the scaler
    # is created disabled (a no-op) in that case.
    use_amp = bool(getattr(config, 'use_amp', False)) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if getattr(config, 'amp_dtype', 'fp16') == 'bf16' else torch.float16
    scaler = _make_grad_scaler(use_amp, amp_dtype)
    if rank == 0:
        print(f"Mixed precision (AMP): {use_amp} (dtype: {amp_dtype if use_amp else 'n/a'})")
        print(
            f"Basemodel batch={config.basemodel_batch_size}, accumulation={accumulation_steps}, "
            f"effective_batch≈{config.basemodel_batch_size * accumulation_steps}"
        )

    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    if getattr(config, 'resume', True):
        checkpoint_path = find_checkpoint(save_dir)
        if checkpoint_path:
            start_epoch, best_val_loss, global_step = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device=device, scaler=scaler
            )
            if global_step is None:
                # Older checkpoint predating global_step tracking -- approximate
                # from the epoch boundary instead of losing resume state entirely.
                global_step = start_epoch * len(train_loader)
            msg = (
                f"Resuming basemodel training from {checkpoint_path}: starting at epoch "
                f"{start_epoch + 1}/{config.basemodel_epochs}, global_step {global_step}, "
                f"best_val_loss so far {best_val_loss:.4f}"
            )
            logger.info(msg)
            if rank == 0:
                print(msg)

    if start_epoch >= config.basemodel_epochs:
        msg = (
            f"Checkpoint already completed all {config.basemodel_epochs} "
            f"configured epochs, nothing to resume"
        )
        logger.info(msg)
        if rank == 0:
            print(msg)
        return best_val_loss

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    batch_idx_global = global_step
    checkpoint_every_n_steps = getattr(config, 'checkpoint_every_n_steps', 0)

    # A mid-epoch checkpoint (see checkpoint_every_n_steps below) records the
    # *previous* completed epoch, so start_epoch here lands back on the
    # interrupted epoch itself. Its dataloader always starts a fresh epoch at
    # batch 0, but global_step already reflects the batches that epoch had
    # finished before the interruption -- replaying them would both waste
    # compute and desync the (already-restored) optimizer/scheduler state
    # from the data position. Skip exactly that many batches once, on the
    # first resumed epoch only.
    _epoch_boundary_steps = start_epoch * len(train_loader)
    skip_batches_in_resumed_epoch = max(0, batch_idx_global % len(train_loader))
    # #region agent log
    try:
        import json, time as _time
        _payload = {
            "sessionId": "1a47a0",
            "runId": "pre-fix",
            "hypothesisId": "H1_H4_H5",
            "location": "finetune_base_model.py:skip_batches",
            "message": "resume skip-batch computation",
            "data": {
                "save_dir": str(save_dir),
                "start_epoch": start_epoch,
                "start_epoch_1indexed": start_epoch + 1,
                "global_step": batch_idx_global,
                "len_train_loader": len(train_loader),
                "epoch_boundary_steps": _epoch_boundary_steps,
                "raw_skip": batch_idx_global - _epoch_boundary_steps,
                "skip_batches_in_resumed_epoch": skip_batches_in_resumed_epoch,
                "basemodel_batch_size": getattr(config, "basemodel_batch_size", None),
                "accumulation_steps": accumulation_steps,
                "implied_steps_per_epoch_from_global": (
                    (batch_idx_global / start_epoch) if start_epoch > 0 else None
                ),
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
        logger.info(f"[DEBUG_RESUME] {json.dumps(_payload)}")
        if rank == 0:
            print(f"[DEBUG_RESUME] {json.dumps(_payload)}")
    except Exception:
        pass
    # #endregion
    if skip_batches_in_resumed_epoch > 0:
        msg = (
            f"Mid-epoch resume: skipping the first {skip_batches_in_resumed_epoch}/"
            f"{len(train_loader)} batches of epoch {start_epoch + 1} (already completed "
            f"before the interruption)"
        )
        logger.info(msg)
        if rank == 0:
            print(msg)

    for epoch in range(start_epoch, config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        skip_batches = skip_batches_in_resumed_epoch if epoch == start_epoch else 0
        raw_model = model.module if use_ddp else model

        epoch_train_loss = 0.0
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            if batch_idx < skip_batches:
                if checkpoint_every_n_steps > 0 and batch_idx % checkpoint_every_n_steps == 0:
                    print(
                        f"Skipping batch {batch_idx} of epoch {epoch} "
                        f"(already completed before the interruption)"
                    )
                continue

            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            batch_x = normalize_features_torch(
                batch_x, config.feature_list, clip=config.clip
            )

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            logic_loss_val = None
            is_accum_boundary = ((train_batches + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader))

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                ce_loss, s1_loss, s2_loss = raw_model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                loss = ce_loss
                if logic_enabled:
                    logic_loss_val = logic_loss_from_logits(
                        tokenizer,
                        logits[0],
                        logits[1],
                        s1_bit_table,
                        s2_bit_table,
                        config.feature_list,
                        ohlc_weight=logic_ohlc_weight,
                        ha_weight=logic_ha_weight,
                        max_timesteps=logic_max_timesteps,
                        use_checkpoint=logic_use_checkpoint,
                    )
                    loss = ce_loss + logic_weight * logic_loss_val
            scaler.scale(loss / accumulation_steps).backward()

            if is_accum_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_train_loss += float(loss.item() if torch.is_tensor(loss) else loss)
            train_batches += 1

            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                loss_item = float(loss.item() if torch.is_tensor(loss) else loss)
                log_msg = (
                    f"[Epoch {epoch+1}/{config.basemodel_epochs}, "
                    f"Step {batch_idx+1}/{len(train_loader)}] "
                    f"LR: {lr:.6f}, Loss: {loss_item:.4f}"
                )
                if logic_loss_val is not None:
                    log_msg += f", CE: {ce_loss.item():.4f}, Logic: {logic_loss_val.item():.4f}"
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)

            batch_idx_global += 1

            if checkpoint_every_n_steps > 0 and rank == 0 and batch_idx_global % checkpoint_every_n_steps == 0:
                loss_item = float(loss.item() if torch.is_tensor(loss) else loss)
                lr = optimizer.param_groups[0]['lr']
                if hasattr(logger, 'log_metric'):
                    logger.log_metric('train_predictor_loss_batch', loss_item, step=batch_idx_global)
                    logger.log_metric('train_S1_loss_each_batch', s1_loss.detach(), step=batch_idx_global)
                    logger.log_metric('train_S2_loss_each_batch', s2_loss.detach(), step=batch_idx_global)
                    logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)
                    if logic_loss_val is not None:
                        logger.log_metric('train_logic_loss_batch', logic_loss_val.item(), step=batch_idx_global)
                # Mid-epoch safety net: record the *previous* completed epoch as the
                # resume point (not the in-progress one), so resuming just re-runs
                # the interrupted epoch from its start rather than attempting exact
                # mid-epoch resumption.
                save_checkpoint(
                    save_dir, model, optimizer, scheduler, epoch - 1, best_val_loss,
                    global_step=batch_idx_global,
                    extra={'mid_epoch_step': batch_idx_global}, scaler=scaler,
                )

        model.eval()
        val_loss = 0.0
        val_batches = 0

        eval_start_msg = (
            f"[Epoch {epoch+1}/{config.basemodel_epochs}] Starting evaluation on "
            f"{len(val_loader)} validation batches..."
        )
        logger.info(eval_start_msg)
        if rank == 0:
            print(eval_start_msg)
        eval_start_time = time.time()

        with torch.no_grad():
            for val_batch_idx, (batch_x, batch_x_stamp) in enumerate(val_loader):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                batch_x = normalize_features_torch(
                    batch_x, config.feature_list, clip=config.clip
                )

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = raw_model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    loss, _, _ = raw_model.head.compute_loss(
                        logits[0], logits[1], token_out[0], token_out[1]
                    )

                val_loss += loss.item()
                val_batches += 1

                if checkpoint_every_n_steps > 0 and (
                    (val_batch_idx + 1) % checkpoint_every_n_steps == 0
                    or (val_batch_idx + 1) == len(val_loader)
                ):
                    running_avg_val_loss = val_loss / val_batches
                    eval_log_msg = (
                        f"[Epoch {epoch+1}/{config.basemodel_epochs}, "
                        f"Eval Step {val_batch_idx+1}/{len(val_loader)}] "
                        f"Loss: {loss.item():.4f}, Running Avg Loss: {running_avg_val_loss:.4f}"
                    )
                    logger.info(eval_log_msg)
                    if rank == 0:
                        print(eval_log_msg)
                elif checkpoint_every_n_steps <= 0 and (val_batch_idx + 1) == len(val_loader):
                    running_avg_val_loss = val_loss / val_batches
                    eval_log_msg = (
                        f"[Epoch {epoch+1}/{config.basemodel_epochs}, "
                        f"Eval Step {val_batch_idx+1}/{len(val_loader)}] "
                        f"Loss: {loss.item():.4f}, Running Avg Loss: {running_avg_val_loss:.4f}"
                    )
                    logger.info(eval_log_msg)
                    if rank == 0:
                        print(eval_log_msg)

        eval_time = time.time() - eval_start_time
        eval_done_msg = (
            f"[Epoch {epoch+1}/{config.basemodel_epochs}] "
            f"Evaluation finished in {eval_time:.2f} seconds"
        )
        logger.info(eval_done_msg)
        if rank == 0:
            print(eval_done_msg)

        if use_ddp:
            tensor_sum = torch.tensor(
                [epoch_train_loss, train_batches, val_loss, val_batches],
                dtype=torch.float64,
                device=device,
            )
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
        epoch_summary = (
            f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
            f"Training Loss: {avg_train_loss:.4f}\n"
            f"Validation Loss: {avg_val_loss:.4f}\n"
            f"Epoch Time: {epoch_time:.2f} seconds\n"
        )
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = (
                    f"Best model saved to: {model_save_path} "
                    f"(validation loss: {best_val_loss:.4f})"
                )
                logger.info(save_msg)
                print(save_msg)

        if rank == 0:
            checkpoint_path = save_checkpoint(
                save_dir, model, optimizer, scheduler, epoch, best_val_loss,
                global_step=batch_idx_global, scaler=scaler,
            )
            resume_msg = (
                f"Checkpoint saved to: {checkpoint_path} "
                f"(resume point: epoch {epoch + 1}/{config.basemodel_epochs})"
            )
            logger.info(resume_msg)
            print(resume_msg)

    return best_val_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Kronos Basemodel Fine-tuning Training (technical-indicator features)'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Configuration file path (default: config.yaml)',
    )
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
    compiled_model = torch.compile(model)

    model_size = sum(p.numel() for p in compiled_model.parameters())
    logger.info(f"Model parameters: {model_size:,}")
    print(f"Model parameters: {model_size:,}")

    logger.info("=== Training Configuration ===")
    logger.info(f"Data files ({len(config.data_paths)}): {config.data_paths}")
    logger.info(f"Feature list ({config.d_in} dims): {config.feature_list}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.basemodel_batch_size}")
    logger.info(f"Learning rate: {config.predictor_learning_rate}")
    logger.info(f"Training epochs: {config.basemodel_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision (AMP): {config.use_amp} (dtype: {config.amp_dtype})")
    logger.info(
        f"Logic loss: enabled={config.logic_loss_enabled}, "
        f"weight={config.logic_loss_weight}"
    )
    logger.info(f"Tokenizer path: {config.finetuned_tokenizer_path}")
    logger.info(f"Pretrained model path: {config.pretrained_predictor_path}")

    logger.info("Starting fine-tuning training...")
    print("Starting fine-tuning training...")
    best_val_loss = train_model(
        compiled_model, tokenizer, device, config, config.basemodel_save_path, logger
    )

    final_msg = (
        f"Training completed! Best validation loss: {best_val_loss:.4f}\n"
        f"Model saved to: {config.basemodel_save_path}"
    )
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
