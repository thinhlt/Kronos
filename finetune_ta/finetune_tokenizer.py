import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import datetime
import logging
from logging.handlers import RotatingFileHandler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../")
from finetune_base_model import CustomKlineDataset
from config_loader import CustomFinetuneConfig
from model_factory import build_initial_tokenizer
from checkpoint_utils import find_checkpoint, load_checkpoint, save_checkpoint


def set_seed(seed: int, rank: int = 0):
    actual_seed = seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"tokenizer_training_rank_{rank}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_file = os.path.join(log_dir, f"tokenizer_training_rank_{rank}.log")
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

    logger.info(f"=== Tokenizer Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return logger


def create_dataloaders(config):
    from torch.utils.data import DataLoader

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating tokenizer training data loaders...")

    train_dataset = CustomKlineDataset(
        data_paths=config.data_paths,
        data_type="train",
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
        data_type="val",
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


def train_tokenizer(model, device, config, save_dir, logger):
    logger.info("Starting tokenizer training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1

    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.tokenizer_epochs,
        pct_start=0.03,
        div_factor=10
    )

    # Mixed precision: only meaningful on CUDA. fp16 needs a GradScaler to
    # avoid gradient underflow; bf16 has fp32's exponent range so the scaler
    # is created disabled (a no-op) in that case.
    use_amp = bool(getattr(config, 'use_amp', False)) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if getattr(config, 'amp_dtype', 'fp16') == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if rank == 0:
        print(f"Mixed precision (AMP): {use_amp} (dtype: {amp_dtype if use_amp else 'n/a'})")

    start_epoch = 0
    best_val_loss = float("inf")
    if getattr(config, 'resume', True):
        checkpoint_path = find_checkpoint(save_dir)
        if checkpoint_path:
            start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device=device, scaler=scaler)
            msg = f"Resuming tokenizer training from {checkpoint_path}: starting at epoch {start_epoch + 1}/{config.tokenizer_epochs}, best_val_loss so far {best_val_loss:.4f}"
            logger.info(msg)
            if rank == 0:
                print(msg)

    if start_epoch >= config.tokenizer_epochs:
        msg = f"Checkpoint already completed all {config.tokenizer_epochs} configured epochs, nothing to resume"
        logger.info(msg)
        if rank == 0:
            print(msg)
        return best_val_loss

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    batch_idx_global = start_epoch * len(train_loader)

    accumulation_steps = getattr(config, 'accumulation_steps', 1)
    checkpoint_every_n_steps = getattr(config, 'checkpoint_every_n_steps', 0)

    for epoch in range(start_epoch, config.tokenizer_epochs):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch_idx, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)

            current_batch_total_loss = 0.0
            for j in range(accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    zs, bsq_loss, _, _ = (model.module if use_ddp else model)(batch_x)
                    z_pre, z = zs

                    recon_loss_pre = F.mse_loss(z_pre, batch_x)
                    recon_loss_all = F.mse_loss(z, batch_x)
                    recon_loss = recon_loss_pre + recon_loss_all
                    loss = (recon_loss + bsq_loss) / 2

                loss_scaled = loss / accumulation_steps
                current_batch_total_loss += loss.item()
                scaler.scale(loss_scaled).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if (batch_idx_global + 1) % config.log_interval == 0:
                avg_loss = current_batch_total_loss / accumulation_steps
                lr = optimizer.param_groups[0]["lr"]
                log_msg = (f"[Epoch {epoch+1}/{config.tokenizer_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {avg_loss:.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)

                detail_msg = (f"  - VQ Loss: {bsq_loss.item():.4f}\n"
                            f"  - Recon Loss Pre: {recon_loss_pre.item():.4f}\n"
                            f"  - Recon Loss All: {recon_loss_all.item():.4f}")
                logger.info(detail_msg)
                if rank == 0:
                    print(detail_msg)

            batch_idx_global += 1

            if checkpoint_every_n_steps > 0 and rank == 0 and batch_idx_global % checkpoint_every_n_steps == 0:
                # Mid-epoch safety net: record the *previous* completed epoch as the
                # resume point (not the in-progress one), so resuming just re-runs
                # the interrupted epoch from its start rather than attempting exact
                # mid-epoch resumption.
                save_checkpoint(save_dir, model, optimizer, scheduler, epoch - 1, best_val_loss,
                                 extra={'mid_epoch_step': batch_idx_global}, scaler=scaler)

        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    zs, _, _, _ = (model.module if use_ddp else model)(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)

                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)

        if use_ddp:
            tensor_sum = torch.tensor([tot_val_loss_sum_rank, val_sample_count_rank], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            tot_val_loss_all = tensor_sum[0].item()
            val_count_all = int(tensor_sum[1].item())
            avg_val_loss = (tot_val_loss_all / val_count_all) if val_count_all > 0 else 0.0
        else:
            avg_val_loss = tot_val_loss_sum_rank / val_sample_count_rank if val_sample_count_rank > 0 else 0

        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.tokenizer_epochs} Summary ---\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Epoch Time: {format_time(epoch_time)}\n"
                       f"Total Training Time: {format_time(time.time() - epoch_start_time)}\n")
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

        if rank == 0:
            checkpoint_path = save_checkpoint(save_dir, model, optimizer, scheduler, epoch, best_val_loss, scaler=scaler)
            resume_msg = f"Checkpoint saved to: {checkpoint_path} (resume point: epoch {epoch + 1}/{config.tokenizer_epochs})"
            logger.info(resume_msg)
            print(resume_msg)

    return best_val_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Kronos Tokenizer Fine-tuning Training (technical-indicator features)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()

    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.tokenizer_save_path, exist_ok=True)

    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)

    set_seed(config.seed)

    logger.info(f"Initializing tokenizer (pre_trained={getattr(config, 'pre_trained_tokenizer', False)}, d_in={config.d_in})...")
    print(f"Initializing tokenizer (pre_trained={getattr(config, 'pre_trained_tokenizer', False)}, d_in={config.d_in})...")
    tokenizer = build_initial_tokenizer(config)
    tokenizer = tokenizer.to(device)

    model_size = get_model_size(tokenizer)
    logger.info(f"Tokenizer parameters: {model_size}")
    print(f"Tokenizer parameters: {model_size}")

    logger.info("=== Training Configuration ===")
    logger.info(f"Data files ({len(config.data_paths)}): {config.data_paths}")
    logger.info(f"Feature list ({config.d_in} dims): {config.feature_list}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.tokenizer_learning_rate}")
    logger.info(f"Training epochs: {config.tokenizer_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision (AMP): {config.use_amp} (dtype: {config.amp_dtype})")
    logger.info(f"Distributed training: False")

    logger.info("Starting tokenizer fine-tuning training...")
    print("Starting tokenizer fine-tuning training...")
    best_val_loss = train_tokenizer(tokenizer, device, config, config.tokenizer_save_path, logger)

    final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.tokenizer_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
