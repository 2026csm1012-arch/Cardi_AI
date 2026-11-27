"""
Training and evaluation loops with Kaggle-friendly progress reporting and checkpointing.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from metrics import compute_regression_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    running_mse = 0.0
    running_mae = 0.0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for clips, targets in pbar:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = clips.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(clips)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_mse = float(torch.mean((preds - targets) ** 2).item())
            batch_mae = float(torch.mean(torch.abs(preds - targets)).item())

        running_mse += batch_mse * bs
        running_mae += batch_mae * bs
        total_samples += bs

        pbar.set_postfix(
            MSE=f"{running_mse / total_samples:.2f}",
            MAE=f"{running_mae / total_samples:.2f}",
        )

    return {
        "mse": running_mse / total_samples,
        "mae": running_mae / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    desc: str = "Val",
) -> Dict[str, float]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    pbar = tqdm(
        loader,
        desc=f"[{desc}]",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for clips, targets in pbar:
        clips = clips.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(clips)

        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(targets.numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    return compute_regression_metrics(preds_np, targets_np)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> Path:
    """Run full training loop, save best checkpoint and training_history.csv."""
    use_amp = cfg.use_amp and (device.type == "cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    best_val_mae = float("inf")
    best_ckpt_path = output_dir / f"best_{cfg.model_name}.pt"
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp, epoch, cfg.epochs
        )
        val_metrics = evaluate(model, val_loader, device, use_amp, desc=f"Epoch {epoch:02d} Val")
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": cfg.model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg.to_dict(),
                },
                best_ckpt_path,
            )

        best_tag = " *BEST*" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{cfg.epochs:02d}] | LR: {current_lr:.2e} | "
            f"Train MSE: {train_metrics['mse']:6.2f} MAE: {train_metrics['mae']:5.2f} | "
            f"Val MSE: {val_metrics['mse']:6.2f} MAE: {val_metrics['mae']:5.2f} "
            f"RMSE: {val_metrics['rmse']:5.2f} R2: {val_metrics['r2']:.3f} | "
            f"{elapsed:5.1f}s{best_tag}",
            flush=True,
        )

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_pearson_r": val_metrics["pearson_r"],
                "elapsed_sec": elapsed,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    return best_ckpt_path
