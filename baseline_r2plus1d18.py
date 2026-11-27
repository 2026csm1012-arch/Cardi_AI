"""
EchoNet-Dynamic Baseline Video Regression Model (R(2+1)D-18)
Project 6: Cardiac Function Prediction from Echocardiography Videos

- Architecture: R(2+1)D-18 modified for 1-channel grey-scale input (B, 1, 32, 112, 112)
- Output: Scalar Left-Ventricular Ejection Fraction (EF)
- Clip Sampling: T = 32 frames, stride s = 2 (32 frames, 1 in 2)
- Official Splits: TRAIN / VAL / TEST from FileList.csv
- Kaggle-ready: Auto-detects /kaggle/input dataset path and writes outputs to /kaggle/working
"""

import os
import sys
import time
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from tqdm.auto import tqdm


# ==============================================================================
# 1. Configuration
# ==============================================================================
@dataclass
class Config:
    # Paths (set data_dir explicitly if not running on Kaggle)
    data_dir: Optional[str] = None
    output_dir: str = "/kaggle/working/outputs" if os.path.exists("/kaggle/working") else "./outputs"

    # Clip settings (Project 6 reference: T=32, s=2 -> 32 frames, 1 in 2)
    clip_len: int = 32
    frame_step: int = 2
    img_size: int = 112

    # Training hyperparameters
    epochs: int = 45
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    pretrained: bool = True
    use_amp: bool = True
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_echonet_dir(explicit_dir: Optional[str] = None) -> Path:
    """Locate the EchoNet-Dynamic directory containing FileList.csv and Videos/."""
    if explicit_dir is not None:
        p = Path(explicit_dir)
        if (p / "FileList.csv").exists() and (p / "Videos").exists():
            return p
        raise FileNotFoundError(f"FileList.csv or Videos/ not found in {p}")

    search_roots = [Path("/kaggle/input"), Path("."), Path("./data")]
    for root in search_roots:
        if not root.exists():
            continue
        if (root / "FileList.csv").exists() and (root / "Videos").exists():
            return root
        for csv_path in root.rglob("FileList.csv"):
            parent = csv_path.parent
            if (parent / "Videos").exists():
                return parent

    raise FileNotFoundError(
        "Could not find EchoNet-Dynamic dataset (FileList.csv + Videos/). "
        "Set Config.data_dir to the dataset root directory."
    )


# ==============================================================================
# 2. Dataset (Official Split + 32-frame 1-in-2 Greyscale Clips)
# ==============================================================================
class EchoNetDynamicDataset(Dataset):
    """
    PyTorch Dataset for EchoNet-Dynamic EF regression.

    Loads FileList.csv, filters by the official `Split` column (TRAIN, VAL, TEST),
    and extracts a 32-frame grey-scale clip (1, 32, 112, 112) sampled at every
    `frame_step`-th frame.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "TRAIN",
        clip_len: int = 32,
        frame_step: int = 2,
        img_size: int = 112,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "Videos"
        self.split = split.upper()
        self.clip_len = clip_len
        self.frame_step = frame_step
        self.img_size = img_size

        csv_path = self.data_dir / "FileList.csv"
        df = pd.read_csv(csv_path)

        # Normalize Split column and filter
        df["Split"] = df["Split"].astype(str).str.upper().str.strip()
        df = df[df["Split"] == self.split].copy()
        df = df.dropna(subset=["FileName", "EF"]).reset_index(drop=True)

        # Resolve video file paths (.avi)
        valid_rows = []
        for idx, row in df.iterrows():
            fname = str(row["FileName"]).strip()
            if not fname.lower().endswith(".avi"):
                fname = f"{fname}.avi"
            vpath = self.videos_dir / fname
            if vpath.exists():
                valid_rows.append((str(vpath), float(row["EF"])))

        if len(valid_rows) == 0:
            raise RuntimeError(f"No valid video files found for split '{self.split}' in {self.videos_dir}")

        self.samples: List[Tuple[str, float]] = valid_rows

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, video_path: str) -> np.ndarray:
        """
        Read a (clip_len, img_size, img_size) uint8 grey-scale clip sampled every
        `frame_step` frames. Pads by repeating the last frame if video is too short.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        span = (self.clip_len - 1) * self.frame_step + 1

        if total_frames > span:
            if self.split == "TRAIN":
                start_idx = random.randint(0, total_frames - span)
            else:
                # Deterministic center clip for VAL / TEST
                start_idx = (total_frames - span) // 2
        else:
            start_idx = 0

        if start_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames: List[np.ndarray] = []
        target_end = min(total_frames, start_idx + span) if total_frames > 0 else span

        for offset in range(target_end - start_idx):
            ret, frame = cap.read()
            if not ret:
                break
            if offset % self.frame_step == 0:
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                if gray.shape[0] != self.img_size or gray.shape[1] != self.img_size:
                    gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                frames.append(gray)
                if len(frames) == self.clip_len:
                    break

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((self.img_size, self.img_size), dtype=np.uint8)]

        # Pad short videos by repeating the last available frame
        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        return np.stack(frames, axis=0)  # (T, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, ef = self.samples[idx]
        clip_np = self._load_clip(video_path)  # (T, H, W), uint8

        # Normalize to [-1, 1] and add channel dim -> (1, T, H, W)
        clip = torch.from_numpy(clip_np).float().unsqueeze(0) / 255.0
        clip = (clip - 0.5) / 0.5

        target = torch.tensor(ef, dtype=torch.float32)
        return clip, target


# ==============================================================================
# 3. Model: R(2+1)D-18 for 1-Channel Greyscale Video Regression
# ==============================================================================
class R2Plus1D18Baseline(nn.Module):
    """
    R(2+1)D-18 baseline video regression network.
    Accepts (B, 1, 32, 112, 112) and outputs (B,) predicted EF.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = R2Plus1D_18_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            self.backbone = r2plus1d_18(weights=weights)
        except Exception as exc:
            print(f"[Warning] Could not load pretrained weights ({exc}); using random init.")
            self.backbone = r2plus1d_18(weights=None)

        # Adapt stem[0] from 3-channel RGB to 1-channel grey-scale
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained and weights is not None:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        self.backbone.stem[0] = new_conv

        # Replace classification head with scalar EF regression head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 112, 112) -> returns (B,)
        return self.backbone(x).squeeze(-1)


# ==============================================================================
# 4. Metrics & Training / Evaluation Loops
# ==============================================================================
def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, R2, and Pearson correlation."""
    diff = preds - targets
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    if np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
        pearson_r = float(np.corrcoef(preds, targets)[0, 1])
    else:
        pearson_r = 0.0

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "pearson_r": pearson_r}


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


# ==============================================================================
# 5. Main Training Pipeline
# ==============================================================================
def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and (device.type == "cuda")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = find_echonet_dir(cfg.data_dir)

    print("=" * 80)
    print("EchoNet-Dynamic Baseline: R(2+1)D-18 Video Regression")
    print("=" * 80)
    print(f"Device       : {device} (AMP: {use_amp})")
    print(f"Dataset Dir  : {data_dir}")
    print(f"Output Dir   : {output_dir}")
    print(f"Clip Config  : {cfg.clip_len} frames, 1 in {cfg.frame_step}, {cfg.img_size}x{cfg.img_size} grey-scale")
    print("-" * 80)

    # Datasets & DataLoaders
    train_ds = EchoNetDynamicDataset(data_dir, "TRAIN", cfg.clip_len, cfg.frame_step, cfg.img_size)
    val_ds = EchoNetDynamicDataset(data_dir, "VAL", cfg.clip_len, cfg.frame_step, cfg.img_size)
    test_ds = EchoNetDynamicDataset(data_dir, "TEST", cfg.clip_len, cfg.frame_step, cfg.img_size)

    print(f"Split Sizes  : TRAIN={len(train_ds)} | VAL={len(val_ds)} | TEST={len(test_ds)}")
    print("=" * 80)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model, Loss, Optimizer, Scheduler
    model = R2Plus1D18Baseline(pretrained=cfg.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params : {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    best_val_mae = float("inf")
    best_ckpt_path = output_dir / "best_r2plus1d18.pt"
    history: List[Dict[str, float]] = []

    # Training Loop
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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg.__dict__,
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

    # Final Test Evaluation using Best Checkpoint
    print("\n" + "=" * 80)
    print(f"Loading best checkpoint from {best_ckpt_path} (Best Val MAE: {best_val_mae:.2f})")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, use_amp, desc="Test")
    print("=" * 80)
    print("OFFICIAL TEST SET RESULTS (Complete Videos: 32 frames, 1 in 2)")
    print("-" * 80)
    print(f"MAE        : {test_metrics['mae']:.4f}  (Project 6 Ref: 4.22)")
    print(f"MSE        : {test_metrics['mse']:.4f}")
    print(f"RMSE       : {test_metrics['rmse']:.4f}  (Project 6 Ref: 5.56)")
    print(f"R2         : {test_metrics['r2']:.4f}  (Project 6 Ref: 0.79)")
    print(f"Pearson r  : {test_metrics['pearson_r']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
