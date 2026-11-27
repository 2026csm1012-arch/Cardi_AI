"""
Entry point for training and evaluating EchoNet-Dynamic EF regression models.

Usage (Kaggle or Local):
    python train.py
    python train.py --model r2plus1d_18 --epochs 45 --batch_size 16 --lr 1e-4
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config, find_echonet_dir, set_seed
from dataset import EchoNetDynamicDataset
from models import build_model
from trainer import evaluate, fit


def parse_args(cfg: Config) -> Config:
    parser = argparse.ArgumentParser(description="Train EchoNet-Dynamic EF regression model")
    parser.add_argument("--model", type=str, default=cfg.model_name, help="Model name in registry")
    parser.add_argument("--data_dir", type=str, default=cfg.data_dir, help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default=cfg.output_dir, help="Output directory")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=cfg.lr, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers, help="DataLoader workers")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable pretrained backbone weights")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--seed", type=int, default=cfg.seed, help="Random seed")

    args, _ = parser.parse_known_args()  # parse_known_args is safe in Kaggle/Jupyter notebooks
    cfg.model_name = args.model
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers
    cfg.pretrained = not args.no_pretrained
    cfg.use_amp = not args.no_amp
    cfg.seed = args.seed
    return cfg


def main() -> None:
    cfg = parse_args(Config())
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and (device.type == "cuda")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = find_echonet_dir(cfg.data_dir)

    print("=" * 80)
    print(f"EchoNet-Dynamic EF Regression | Model: {cfg.model_name}")
    print("=" * 80)
    print(f"Device       : {device} (AMP: {use_amp})")
    print(f"Dataset Dir  : {data_dir}")
    print(f"Output Dir   : {output_dir}")
    print(f"Clip Config  : {cfg.clip_len} frames, 1 in {cfg.frame_step}, {cfg.img_size}x{cfg.img_size} grey-scale")
    print("-" * 80)

    train_ds = EchoNetDynamicDataset(data_dir, "TRAIN", cfg.clip_len, cfg.frame_step, cfg.img_size)
    val_ds = EchoNetDynamicDataset(data_dir, "VAL", cfg.clip_len, cfg.frame_step, cfg.img_size)
    test_ds = EchoNetDynamicDataset(data_dir, "TEST", cfg.clip_len, cfg.frame_step, cfg.img_size)
    print(f"Split Sizes  : TRAIN={len(train_ds)} | VAL={len(val_ds)} | TEST={len(test_ds)}")
    print("=" * 80)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
    )

    model = build_model(cfg.model_name, pretrained=cfg.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params : {n_params:,}")

    best_ckpt_path = fit(model, train_loader, val_loader, cfg, device, output_dir)

    print("\n" + "=" * 80)
    print(f"Loading best checkpoint: {best_ckpt_path}")
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
