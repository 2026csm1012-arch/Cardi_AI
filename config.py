"""
Central configuration for EchoNet-Dynamic EF regression experiments.
Adjust hyperparameters, paths, and model selection here (or override via CLI in train.py).
"""

import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class Config:
    # Model selection (registered in models/__init__.py)
    model_name: str = "r2plus1d_18"
    pretrained: bool = True

    # Paths (leave data_dir=None to auto-detect on Kaggle or ./EchoNet-Dynamic)
    data_dir: Optional[str] = None
    output_dir: str = (
        "/kaggle/working/outputs"
        if os.path.exists("/kaggle/working")
        else "./outputs"
    )

    # Clip settings (Project 6 reference: T=32 frames, stride s=2 -> 1 in 2)
    clip_len: int = 32
    frame_step: int = 2
    img_size: int = 112

    # Training hyperparameters
    epochs: int = 45
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    num_workers: int = 2
    use_amp: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
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
        "Set Config.data_dir in config.py or pass --data_dir."
    )
