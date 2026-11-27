"""
EchoNet-Dynamic PyTorch Dataset.
Loads FileList.csv, filters by official Split (TRAIN / VAL / TEST),
and extracts 32-frame grey-scale clips sampled at every `frame_step`-th frame.
"""

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EchoNetDynamicDataset(Dataset):
    """
    PyTorch Dataset for EchoNet-Dynamic EF regression.

    Returns:
        clip: torch.Tensor of shape (1, clip_len, img_size, img_size), float32
        target: torch.Tensor scalar (EF), float32
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

        df["Split"] = df["Split"].astype(str).str.upper().str.strip()
        df = df[df["Split"] == self.split].copy()
        df = df.dropna(subset=["FileName", "EF"]).reset_index(drop=True)

        valid_rows: List[Tuple[str, float]] = []
        for _, row in df.iterrows():
            fname = str(row["FileName"]).strip()
            if not fname.lower().endswith(".avi"):
                fname = f"{fname}.avi"
            vpath = self.videos_dir / fname
            if vpath.exists():
                valid_rows.append((str(vpath), float(row["EF"])))

        if not valid_rows:
            raise RuntimeError(
                f"No valid video files found for split '{self.split}' in {self.videos_dir}"
            )

        self.samples = valid_rows

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, video_path: str) -> np.ndarray:
        """
        Read a (clip_len, img_size, img_size) uint8 grey-scale clip sampled every
        `frame_step` frames. Pads short videos by repeating the last frame.
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if gray.shape[0] != self.img_size or gray.shape[1] != self.img_size:
                    gray = cv2.resize(
                        gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
                    )
                frames.append(gray)
                if len(frames) == self.clip_len:
                    break

        cap.release()

        if not frames:
            frames = [np.zeros((self.img_size, self.img_size), dtype=np.uint8)]

        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, ef = self.samples[idx]
        clip_np = self._load_clip(video_path)

        clip = torch.from_numpy(clip_np).float().unsqueeze(0) / 255.0
        clip = (clip - 0.5) / 0.5

        target = torch.tensor(ef, dtype=torch.float32)
        return clip, target
