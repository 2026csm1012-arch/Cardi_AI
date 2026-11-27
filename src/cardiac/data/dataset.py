from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from cardiac.data.sampling import clip_indices
from cardiac.data.missing import make_mask, handle_missing, RATIOS
from cardiac.utils.io import stable_seed, digest


def load_video(path, expected=None, size=112):
    """Load video frames with OpenCV, convert to grayscale and normalize."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Video file not found: {path}")

    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape != (size, size):
                gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        cap.release()
    except Exception as e:
        raise RuntimeError(f"[ERROR] Exception occurred while reading video {path}: {str(e)}") from e

    if not frames:
        raise ValueError(f'[ERROR] Decoded 0 frames from video: {path}')

    if expected is not None and len(frames) != expected:
        # Permit minor container discrepancy (1 frame difference) if frames exist, otherwise raise
        if abs(len(frames) - expected) > 1:
            raise ValueError(f'[ERROR] Video frame count mismatch for {path}: decoded {len(frames)} frames, expected {expected}')

    return np.stack(frames).astype(np.float32)[:, None] / 255.0


def normalize(frames):
    """Fixed data-independent normalization. A raw black pixel becomes -1."""
    return np.repeat((frames - 0.5) / 0.5, 3, axis=1).astype(np.float32)


def verify_files(root, manifest, check_all_hashes=False):
    """Verify metadata and video files existence and optional integrity hashes."""
    if manifest.get('errors'):
        print(f"[WARNING] Manifest contains {len(manifest['errors'])} recorded errors.")

    for name, expected in manifest.get('source_hashes', {}).items():
        src_file = Path(root) / name
        if not src_file.exists():
            raise FileNotFoundError(f"[ERROR] Dataset metadata file missing: {src_file}")
        if digest(src_file) != expected:
            raise ValueError(f'[ERROR] Dataset metadata changed: {name}. Expected SHA256 {expected[:8]}...')

    records = manifest.get('records', [])
    if not records:
        raise ValueError("[ERROR] Manifest contains 0 valid video records.")

    if check_all_hashes:
        print(f"[CHECKPOINT] Auditing SHA256 hashes for all {len(records)} videos (strict mode)...")
        for row in records:
            p = Path(root) / row['path']
            if not p.exists():
                raise FileNotFoundError(f"[ERROR] Video file missing: {p}")
            if digest(p) != row['sha256']:
                raise ValueError(f"[ERROR] Video SHA256 mismatch for ID: {row['id']}. Rebuild manifest.")
    else:
        # Fast verification: verify existence of files and sample checks
        sample_count = min(len(records), 10)
        for row in records[:sample_count]:
            p = Path(root) / row['path']
            if not p.exists():
                raise FileNotFoundError(f"[ERROR] Video file missing: {p}")


class EchoDataset(Dataset):
    def __init__(self, root, manifest, split, config, train=False, limit=None):
        self.root, self.config, self.train = Path(root), config, train
        self.records = sorted([r for r in manifest['records'] if r['split'] == split], key=lambda r: r['id'])
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f'[ERROR] No records found for split {split}.')
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, c = self.records[index], self.config
        rng = np.random.default_rng(stable_seed(c['seed'], self.epoch, row['id']))
        video_path = self.root / row['path']

        try:
            video = load_video(video_path, row['frames'], c['size'])
        except Exception as e:
            # Provide helpful context on error
            raise RuntimeError(f"Error loading sample index {index} (ID: {row['id']}) from {video_path}: {e}")

        indices = clip_indices(len(video), c['length'], c['stride'], rng if self.train else None)
        frames = video[indices]
        mask = np.ones(len(indices), bool)
        if self.train and c['corruption']:
            ratio = float(rng.choice(RATIOS))
            pattern = str(rng.choice(['random', 'block']))
            mask = make_mask(len(indices), ratio, pattern, int(rng.integers(2**32)))

        # Times are in sampled-frame intervals; repeated short-video frames have zero elapsed time.
        times = (indices - indices[0]).astype(np.float32) / c['stride']
        frames, mask, times = handle_missing(frames, mask, times, 'zero')

        return {
            'video': torch.from_numpy(normalize(frames)),
            'mask': torch.from_numpy(mask),
            'times': torch.from_numpy(times),
            'target': torch.tensor(row['ef'], dtype=torch.float32),
            'id': row['id']
        }
