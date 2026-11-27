"""Pure NumPy missingness operations on T,C,H,W images before normalization."""
import numpy as np

RATIOS = (0., .10, .25, .50, .75)
PATTERNS = ('random', 'block', 'ed', 'es', 'systole', 'diastole')
HANDLINGS = ('zero', 'removal', 'repeat', 'interpolate')


def make_mask(length, ratio, pattern, seed, phase=None):
    if length < 1 or not 0 <= ratio < 1 or pattern not in PATTERNS:
        raise ValueError('Invalid clip length, missing ratio, or pattern.')
    k = min(length - 1, int(np.floor(length * ratio + .5)))
    mask = np.ones(length, dtype=bool)
    if k == 0:
        return mask
    rng = np.random.default_rng(seed)
    if pattern == 'random':
        mask[rng.choice(length, k, replace=False)] = False
        return mask
    if pattern == 'block':
        start = int(rng.integers(length - k + 1))
    else:
        if phase is None or not (0 <= phase['ed'] < phase['es'] < length - 1):
            raise ValueError('Valid ED/ES sample positions and post-ES samples are required.')
        centers = {'ed': phase['ed'], 'es': phase['es'],
                   'systole': (phase['ed'] + phase['es']) / 2,
                   'diastole': (phase['es'] + length - 1) / 2}
        start = int(np.floor(centers[pattern] - (k - 1) / 2 + .5))
        start = int(np.clip(start, 0, length - k))
    mask[start:start + k] = False
    return mask


def handle_missing(frames, mask, times, method):
    frames, mask, times = np.asarray(frames), np.asarray(mask, bool), np.asarray(times)
    if method not in HANDLINGS or len(frames) != len(mask) or len(times) != len(mask):
        raise ValueError('Invalid handling method or inconsistent sequence lengths.')
    if not mask.any():
        raise ValueError('At least one observed frame is required.')
    if method == 'removal':
        return frames[mask].copy(), np.ones(mask.sum(), bool), times[mask].copy()
    out = frames.copy()
    out[~mask] = 0
    if method == 'repeat':
        last = None
        for i in range(len(mask)):
            if mask[i]:
                last = out[i].copy()
            elif last is not None:
                out[i] = last
        return out, mask.copy(), times.copy()
    if method == 'interpolate':
        obs_idx = np.flatnonzero(mask)
        obs_times = times[obs_idx]
        orig_shape = out.shape
        flat = out.reshape(len(out), -1)
        res = np.zeros_like(flat)
        for j in range(flat.shape[1]):
            res[:, j] = np.interp(times, obs_times, flat[obs_idx, j])
        return res.reshape(orig_shape), mask.copy(), times.copy()
    return out, mask.copy(), times.copy()
