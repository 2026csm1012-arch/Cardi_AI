"""Sampling utilities for video frames."""
import numpy as np


def clip_indices(num_frames, length=32, stride=2, rng=None):
    if length <= 0 or stride <= 0:
        raise ValueError('Length and stride must be positive.')
    span = (length - 1) * stride + 1
    if num_frames > span and rng is not None:
        start = int(rng.integers(num_frames - span + 1))
    else:
        start = 0
    indices = np.arange(start, start + length * stride, stride)
    return np.clip(indices, 0, num_frames - 1)


def phase_indices(num_frames, ed, es, length=32, stride=2):
    if ed >= es:
        raise ValueError('ED must be strictly before ES.')
    if (es - ed) % stride != 0:
        raise ValueError('parity: (es - ed) must be divisible by stride.')
    span_samples = (es - ed) // stride
    if span_samples >= length - 1:
        raise ValueError('ED/ES span exceeds or equals length - 1.')
    
    needed_frames = (length - 1) * stride + 1
    if num_frames < needed_frames:
        raise ValueError(f'Video length ({num_frames}) insufficient for phase clip ({needed_frames}).')

    k = (length - span_samples) // 2
    start = ed - k * stride
    if start < 0:
        start = ed % stride
    if start + (length - 1) * stride >= num_frames:
        start = num_frames - 1 - (length - 1) * stride
        start = start - ((start - ed) % stride)

    if start < 0 or start > ed or start + (length - 1) * stride >= num_frames:
        raise ValueError('Cannot fit ED/ES in valid clip range.')

    indices = np.arange(start, start + length * stride, stride)
    phase = {'ed': int((ed - start) // stride), 'es': int((es - start) // stride)}
    if not (0 <= phase['ed'] < phase['es'] < length - 1):
        raise ValueError('Invalid phase positions inside clip.')
    return indices, phase
