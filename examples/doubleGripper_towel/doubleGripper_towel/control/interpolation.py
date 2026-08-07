"""Interpolation helpers for auto-step control."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def locate_time_interval(t: float, key_times: Sequence[float]) -> tuple[int, int, float]:
    """Find left/right keyframe index and interpolation alpha."""
    if not key_times:
        raise ValueError("key_times cannot be empty")
    if t <= key_times[0]:
        return 0, 0, 0.0
    if t >= key_times[-1]:
        last = len(key_times) - 1
        return last, last, 0.0

    left = 0
    for i in range(len(key_times) - 1):
        if key_times[i] <= t < key_times[i + 1]:
            left = i
            break
    right = left + 1
    t0 = key_times[left]
    t1 = key_times[right]
    if t1 <= t0:
        return left, right, 1.0
    alpha = (t - t0) / (t1 - t0)
    return left, right, float(alpha)


def lerp_scalar(v0: float, v1: float, alpha: float) -> float:
    """Scalar linear interpolation."""
    return float((1.0 - alpha) * np.asarray(v0) + alpha * np.asarray(v1))
