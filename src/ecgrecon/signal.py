"""Signal helpers shared by the reconstruction methods.

Runs of observed samples, zero-phase high-pass of observed runs, gap filling
and a two-stage moving-median baseline estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import sosfiltfilt

FS_DEFAULT = 500


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` of the True runs of a 1-D boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def highpass_observed(x: np.ndarray, m: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase high-pass of every observed run of every column of ``x``.

    Unobserved samples are NaN in the output. Runs shorter than the filter's
    padding are only mean-removed.
    """
    out = np.full(x.shape, np.nan)
    pad_needed = 3 * (2 * len(sos) + 1)
    for k in range(x.shape[1]):
        for a, b in contiguous_runs(m[:, k]):
            seg = x[a:b, k]
            out[a:b, k] = sosfiltfilt(sos, seg) if b - a > pad_needed else seg - seg.mean()
    return out


def fill_gaps(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Copy of the 1-D signal ``x`` with the samples where ``m`` is False
    replaced by linear interpolation between the neighbouring observed
    samples (constant extrapolation at the edges). All-unobserved: zeros."""
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=bool)
    if m.all():
        return x.copy()
    if not m.any():
        return np.zeros_like(x)
    t = np.arange(len(x))
    out = x.copy()
    out[~m] = np.interp(t[~m], t[m], x[m])
    return out


def median_baseline(x: np.ndarray, fs: int = FS_DEFAULT, decim: int = 10,
                    mode: str = "reflect") -> np.ndarray:
    """Baseline-wander estimate of a 1-D or ``(n, k)`` signal.

    Classic two-stage moving median (200 ms removes QRS/P, 600 ms removes the
    T wave), evaluated on 20 ms block means for speed and linearly
    interpolated back to the full rate.
    """
    x = np.asarray(x, float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]
    nb = int(np.ceil(n / decim))
    pad = nb * decim - n
    xp = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)]) if pad else x
    blocks = xp.reshape(nb, decim, -1).mean(axis=1)
    w1 = max(1, int(round(0.2 * fs / decim)) | 1)
    w2 = max(1, int(round(0.6 * fs / decim)) | 1)
    b = median_filter(blocks, size=(w1, 1), mode=mode)
    b = median_filter(b, size=(w2, 1), mode=mode)
    # linear interpolation from block centres back to every sample
    pos = np.clip((np.arange(n) + 0.5) / decim - 0.5, 0, nb - 1)
    i0 = np.minimum(np.floor(pos).astype(int), max(nb - 2, 0))
    fr = (pos - i0)[:, None]
    i1 = np.minimum(i0 + 1, nb - 1)
    out = (1 - fr) * b[i0] + fr * b[i1]
    return out[:, 0] if squeeze else out
