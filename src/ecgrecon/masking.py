"""Observation masks for the reconstruction task.

The default task mimics a printed 12-lead ECG (3x4 layout plus rhythm strips):

=========  ==========  ============  ===========  ===========
leads      0-2.5 s     2.5-5 s       5-7.5 s      7.5-10 s
=========  ==========  ============  ===========  ===========
I,II,III   observed    -             -            -
aVR,aVL,F  -           observed      -            -
V1,V2,V3   -           -             observed     -
V4,V5,V6   -           -             -            observed
II,V1,V5   observed for the full 10 s (rhythm strips)
=========  ==========  ============  ===========  ===========

This matches ``process_ecg`` in the original U-Net notebook exactly, including
the 15-channel network input (12 masked leads + full V1, II, V5).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .leads import INDEPENDENT_IDX, LEADS, LIMB_FROM_I_II, LIMB_LEADS

#: Channel order of the 15-channel U-Net input used in the notebook.
UNET_CHANNELS: tuple[int, ...] = tuple(range(12)) + (6, 1, 10)

PAPER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("I", "II", "III"),
    ("aVR", "aVL", "aVF"),
    ("V1", "V2", "V3"),
    ("V4", "V5", "V6"),
)
RHYTHM_LEADS: tuple[str, ...] = ("II", "V1", "V5")


def paper_mask(
    n_samples: int = 5000,
    segment: int = 1250,
    rhythm: Sequence[str] = RHYTHM_LEADS,
) -> np.ndarray:
    """Boolean mask ``(n_samples, 12)``; True where a sample is observed."""
    mask = np.zeros((n_samples, 12), dtype=bool)
    for q, group in enumerate(PAPER_GROUPS):
        start = q * segment
        stop = n_samples if q == len(PAPER_GROUPS) - 1 else (q + 1) * segment
        for name in group:
            mask[start:stop, LEADS.index(name)] = True
    for name in rhythm:
        mask[:, LEADS.index(name)] = True
    return mask


def apply_mask(ecg: np.ndarray, mask: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Copy of ``ecg`` with unobserved samples replaced by ``fill``."""
    out = np.array(ecg, dtype=float, copy=True)
    out[~mask] = fill
    return out


def to_unet_input(ecg: np.ndarray, quartile: int = 1250) -> np.ndarray:
    """Reproduce the notebook's ``process_ecg`` for one record: ``(n, 15)``."""
    ecg = np.asarray(ecg, dtype=float)
    n = ecg.shape[0]
    x = ecg[:, list(UNET_CHANNELS)].copy()
    x[quartile:, 0:3] = 0
    x[np.r_[0:quartile, 2 * quartile:n], 3:6] = 0
    x[np.r_[0:2 * quartile, 3 * quartile:n], 6:9] = 0
    x[: 3 * quartile, 9:12] = 0
    return x


def from_unet_input(x15: np.ndarray, quartile: int = 1250) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 15-channel notebook input into ``(ecg_obs, mask)`` where
    ``ecg_obs`` is ``(n, 12)`` with NaN at unobserved samples."""
    x15 = np.asarray(x15, dtype=float)
    n = x15.shape[0]
    mask = paper_mask(n, quartile)
    ecg = x15[:, :12].copy()
    for src, lead in zip((12, 13, 14), ("V1", "II", "V5")):
        ecg[:, LEADS.index(lead)] = x15[:, src]
    return apply_mask(ecg, mask), mask


def limb_basis_from_observed(
    ecg_obs: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover leads I and II wherever two independent limb leads are observed.

    Returns ``(i_ii, known)`` with ``i_ii`` shape ``(n, 2)`` (NaN where not
    determined) and ``known`` shape ``(n, 2)`` boolean. Where only one limb
    lead is observed and it is I or II itself, that lead is marked known.
    """
    n = ecg_obs.shape[0]
    M = np.array([LIMB_FROM_I_II[k] for k in LIMB_LEADS])           # (6, 2)
    limb_idx = [LEADS.index(k) for k in LIMB_LEADS]
    lm = mask[:, limb_idx] & np.isfinite(ecg_obs[:, limb_idx])      # NaN counts as missing
    lv = np.where(lm, ecg_obs[:, limb_idx], 0.0)
    out = np.full((n, 2), np.nan)
    known = np.zeros((n, 2), dtype=bool)
    # Group samples by their limb observation pattern (few distinct patterns).
    patterns, inverse = np.unique(lm, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    for p_idx, pat in enumerate(patterns):
        rows = np.flatnonzero(inverse == p_idx)
        if not pat.any():
            continue
        Mp = M[pat]
        if np.linalg.matrix_rank(Mp) == 2:
            out[rows] = lv[rows][:, pat] @ np.linalg.pinv(Mp).T
            known[rows] = True
        else:
            for c, name in enumerate(("I", "II")):
                j = LIMB_LEADS.index(name)
                if pat[j]:
                    out[rows, c] = lv[rows, j]
                    known[rows, c] = True
    return out, known


def independent_observations(
    ecg_obs: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Observed values of the 8 independent leads (I, II, V1-V6).

    Returns ``(x8, m8)``: ``x8`` ``(n, 8)`` with NaN where unknown and the
    boolean mask ``m8``. Limb leads are pooled via Einthoven/Goldberger, so
    e.g. observing aVL and II also determines I.
    """
    i_ii, known = limb_basis_from_observed(ecg_obs, mask)
    chest_idx = list(INDEPENDENT_IDX[2:])
    chest_known = mask[:, chest_idx] & np.isfinite(ecg_obs[:, chest_idx])
    x8 = np.concatenate([i_ii, np.where(chest_known, ecg_obs[:, chest_idx], np.nan)], axis=1)
    m8 = np.concatenate([known, chest_known], axis=1)
    return x8, m8


def determined_mask(mask: np.ndarray) -> np.ndarray:
    """Samples whose value is known from the observations alone.

    True where a lead is observed, and for all six limb leads wherever two
    limb leads are observed (Einthoven/Goldberger: any two limb leads
    determine the others). ``mask`` is ``(..., n, 12)``; values are not
    needed. ``~determined_mask(mask)`` are the samples a method genuinely has
    to predict.
    """
    mask = np.asarray(mask, dtype=bool)
    limb_idx = [LEADS.index(k) for k in LIMB_LEADS]
    M = np.array([LIMB_FROM_I_II[k] for k in LIMB_LEADS])           # (6, 2)
    lm = mask[..., limb_idx].reshape(-1, len(limb_idx))
    patterns, inverse = np.unique(lm, axis=0, return_inverse=True)
    rank2 = np.array([bool(p.any()) and np.linalg.matrix_rank(M[p]) == 2 for p in patterns])
    full = rank2[inverse.reshape(-1)].reshape(mask.shape[:-1])
    out = mask.copy()
    out[..., limb_idx] |= full[..., None]
    return out
