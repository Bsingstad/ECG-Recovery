"""Common interface for reconstruction methods.

Contract for :meth:`Reconstructor.reconstruct`:

* input ``ecg_obs`` ``(n, 12)`` with NaN (or arbitrary values) where
  ``mask`` ``(n, 12)`` is False;
* output ``(n, 12)`` where observed samples are returned unchanged and the
  limb leads satisfy Einthoven/Goldberger wherever the observations do.

Methods predict only the 8 independent leads (I, II, V1-V6); use
:func:`finalize` to expand to 12 leads and re-insert the observations.
"""

from __future__ import annotations

import numpy as np

from ..leads import expand_independent
from ..masking import independent_observations


class Reconstructor:
    """Base class. Subclasses implement :meth:`predict_independent`."""

    name: str = "base"

    def fit(self, train_ecgs: np.ndarray | None = None) -> "Reconstructor":
        """Optional population-level fitting on training ECGs ``(N, n, 12)``."""
        return self

    def predict_independent(self, x8: np.ndarray, m8: np.ndarray) -> np.ndarray:
        """Given observed independent leads ``x8`` (NaN where ``~m8``),
        return a full ``(n, 8)`` estimate."""
        raise NotImplementedError

    def reconstruct(self, ecg_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x8, m8 = independent_observations(ecg_obs, mask)
        est8 = self.predict_independent(x8, m8)
        return finalize(est8, x8, m8, ecg_obs, mask)

    def reconstruct_batch(self, ecg_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            mask = np.broadcast_to(mask, ecg_obs.shape)
        return np.stack([self.reconstruct(e, m) for e, m in zip(ecg_obs, mask)])


def finalize(
    est8: np.ndarray, x8: np.ndarray, m8: np.ndarray, ecg_obs: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Insert known independent values, expand to 12 leads, keep observations.

    Non-finite estimates are replaced by 0 so NaN can never spread across
    leads through the Einthoven expansion; non-finite "observed" samples are
    replaced by the reconstruction."""
    est8 = np.where(m8 & np.isfinite(x8), x8, est8)
    est8 = np.where(np.isfinite(est8), est8, 0.0)
    out = expand_independent(est8)
    return np.where(mask & np.isfinite(ecg_obs), ecg_obs, out)
