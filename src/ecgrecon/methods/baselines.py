"""Baseline reconstruction methods (population-level, no patient adaptation)."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..leads import INDEPENDENT_LEADS, independent_from_12, lead_matrix
from .base import Reconstructor


def iter_patterns(m8: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(pattern, rows)`` for each distinct per-sample observation
    pattern of the independent leads."""
    patterns, inverse = np.unique(m8, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    for k, pat in enumerate(patterns):
        yield pat, np.flatnonzero(inverse == k)


def lead_offsets(x8: np.ndarray, m8: np.ndarray) -> np.ndarray:
    """Per-lead median of the observed samples (0 for never-observed leads)."""
    off = np.zeros(x8.shape[1])
    for j in range(x8.shape[1]):
        if m8[:, j].any():
            off[j] = np.median(x8[m8[:, j], j])
    return off


class DowerDipole(Reconstructor):
    """Single-dipole inverse/forward with the fixed Dower lead matrix.

    Per sample: ``h = S A^T (A S A^T + lam I)^-1 x_obs`` (Gaussian conditioning,
    as in the edit-propagation model) and ``x_missing = L_missing h``.
    """

    name = "dower_dipole"

    def __init__(self, lam: float = 1e-3, prior: np.ndarray | None = None):
        self.lam = lam
        self.S = np.eye(3) if prior is None else np.asarray(prior, float)
        self.L8 = lead_matrix(INDEPENDENT_LEADS)

    def predict_independent(self, x8, m8):
        off = lead_offsets(x8, m8)
        xc = np.where(m8, x8 - off, 0.0)
        est = np.zeros_like(xc)
        for pat, rows in iter_patterns(m8):
            if not pat.any():
                continue
            A = self.L8[pat]
            G = self.S @ A.T @ np.linalg.inv(A @ self.S @ A.T + self.lam * np.eye(pat.sum()))
            est[rows] = xc[rows][:, pat] @ G.T @ self.L8.T
        return est + off


class PopulationGaussian(Reconstructor):
    """Conditional mean under a population Gaussian for the 8 independent
    leads: ``x_m = mu_m + S_mo S_oo^-1 (x_o - mu_o)``, where ``S`` is learned
    from training ECGs and ``mu`` is the record's own per-lead median."""

    name = "population_gaussian"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.cov: np.ndarray | None = None

    def fit(self, train_ecgs=None):
        if train_ecgs is None:
            raise ValueError("PopulationGaussian needs training ECGs")
        x = independent_from_12(np.asarray(train_ecgs, dtype=np.float64))
        x = x - np.median(x, axis=1, keepdims=True)
        x = x.reshape(-1, 8)
        self.cov = x.T @ x / len(x)
        return self

    def predict_independent(self, x8, m8):
        if self.cov is None:
            raise RuntimeError("call fit() first")
        off = lead_offsets(x8, m8)
        xc = np.where(m8, x8 - off, 0.0)
        est = np.zeros_like(xc)
        S = self.cov
        for pat, rows in iter_patterns(m8):
            if not pat.any():
                continue
            o, m = pat, ~pat
            Soo = S[np.ix_(o, o)] + self.ridge * np.eye(o.sum())
            W = S[np.ix_(m, o)] @ np.linalg.inv(Soo)
            est[np.ix_(rows, np.flatnonzero(m))] = xc[rows][:, o] @ W.T
        return est + off
