"""Patient-adapted spatio-temporal Gaussian conditioning.

Model
-----
After removing the baseline, the eight independent leads
``s(t) = [I, II, V1..V6](t)`` of one record are modelled as a zero-mean,
stationary Gaussian process. Stacking a short temporal context::

    z(t) = [ s(t + l_1), ..., s(t + l_K) ]          (dimension 8K)

gives a Gaussian vector whose covariance is block-Toeplitz,
``Cov(z)[a, b] = C(l_b - l_a)`` with the lead cross-covariance function
``C(tau)[i, j] = E[s_i(t) s_j(t + tau)]``.  With ``lags = (0,)`` this reduces
to the purely spatial 8x8 lead covariance of the ``population_gaussian``
baseline.

Every missing sample is predicted by the conditional mean given whatever is
observed in its temporal neighbourhood::

    s_m(t) = Sigma_mo Sigma_oo^-1 z_o(t)

Physiology
----------
In the dipole picture ``s(t) = L h(t) + r(t)`` the lead covariance is
``L S_h L^T + Psi``.  Both the lead field ``L`` (heart position/orientation,
torso shape, electrode placement) and the source covariance ``S_h`` (QRS axis,
loop shape) vary strongly between patients, so a population covariance is a
blurry average.  In the printed layout, however, every lead is observed for
2.5 s *together with* the rhythm leads II, V1, V5, which roughly span the
three dipole axes (Y, Z, X).  The record therefore carries direct information
about its own lead covariance; e.g. the V2-vs-(II, V1, V5) relation is
measured in 5-7.5 s and used to predict V2 in the other 7.5 s.

Patient covariance with missing data
------------------------------------
The prior is the population cross-correlation function re-scaled by the
patient's own lead amplitudes, ``P_ij(tau) = sigma_i sigma_j R_pop_ij(tau)``:
amplitudes are well determined from a single quarter, only the *shape* of
the dependence needs the population.

``estimator="em"`` (default) computes the MAP covariance of the stacked
vector under an inverse-Wishart prior with ``kappa`` pseudo-samples of ``P``
by EM for a Gaussian with missing values (see :meth:`em_covariance`). Lead
pairs that are never observed together (I with V2-V4, V6; V2/V3 with V4/V6)
are filled consistently with the record's own regressions on the rhythm
leads, i.e. through what the leads share with II, V1, V5.

``estimator="pairwise"`` is the simpler ablation: every entry is the
pairwise-available estimate shrunk entry-wise toward the prior,
``C_ij(tau) = (S_ij + kappa P_ij) / (n_ij + kappa)``, never-co-observed pairs
falling back to ``P``. It is much worse (val n=100: record-median r 0.86 vs
0.93), because mixing patient and prior entries produces a covariance whose
implied regressions are inconsistent.

Both estimates are made block-Toeplitz (stationarity) and projected onto the
PSD cone.

Baseline wander
---------------
Each contiguous observed run of each lead is high-passed (zero-phase
Butterworth, ``hp_cutoff`` Hz) after removing its median; the covariance and
the conditioning above use only this band, so run-specific offsets and
wander cannot bias the lead transform. The low-frequency remainder of the
missing leads is predicted separately (``wander=True``) by spatial
conditioning on the observed leads' low-frequency deviations with a
population covariance (wander is largely electrode-level and therefore
shared by leads with a common electrode or via Wilson's terminal), and the
result is put on the lead's own level, the median of its observed samples.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, sosfiltfilt

from ..leads import independent_from_12
from ..masking import independent_observations
from ..signal import contiguous_runs
from .base import Reconstructor, finalize
from .baselines import iter_patterns

FS = 500.0
_EM_MIN_ROWS = 32


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_runs = contiguous_runs   # ``(start, stop)`` of the True runs of a boolean vector


def _highpass(seg: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    """Median-removed, optionally zero-phase high-passed copy of ``seg``
    (filtered along axis 0). Short segments only get the median removed."""
    seg = seg - np.median(seg, axis=0)
    if sos is None:
        return seg
    padlen = min(len(seg) - 1, 3 * FS)
    if len(seg) < 64:
        return seg
    return sosfiltfilt(sos, seg, axis=0, padtype="odd", padlen=int(padlen))


def _crosscov_sums(s: np.ndarray, m: np.ndarray, taus: Sequence[int]):
    """Sums of lagged products and co-observation counts.

    ``S[k, i, j] = sum_t s_i(t) s_j(t + taus[k])`` over pairs where both are
    observed (``s`` must be zero where unobserved) and ``N`` the matching
    counts. Accepts ``s`` of shape (n, 8) or (R, n, 8) (summed over records).
    """
    s = s if s.ndim == 3 else s[None]
    mf = (m if m.ndim == 3 else m[None]).astype(np.float64)
    n = s.shape[1]
    S = np.empty((len(taus), s.shape[2], s.shape[2]))
    N = np.empty_like(S)
    for k, tau in enumerate(taus):
        a = abs(int(tau))
        A, B = s[:, : n - a], s[:, a:]
        MA, MB = mf[:, : n - a], mf[:, a:]
        Sk = np.einsum("rti,rtj->ij", A, B)
        Nk = np.einsum("rti,rtj->ij", MA, MB)
        if tau < 0:                       # C(-a) = C(a)^T
            Sk, Nk = Sk.T, Nk.T
        S[k], N[k] = Sk, Nk
    return S, N


def _stack_mask(m8: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """(n, 8K) observation mask of the stacked context vector z(t)."""
    n = m8.shape[0]
    out = np.zeros((n, 8 * len(lags)), dtype=bool)
    for a, l in enumerate(lags):
        blk = out[:, 8 * a: 8 * (a + 1)]
        if l >= 0:
            blk[: n - l] = m8[l:]
        else:
            blk[-l:] = m8[: n + l]
    return out


def _stack_values(s: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """(n, 8K) stacked context values (zero outside the record)."""
    n = s.shape[0]
    out = np.zeros((n, 8 * len(lags)))
    for a, l in enumerate(lags):
        blk = out[:, 8 * a: 8 * (a + 1)]
        if l >= 0:
            blk[: n - l] = s[l:]
        else:
            blk[-l:] = s[: n + l]
    return out


def _nearest_psd(M: np.ndarray, rel_floor: float = 1e-6) -> np.ndarray:
    """Symmetrise and clip eigenvalues at ``rel_floor * max eigenvalue``."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, rel_floor * max(w.max(), 1e-12))
    return (V * w) @ V.T


# --------------------------------------------------------------------------
# method
# --------------------------------------------------------------------------
class SpatioTemporalGaussian(Reconstructor):
    """Conditional mean under a patient-adapted spatio-temporal Gaussian.

    Parameters
    ----------
    lags : sequence of int
        Temporal context in samples (500 Hz); must contain 0. ``(0,)`` is
        purely spatial conditioning.
    kappa : float
        Prior pseudo-count (in samples) for shrinking the record's own
        covariance toward the amplitude-scaled population prior.
        ``0`` = patient data only; ``inf`` = population shape only (with the
        patient's amplitudes).
    hp_cutoff : float
        High-pass cutoff (Hz) for baseline removal; ``0`` = median only.
    noise : float
        Observation-noise variance added to the observed block, relative to
        each observed entry's variance (regularises the inversion and
        down-weights noisy/redundant context).
    estimator : {"em", "pairwise"}
        Patient covariance estimator (see module docstring).
    n_iter : int
        EM iterations (converged after ~10).
    wander : bool
        Also predict the low-frequency baseline of missing leads.

    Defaults are the values tuned on the validation split.
    """

    name = "spatiotemporal_gauss"

    def __init__(
        self,
        lags: Sequence[int] = (-4, -2, 0, 2, 4),
        kappa: float = 100.0,
        hp_cutoff: float = 0.5,
        noise: float = 1e-2,
        estimator: str = "em",
        n_iter: int = 20,
        wander: bool = True,
    ):
        lags = sorted({int(l) for l in lags})
        if 0 not in lags:
            raise ValueError("lags must contain 0")
        self.lags = tuple(lags)
        self.kappa = float(kappa)
        self.hp_cutoff = float(hp_cutoff)
        self.noise = float(noise)
        if estimator not in ("em", "pairwise"):
            raise ValueError("estimator must be 'em' or 'pairwise'")
        self.estimator = estimator
        self.n_iter = int(n_iter)
        self.wander = bool(wander)
        self.sos = butter(2, hp_cutoff, btype="highpass", fs=FS, output="sos") if hp_cutoff > 0 else None
        # all lag differences l_b - l_a needed for the block-Toeplitz matrix
        self.taus = tuple(sorted({b - a for a in self.lags for b in self.lags}))
        self._tau_index = {t: k for k, t in enumerate(self.taus)}
        self.R_pop: np.ndarray | None = None        # (T, 8, 8) cross-correlation
        self.C_wander: np.ndarray | None = None     # (8, 8) low-frequency covariance

    # ---------------------------------------------------------------- fit
    def fit(self, train_ecgs=None):
        """Population priors from training ECGs ``(N, n, 12)``: the
        cross-correlation function ``R_pop(tau)`` of the high-passed
        independent leads and the covariance of their low-frequency part."""
        if train_ecgs is None:
            raise ValueError("SpatioTemporalGaussian needs training ECGs")
        x = independent_from_12(np.asarray(train_ecgs, dtype=np.float64))
        x = x - np.median(x, axis=1, keepdims=True)
        hf = np.stack([_highpass(r, self.sos) for r in x])
        lf = (x - hf).reshape(-1, 8)
        self.C_wander = lf.T @ lf / len(lf)
        x = hf
        S, N = _crosscov_sums(x, np.ones(x.shape, bool), self.taus)
        C = S / N
        sd = np.sqrt(np.diag(C[self._tau_index[0]]))
        self.R_pop = C / np.outer(sd, sd)[None]
        return self

    # ------------------------------------------------------- per record
    def _baseline_removed(self, x8, m8):
        """High-passed observed runs (0 where unobserved) and per-lead level."""
        s = np.zeros_like(x8, dtype=np.float64)
        level = np.zeros(8)
        for j in range(8):
            obs = m8[:, j]
            if not obs.any():
                continue
            level[j] = np.median(x8[obs, j])
            for a, b in _runs(obs):
                s[a:b, j] = _highpass(x8[a:b, j], self.sos)
        return s, level

    def _amplitudes(self, S0: np.ndarray, N0: np.ndarray) -> np.ndarray:
        """Per-lead RMS of the observed (baseline-removed) samples; leads that
        are never observed get the median amplitude of the others."""
        n0 = np.diag(N0)
        sd = np.sqrt(np.maximum(np.diag(S0), 0.0) / np.maximum(n0, 1))
        if np.any(n0 == 0):
            sd[n0 == 0] = np.median(sd[n0 > 0]) if np.any(n0 > 0) else 1.0
        return sd

    def stacked(self, C: np.ndarray) -> np.ndarray:
        """Block-Toeplitz (8K x 8K) matrix from a cross-covariance function
        ``C(tau)`` of shape (T, 8, 8)."""
        K = len(self.lags)
        Sig = np.empty((8 * K, 8 * K))
        for a, la in enumerate(self.lags):
            for b, lb in enumerate(self.lags):
                Sig[8 * a: 8 * a + 8, 8 * b: 8 * b + 8] = C[self._tau_index[lb - la]]
        return Sig

    def toeplitz_project(self, Sig: np.ndarray) -> np.ndarray:
        """Average the blocks of equal lag difference (stationarity)."""
        C = np.zeros((len(self.taus), 8, 8))
        cnt = np.zeros(len(self.taus))
        for a, la in enumerate(self.lags):
            for b, lb in enumerate(self.lags):
                k = self._tau_index[lb - la]
                C[k] += Sig[8 * a: 8 * a + 8, 8 * b: 8 * b + 8]
                cnt[k] += 1
        return self.stacked(C / cnt[:, None, None])

    def pairwise_covariance(self, s, m8) -> np.ndarray:
        """Entry-wise shrinkage of the pairwise-available lagged covariances
        toward the prior, assembled block-Toeplitz and projected to PSD."""
        S, N = _crosscov_sums(s, m8, self.taus)
        k0 = self._tau_index[0]
        sd = self._amplitudes(S[k0], N[k0])
        P = self.R_pop * np.outer(sd, sd)[None]
        if np.isinf(self.kappa):
            return _nearest_psd(self.stacked(P))
        C = (S + self.kappa * P) / np.maximum(N + self.kappa, 1e-12)
        return _nearest_psd(self.stacked(C))

    def em_covariance(self, patterns, zv, sd) -> np.ndarray:
        """MAP estimate of the stacked covariance by EM for a Gaussian with
        missing values.

        With ``kappa`` pseudo-samples of covariance ``P`` (inverse-Wishart
        prior), each iteration is::

            E:  for each observation pattern p (samples Z_p, observed set o):
                W = Sig_mo Sig_oo^-1
                E[z z^T] summed over p = [[S_oo, S_oo W^T], [W S_oo, W S_oo W^T + n_p (Sig_mm - W Sig_om)]]
            M:  Sig = (sum_p E[z z^T] + kappa P) / (n + kappa)

        followed by block-Toeplitz averaging (stationarity) and a PSD
        projection. Unlike entry-wise shrinkage, EM fills covariances of lead
        pairs that are never observed together in a way that is consistent
        with the record's own regressions on the rhythm leads.
        """
        D = zv.shape[1]
        P = self.stacked(self.R_pop * np.outer(sd, sd)[None])
        if np.isinf(self.kappa):
            return _nearest_psd(P)
        # Sufficient statistics per pattern. Patterns occupied by only a few
        # samples (context windows straddling a quarter boundary or the record
        # edge) carry negligible information and are skipped for speed.
        stats = []
        n_tot = 0
        for pat, rows in patterns:
            o = np.flatnonzero(pat)
            if o.size == 0 or len(rows) < _EM_MIN_ROWS:
                continue
            m = np.flatnonzero(~pat)
            Zo = zv[np.ix_(rows, o)]
            stats.append((np.ix_(o, o), np.ix_(m, o), np.ix_(m, m), m.size, len(rows), Zo.T @ Zo))
            n_tot += len(rows)
        if n_tot == 0:
            return _nearest_psd(P)
        Sig = P.copy()
        jitter = 1e-9 * np.trace(P) / D
        for _ in range(self.n_iter):
            T = np.zeros((D, D))
            for ioo, imo, imm, n_m, n_p, Soo_p in stats:
                T[ioo] += Soo_p
                if n_m == 0:
                    continue
                Soo = Sig[ioo]
                Smo = Sig[imo]
                W = np.linalg.solve(Soo + jitter * np.eye(len(Soo)), Smo.T).T
                WS = W @ Soo_p
                T[imo] += WS
                T[imo[1].T, imo[0].T] += WS.T
                T[imm] += WS @ W.T + n_p * (Sig[imm] - W @ Smo.T)
            Sig = (T + self.kappa * P) / (n_tot + self.kappa)
            Sig = _nearest_psd(self.toeplitz_project(Sig))
        return Sig

    def reconstruct(self, ecg_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """As :meth:`Reconstructor.reconstruct`, but samples flagged observed
        that are not finite are treated as missing for the independent leads,
        so a NaN in one observed lead cannot spread to other leads through
        the 12-lead expansion. Observed samples are still returned unchanged."""
        x8, m8 = independent_observations(ecg_obs, mask)
        m8 = m8 & np.isfinite(x8)
        est8 = self.predict_independent(x8, m8)
        return finalize(est8, x8, m8, ecg_obs, mask)

    def predict_independent(self, x8, m8):
        if self.R_pop is None:
            raise RuntimeError("call fit() first")
        m8 = m8 & np.isfinite(x8)
        x8 = np.where(m8, x8, 0.0)
        s, level = self._baseline_removed(x8, m8)

        a0 = self.lags.index(0)
        zm = _stack_mask(m8, self.lags)
        zv = _stack_values(s, self.lags)
        # group samples by their (context) observation pattern
        packed = np.packbits(zm, axis=1)
        _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(first) + 1))
        patterns = [(zm[t0], order[bounds[k]: bounds[k + 1]]) for k, t0 in enumerate(first)]

        if self.estimator == "em":
            S0 = (s * m8).T @ s
            N0 = m8.T.astype(float) @ m8
            Sig = self.em_covariance(patterns, zv, self._amplitudes(S0, N0))
        else:
            Sig = self.pairwise_covariance(s, m8)

        est = np.zeros_like(s)
        diag = np.diag(Sig)
        for pat, rows in patterns:
            tgt = np.flatnonzero(~pat[8 * a0: 8 * a0 + 8])
            o = np.flatnonzero(pat)
            if tgt.size == 0 or o.size == 0:
                continue
            Soo = Sig[np.ix_(o, o)] + np.diag(self.noise * diag[o] + 1e-12)
            Smo = Sig[np.ix_(8 * a0 + tgt, o)]
            W = cho_solve(cho_factor(Soo), Smo.T).T
            est[np.ix_(rows, tgt)] = zv[np.ix_(rows, o)] @ W.T
        if self.wander and self.sos is not None:
            est += self._predict_wander(x8, m8, s, level)
        return est + level

    def _predict_wander(self, x8, m8, s, level) -> np.ndarray:
        """Low-frequency (baseline) part of the missing leads.

        Baseline wander is largely electrode-level (half-cell drift, motion,
        respiration) and therefore shared between leads with a common
        electrode or through Wilson's central terminal. It is predicted by
        purely spatial conditioning on the observed leads' low-frequency
        deviations, with the population low-frequency covariance."""
        lf = np.where(m8, x8 - level - s, 0.0)
        out = np.zeros_like(lf)
        C = self.C_wander
        for pat, rows in iter_patterns(m8):
            o, m = np.flatnonzero(pat), np.flatnonzero(~pat)
            if o.size == 0 or m.size == 0:
                continue
            Coo = C[np.ix_(o, o)] + np.diag(self.noise * np.diag(C)[o] + 1e-12)
            W = np.linalg.solve(Coo, C[np.ix_(m, o)].T).T
            out[np.ix_(rows, m)] = lf[np.ix_(rows, o)] @ W.T
        return out
