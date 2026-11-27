"""Patient-specific Bayesian lead-transform regression.

Physiology
----------
Under the dipole model of :mod:`ecgrecon.leads` every lead is a projection of
the heart vector, ``x_k(t) = l_k . h(t) + r_k(t)``. The three rhythm strips
II, V1 and V5 point roughly along the Y, Z and X axes, so ``h(t)`` -- and with
it any other lead -- is approximately a *fixed linear combination* of them::

    x_j(t) ~= sum_p a_jp x_p(t)          (p in {II, V1, V5})

The coefficients depend on the individual torso and heart (electrode
positions, heart orientation, conductivity), which is why a population matrix
such as Dower's transfers poorly. In the printed 3x4 layout, however, every
missing lead is *observed* during one quarter together with the rhythm strips
(I in 0-5 s, V2/V3 in 5-7.5 s, V4/V6 in 7.5-10 s). The record therefore
contains enough information to learn its own lead transform, which is then
applied to the quarters where the lead is missing.

Model
-----
**Stage 1 -- cardiac part (patient-specific lead transform).** For a target
lead ``j`` observed on samples ``R_j``, with ``f(t)`` the high-passed
predictor leads at a few time lags (a short FIR context: it absorbs slight
temporal dispersion and the imperfect, frequency-dependent projection of a
non-ideal dipole)::

    x_j(t) = f(t)^T w_j + b_j(t) + e(t),      t in R_j

The lead's slow baseline ``b_j`` is modelled by a low-order cosine (DCT) basis
per observed segment and removed exactly by projection (Frisch-Waugh-Lovell),
so it never contaminates ``w_j``. The coefficients get a Gaussian population
prior ``w_j ~ N(w0_j, g S0_j)``: ``w0_j`` / ``S0_j`` are the mean / covariance
across training patients of their own full-record coefficients (hierarchical,
empirical-Bayes prior; ``S0`` is the genuine between-patient variability of
the lead transform). The posterior mean is the generalised ridge estimate::

    w_j = (X'X / s2 + (g S0)^-1)^-1 (X'y / s2 + (g S0)^-1 w0)

The ECG residual is strongly autocorrelated, so a white-noise likelihood would
overstate the information in 1250 correlated samples: ``s2 = sigma^2 tau``
uses the residual variance times its integrated autocorrelation time
(effective sample size ``n / tau``). ``g`` is fixed or chosen per record and
lead by maximising the (effective-sample-size corrected) evidence.

**Baseline wander.** After stage 1, ``d_j = x_j - f^T w_j`` on ``R_j`` is the
lead's baseline plus non-dipolar residual. It is split into

* a part shared with the rhythm strips' own slow components ``s(t)``
  (common electrodes: right arm in I and II, the Wilson terminal in all chest
  leads), ``d_j ~ s(t)^T gamma_j``, fitted with the same Bayesian machinery
  and a population prior on ``gamma_j`` (``wander_regression``), and
* the lead's own wander, extrapolated into the missing quarters as the
  conditional mean of an Ornstein-Uhlenbeck process: relaxation from the value
  at the nearest observed edge to the segment mean with correlation time
  ``baseline_tau``.

**Stage 2 -- non-dipolar residual correction.** Within a quarter other leads
may be observed that are not rhythm strips (e.g. V4 and V6 while lead I is
predicted in 7.5-10 s). Their part not explained by stage 1 (``x_k - f^T
w_k``) correlates with the unexplained part of the target, but the two are
never observed together in one record, so ``e_j ~= e_K beta`` uses a
population ``beta`` learned in :meth:`PatientRegression.fit` by running
stage 1 on masked training records.

Everything is linear in the observations given the hyperparameters,
deterministic, and runs per record.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import butter

from ..leads import INDEPENDENT_LEADS, independent_from_12, lead_matrix
from ..masking import independent_observations, paper_mask
from ..signal import contiguous_runs, fill_gaps, highpass_observed  # noqa: F401
from .base import Reconstructor

_N_IND = len(INDEPENDENT_LEADS)
_EVIDENCE_GRID = np.logspace(-2, 3, 21)
#: a lead observed on at least this fraction of the record is used as a
#: predictor; its few unobserved samples (e.g. digitisation gaps) are filled
#: by linear interpolation for the features only
_PREDICTOR_COVERAGE = 0.99


# --------------------------------------------------------------------------
# signal helpers
# --------------------------------------------------------------------------
def lagged_features(z: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Stack ``z[t + lag]`` for every column of ``z`` (n, p) and every lag;
    the record edges are clamped. Output ``(n, p * len(lags))`` ordered
    predictor-major (all lags of predictor 0 first)."""
    n = z.shape[0]
    t = np.arange(n)
    cols = [z[np.clip(t + lag, 0, n - 1), p] for p in range(z.shape[1]) for lag in lags]
    return np.stack(cols, axis=1)


def nuisance_basis(runs: Sequence[tuple[int, int]], rows: np.ndarray, fs: float,
                   cutoff: float) -> np.ndarray:
    """Orthonormal basis (``len(rows)`` x q) of slow baseline wander on the
    sorted sample indices ``rows`` (the union of ``runs``): per contiguous run
    a constant plus the DCT-II cosines below ``cutoff`` Hz
    (``floor(2 * duration * cutoff)`` of them; ``cutoff=0``: constant only)."""
    blocks = []
    for a, b in runs:
        n = b - a
        k_max = int(np.floor(2.0 * n / fs * cutoff))
        tt = (np.arange(n) + 0.5) / n
        B = np.zeros((len(rows), k_max + 1))
        idx = np.searchsorted(rows, np.arange(a, b))
        B[idx, 0] = 1.0
        for k in range(1, k_max + 1):
            B[idx, k] = np.cos(np.pi * k * tt)
        blocks.append(B)
    Q, _ = np.linalg.qr(np.concatenate(blocks, axis=1))
    return Q


def integrated_autocorr_time(r: np.ndarray, max_lag: int | None = None) -> float:
    """``tau = 1 + 2 sum_k rho_k`` summed up to the first non-positive
    autocorrelation (initial positive sequence); ``n / tau`` is the effective
    number of independent samples of an autocorrelated residual."""
    r = r - r.mean()
    n = len(r)
    if n < 4 or not np.any(r):
        return 1.0
    max_lag = n // 2 if max_lag is None else min(max_lag, n - 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(r, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    rho = ac / ac[0]
    neg = np.flatnonzero(rho[1:] <= 0)
    stop = neg[0] + 1 if len(neg) else len(rho)
    return float(min(max(1.0, 1.0 + 2.0 * rho[1:stop].sum()), n))


# --------------------------------------------------------------------------
# Bayesian linear regression with a population prior
# --------------------------------------------------------------------------
def _log_evidence(XtX, Xty, yty, sigma2, w0, S0):
    """Log marginal likelihood (up to constants) of ``y = Xw + e``,
    ``e ~ N(0, sigma2 I)``, ``w ~ N(w0, S0)``, from sufficient statistics."""
    Xty_c = Xty - XtX @ w0                   # shift so the prior mean is zero
    yty_c = yty - 2 * w0 @ Xty + w0 @ XtX @ w0
    sign, logdet = np.linalg.slogdet(np.eye(len(w0)) + S0 @ XtX / sigma2)
    if sign <= 0:
        return -np.inf
    A = XtX / sigma2 + np.linalg.inv(S0)     # Woodbury for the quadratic form
    quad = yty_c / sigma2 - Xty_c @ np.linalg.solve(A, Xty_c) / sigma2 ** 2
    return -0.5 * (logdet + quad)


def bayes_regression(X: np.ndarray, y: np.ndarray, w0: np.ndarray, S0: np.ndarray,
                     prior_scale: float | str) -> np.ndarray:
    """Posterior mean of ``w`` for ``y = X w + e`` with prior
    ``N(w0, g S0)`` and autocorrelated noise.

    The noise variance ``sigma^2`` and integrated autocorrelation time ``tau``
    come from a lightly regularised least-squares fit; the likelihood is
    tempered by ``1/tau`` (effective sample size). ``prior_scale="auto"``
    picks ``g`` on a log grid by maximum evidence.
    """
    p = len(w0)
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y
    w_ls = np.linalg.solve(XtX + 1e-6 * (np.trace(XtX) / p + 1e-12) * np.eye(p), Xty)
    r = y - X @ w_ls
    sigma2 = max(r @ r / max(len(y) - p, 1), 1e-10)
    tau = integrated_autocorr_time(r)
    if prior_scale == "auto":
        ev = [_log_evidence(XtX / tau, Xty / tau, yty / tau, sigma2, w0, g * S0)
              for g in _EVIDENCE_GRID]
        g = float(_EVIDENCE_GRID[int(np.argmax(ev))])
    else:
        g = float(prior_scale)
    S0inv = np.linalg.inv(g * S0)
    s2 = sigma2 * tau
    return np.linalg.solve(XtX / s2 + S0inv, Xty / s2 + S0inv @ w0)


def population_prior(W: np.ndarray, trim: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of per-patient coefficient vectors ``W`` (N, p)
    after dropping the ``trim`` fraction with the largest Mahalanobis
    distance from the median (flat, noisy or mis-placed leads)."""
    med = np.median(W, axis=0)
    C = np.atleast_2d(np.cov(W.T)) + 1e-9 * np.eye(W.shape[1])
    dist = np.einsum("ij,jk,ik->i", W - med, np.linalg.inv(C), W - med)
    Wk = W[dist <= np.quantile(dist, 1.0 - trim)]
    return Wk.mean(axis=0), np.atleast_2d(np.cov(Wk.T)) + 1e-6 * np.eye(W.shape[1])


class _TargetFit:
    """Stage-1 result for one target lead: lead-transform coefficients ``w``,
    wander coefficients ``gamma``, and the lead's own baseline model (segment
    mean ``c`` and values at the observed segment edges)."""

    __slots__ = ("w", "gamma", "c", "edges", "edge_values")

    def __init__(self, w, gamma, c=0.0, edges=(), edge_values=()):
        self.w = w
        self.gamma = gamma
        self.c = c
        self.edges = np.asarray(edges, dtype=int)
        self.edge_values = np.asarray(edge_values, dtype=float)


# --------------------------------------------------------------------------
# method
# --------------------------------------------------------------------------
class PatientRegression(Reconstructor):
    """Patient-specific Bayesian lead-transform regression (see module doc).

    Parameters
    ----------
    lags : sequence of int
        FIR context in samples applied to every predictor lead
        (500 Hz: ``(-16, -8, 0, 8, 16)`` = +-32 ms).
    prior_scale : float or "auto"
        Multiplier ``g`` of the population prior covariance of the lead
        transform; larger = trust the record more. ``"auto"``: per record and
        lead by maximum evidence (empirical Bayes).
    hp_cutoff : float
        High-pass cut-off (Hz) separating cardiac activity from baseline.
    baseline_tau : float
        Correlation time (s) of the Ornstein-Uhlenbeck model of a lead's own
        baseline wander; 0 = constant segment mean.
    wander_regression : bool
        Also explain the missing lead's baseline by the rhythm strips' slow
        components (shared electrodes), with a population prior.
    wander_prior_scale : float or "auto"
        Prior scale for the wander coefficients. A fixed, fairly strong prior
        works better than ``"auto"``: the slow residual has only a handful of
        effective samples per segment, so the evidence over-trusts the record.
    residual_correction : bool
        Enable stage 2 (population model of the non-dipolar residual). Off by
        default: on PTB-XL validation it changed mean r by < 0.001 (the
        unexplained part of a missing lead is dominated by baseline wander,
        not by high-frequency non-dipolar content) while making ``fit``
        several times slower.
    residual_ridge : float
        Relative ridge for the stage-2 population regression.
    fs : float
        Sampling rate (Hz).
    """

    name = "patient_regression"

    def __init__(
        self,
        lags: Sequence[int] = (-16, -8, 0, 8, 16),
        prior_scale: float | str = "auto",
        hp_cutoff: float = 0.5,
        baseline_tau: float = 1.5,
        wander_regression: bool = True,
        wander_prior_scale: float | str = 0.3,
        residual_correction: bool = False,
        residual_ridge: float = 1e-3,
        fs: float = 500.0,
    ):
        self.lags = tuple(int(v) for v in lags)
        self.prior_scale = prior_scale
        self.hp_cutoff = float(hp_cutoff)
        self.baseline_tau = float(baseline_tau)
        self.wander_regression = bool(wander_regression)
        self.wander_prior_scale = wander_prior_scale
        self.residual_correction = bool(residual_correction)
        self.residual_ridge = float(residual_ridge)
        self.fs = float(fs)
        self.sos = butter(2, self.hp_cutoff, btype="highpass", fs=self.fs, output="sos")
        #: lead-transform priors {(target, predictors): (w0, S0)}
        self.priors: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
        #: wander priors {(target, predictors): (gamma0, G0)}
        self.wander_priors: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
        #: stage-2 coefficients {(target, extra leads): beta}
        self.betas: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    # ---------------------------------------------------------------- priors
    def _prior(self, j: int, P: tuple[int, ...]):
        """Population prior, or (no training data) the Dower-implied
        instantaneous transform at lag 0 with a broad isotropic covariance."""
        if (j, P) in self.priors:
            return self.priors[(j, P)]
        L = lead_matrix(INDEPENDENT_LEADS)
        a = L[j] @ np.linalg.pinv(L[list(P)])
        w0 = np.zeros(len(P) * len(self.lags))
        if 0 in self.lags:
            w0[np.arange(len(P)) * len(self.lags) + self.lags.index(0)] = a
        return w0, 0.25 * np.eye(len(w0))

    def _wander_prior(self, j: int, P: tuple[int, ...]):
        if (j, P) in self.wander_priors:
            return self.wander_priors[(j, P)]
        return np.zeros(len(P)), 0.01 * np.eye(len(P))

    def fit(self, train_ecgs: np.ndarray | None = None) -> "PatientRegression":
        """Learn the population quantities for the paper layout.

        1. Priors: for every target lead and training record, the full-record
           (10 s) least-squares lead transform (and wander coefficients) are
           computed with the inference features; the prior is their trimmed
           mean / covariance over patients.
        2. Stage 2 (``residual_correction``): stage 1 is run on the *masked*
           training records and ``beta`` is fitted by pooled ridge regression
           of the target's residual on the co-observed leads' residuals.
        """
        if train_ecgs is None:
            return self
        X12 = np.asarray(train_ecgs, dtype=np.float64)
        n = X12.shape[1]
        _, m8 = independent_observations(np.zeros((n, 12)), paper_mask(n, n // 4))
        P = tuple(np.flatnonzero(m8.all(axis=0)).tolist())
        targets = [j for j in range(_N_IND) if j not in P]
        x8s = independent_from_12(X12)
        x8s = x8s[np.all(x8s.std(axis=1) >= 1e-3, axis=1)]   # drop flat leads

        self._fit_priors(x8s, P, targets)
        if self.residual_correction:
            self._fit_residual_model(x8s, m8, P)
        return self

    def _fit_priors(self, x8s, P, targets):
        n = x8s.shape[1]
        rows = np.arange(n)
        Q = nuisance_basis([(0, n)], rows, self.fs, self.hp_cutoff)
        full = np.ones((n, _N_IND), bool)
        W = {j: [] for j in targets}
        G = {j: [] for j in targets}
        for x8 in x8s:
            xhp = highpass_observed(x8, full, self.sos)
            F = lagged_features(xhp[:, list(P)], self.lags)
            F = F - Q @ (Q.T @ F)
            FtF = F.T @ F + 1e-8 * np.eye(F.shape[1])
            S = x8[:, list(P)] - xhp[:, list(P)]
            S = S - S.mean(axis=0)
            StS = S.T @ S + 1e-8 * np.eye(len(P))
            for j in targets:
                y = x8[:, j] - Q @ (Q.T @ x8[:, j])
                W[j].append(np.linalg.solve(FtF, F.T @ y))
                slow = x8[:, j] - xhp[:, j]
                G[j].append(np.linalg.solve(StS, S.T @ (slow - slow.mean())))
        for j in targets:
            self.priors[(j, P)] = population_prior(np.asarray(W[j]))
            self.wander_priors[(j, P)] = population_prior(np.asarray(G[j]))

    def _fit_residual_model(self, x8s, m8, P):
        full = np.ones_like(m8)
        stats: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        for x8 in x8s:
            fits, F, xhp, _ = self._stage1(np.where(m8, x8, np.nan), m8, P)
            truth_hp = highpass_observed(x8, full, self.sos)
            for j, extra, rows, eK in self._stage2_inputs(fits, F, xhp, m8, P):
                Z = np.column_stack([eK, truth_hp[rows, j] - F[rows] @ fits[j].w])
                stats.setdefault((j, extra), np.zeros((Z.shape[1],) * 2))
                stats[(j, extra)] += Z.T @ Z
        for key, Gm in stats.items():
            k = Gm.shape[0] - 1
            A = Gm[:k, :k] + self.residual_ridge * np.trace(Gm[:k, :k]) / k * np.eye(k)
            self.betas[key] = np.linalg.solve(A, Gm[:k, k])

    # --------------------------------------------------------------- stage 1
    def _fit_target(self, j, P, F, S, x8, m8) -> _TargetFit:
        observed = m8[:, j]
        rows = np.flatnonzero(observed)
        w0, S0 = self._prior(j, P)
        g0, G0 = self._wander_prior(j, P)
        if len(rows) < 2 * F.shape[1]:          # (almost) never observed
            c = float(np.median(x8[rows, j])) if len(rows) else 0.0
            return _TargetFit(w0, g0 if self.wander_regression else np.zeros_like(g0), c)
        runs = contiguous_runs(observed)
        X, y = F[rows], x8[rows, j]

        # cardiac part: the lead's slow baseline is projected out exactly
        Q = nuisance_basis(runs, rows, self.fs, self.hp_cutoff)
        w = bayes_regression(X - Q @ (Q.T @ X), y - Q @ (Q.T @ y), w0, S0, self.prior_scale)
        d = y - X @ w                            # baseline + non-dipolar residual

        # baseline shared with the rhythm strips' slow components
        Q0 = nuisance_basis(runs, rows, self.fs, 0.0)     # per-run constants
        if self.wander_regression:
            Sr = S[rows]
            gamma = bayes_regression(Sr - Q0 @ (Q0.T @ Sr), d - Q0 @ (Q0.T @ d),
                                     g0, G0, self.wander_prior_scale)
            d = d - Sr @ gamma
        else:
            gamma = np.zeros(len(P))

        # own wander: slow part of d, summarised by its mean and edge values
        base = Q @ (Q.T @ d)
        k = max(1, int(0.1 * self.fs))           # edge value = mean of 100 ms
        edges, vals = [], []
        for a, b in runs:
            ia, ib = np.searchsorted(rows, [a, b - 1])
            edges += [a, b - 1]
            vals += [base[ia: ia + k].mean(), base[max(ib - k + 1, 0): ib + 1].mean()]
        return _TargetFit(w, gamma, float(base.mean()), edges, vals)

    def _stage1(self, x8, m8, P):
        """Fit all target leads; returns (fits, features, high-passed x8,
        slow components of the predictors)."""
        x0 = np.nan_to_num(x8)
        xhp = highpass_observed(x0, m8, self.sos)
        F = lagged_features(xhp[:, list(P)], self.lags)
        S = x0[:, list(P)] - xhp[:, list(P)]
        fits = {j: self._fit_target(j, P, F, S, x8, m8) for j in range(_N_IND) if j not in P}
        return fits, F, xhp, S

    def _own_baseline(self, fit: _TargetFit, miss: np.ndarray) -> np.ndarray:
        """Ornstein-Uhlenbeck conditional mean of the lead's own baseline:
        ``c + (b_edge - c) exp(-|t - t_edge| / tau)`` from the nearest edge."""
        if self.baseline_tau <= 0 or len(fit.edges) == 0:
            return np.full(len(miss), fit.c)
        nearest = np.abs(miss[:, None] - fit.edges[None, :]).argmin(axis=1)
        dist = np.abs(miss - fit.edges[nearest]) / self.fs
        return fit.c + (fit.edge_values[nearest] - fit.c) * np.exp(-dist / self.baseline_tau)

    # --------------------------------------------------------------- stage 2
    def _stage2_inputs(self, fits, F, xhp, m8, P):
        """Yield ``(target, extra_leads, rows, residuals_of_extra_leads)`` for
        every missing-sample pattern of every target that has co-observed
        non-predictor leads."""
        patterns, inverse = np.unique(m8, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for pidx, pat in enumerate(patterns):
            rows = np.flatnonzero(inverse == pidx)
            extra = tuple(k for k in np.flatnonzero(pat).tolist() if k not in P and k in fits)
            if not extra:
                continue
            eK = np.column_stack([xhp[rows, k] - F[rows] @ fits[k].w for k in extra])
            for j in np.flatnonzero(~pat).tolist():
                if j in fits:
                    yield j, extra, rows, eK

    # --------------------------------------------------------------- predict
    def predict_independent(self, x8: np.ndarray, m8: np.ndarray) -> np.ndarray:
        return self.predict_with_state(x8, m8)[0]

    def predict_with_state(self, x8: np.ndarray, m8: np.ndarray) -> tuple[np.ndarray, dict | None]:
        """As :meth:`predict_independent`, but also return the stage-1 state
        ``(est, state)``; ``state`` is None when no lead is (almost) fully
        observed, else a dict with ``P`` (predictor leads: observed on at
        least 99% of the samples; their short gaps are linearly interpolated
        for the features), ``fits`` ({target: :class:`_TargetFit`}), ``F``
        (lagged high-passed predictor features), ``xhp`` (high-passed
        observed leads) and ``S`` (predictors' slow components). Used by
        methods that build on this one."""
        x8 = np.asarray(x8, dtype=np.float64)
        m8 = np.asarray(m8, dtype=bool)
        P = tuple(np.flatnonzero(m8.mean(axis=0) >= _PREDICTOR_COVERAGE).tolist())
        gaps = [k for k in P if not m8[:, k].all()]
        if gaps:                        # short gaps in a predictor lead
            x8, m8 = x8.copy(), m8.copy()
            for k in gaps:
                x8[:, k] = fill_gaps(x8[:, k], m8[:, k])
                m8[:, k] = True
        est = np.where(m8, np.nan_to_num(x8), 0.0)
        if not P:                       # nothing to regress on: per-lead median
            for j in range(_N_IND):
                if m8[:, j].any():
                    est[~m8[:, j], j] = np.median(x8[m8[:, j], j])
            return est, None

        fits, F, xhp, S = self._stage1(x8, m8, P)
        for j, fit in fits.items():
            miss = np.flatnonzero(~m8[:, j])
            if len(miss):
                est[miss, j] = (F[miss] @ fit.w + S[miss] @ fit.gamma
                                + self._own_baseline(fit, miss))

        if self.residual_correction and self.betas:
            for j, extra, rows, eK in self._stage2_inputs(fits, F, xhp, m8, P):
                beta = self.betas.get((j, extra))
                if beta is not None:
                    est[rows, j] += eK @ beta
        state = {"P": P, "fits": fits, "F": F, "xhp": xhp, "S": S}
        return np.nan_to_num(est, nan=0.0, posinf=0.0, neginf=0.0), state
