"""Patient-specific lead transform + beat-synchronous non-dipolar residual.

Physiological model
-------------------
Every lead is a projection of the cardiac dipole plus a non-dipolar residual::

    x_j(t) = l_j . h(t) + r_j(t)

The rhythm strips (II, V1, V5) are observed for the full 10 s and roughly span
the three dipole axes (Y, Z, X). Hence each missing lead is, to first order, a
*linear combination of the rhythm leads*. The coefficients depend on the
patient (heart orientation, torso geometry, electrode placement), but on the
printed layout every missing lead is observed in some quarter *together with*
the rhythm leads, so the record itself tells us its own lead transform.

Step 1 - patient-specific lead transform (Bayesian ridge)
    For a missing lead ``j`` with predictors ``C`` (leads known both where
    ``j`` is missing and where ``j`` is observed; II, V1, V5 here) we fit::

        y = X w + e,   e ~ N(0, s2),   w ~ N(w0, Sw)

    on the samples where ``j`` is observed. ``w0`` and ``Sw`` are the mean and
    covariance of the per-record least-squares coefficients across the training
    population (learned in :meth:`BeatTemplate.fit`), i.e. the population
    distribution of lead transforms. ECG samples are strongly autocorrelated,
    so the data term is down-weighted by an effective sample count
    ``duration / corr_time``. The posterior mean is::

        w = (X'X / (s2 n_c) + Sw^-1)^-1 (X'y / (s2 n_c) + Sw^-1 w0)

    with ``n_c = corr_time * fs``. Small ``corr_time`` trusts the record,
    large ``corr_time`` falls back to the population transform.

Step 2 - beat-locked residual templates
    The part of ``x_j`` that the linear map cannot explain (local, near-field
    potentials, e.g. the V2/V3 morphology, and the patient's own T-wave
    shape) repeats from beat to beat. In the quarter where ``j`` is observed we
    form ``r_j = x_j - X w``, cut it around each beat fiducial and add it to
    the beats of the missing quarters. Beats are detected on the full-length
    rhythm leads; a target beat only receives a template from observed beats
    whose rhythm-lead morphology is similar (Pearson correlation above
    ``corr_threshold``), so ectopic beats are matched to ectopic beats or left
    with the step-1 estimate. The QRS part of the window is used as is, the
    post-QRS (ST-T) part is time-scaled with ``(RR_source/RR_target)^warp``
    (QT shortens at higher heart rate). The stretch is anchored at the QRS
    end (fiducial + 100 ms), so it is an RR-dependent ST-T stretch rather than
    an exact QT correction (``warp=0.5`` is not equivalent to Bazett's
    formula, which scales from QRS onset). Window edges are tapered to avoid
    discontinuities.

Baseline wander
    All fitting is done on baseline-corrected signals (two-stage moving median,
    200 ms then 600 ms, computed separately on each contiguous observed run,
    since stitched segments can have different offsets). The low-frequency
    level of a missing lead is its median observed baseline plus a strongly
    shrunk ridge map of the rhythm leads' baselines (respiration and torso
    motion move several electrodes together), and near quarter boundaries it
    is blended towards the observed neighbouring baseline with an exponential
    decay (``edge_tau``).

Validation notes (PTB-XL strat_fold 9, 500 records)
    With step 2 enabled, the patient-specific transform and the population
    transform give similar results on artifact-free records, because the
    beat-locked residual absorbs most transform error; without step 2 the
    patient-specific transform is clearly better. ``corr_time = 1 s`` (about
    2-3 effective samples per 2.5 s quarter, i.e. a fairly strong prior) was
    chosen on the plateau of the artifact-free scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..beats import (  # noqa: F401  (re-exported for backwards compatibility)
    beat_morphology, correlation_matrix, detect_beats, refine_fiducials, transplant_residual,
)
from ..leads import independent_from_12
from ..signal import FS_DEFAULT, contiguous_runs, median_baseline  # noqa: F401
from .base import Reconstructor


# --------------------------------------------------------------------------
# signal utilities
# --------------------------------------------------------------------------
def remove_baseline(x8: np.ndarray, m8: np.ndarray, fs: int = FS_DEFAULT):
    """Baseline-correct each observed run of each lead separately.

    Returns ``(hp, base)``: corrected signal and baseline, NaN where ``~m8``.
    Runs are treated separately because the paper layout stitches segments
    that may have different offsets.
    """
    hp = np.full_like(x8, np.nan, dtype=float)
    base = np.full_like(x8, np.nan, dtype=float)
    for j in range(x8.shape[1]):
        for a, b in contiguous_runs(m8[:, j]):
            seg = np.nan_to_num(x8[a:b, j])
            bl = median_baseline(seg, fs)
            base[a:b, j] = bl
            hp[a:b, j] = seg - bl
    return hp, base


# --------------------------------------------------------------------------
# population prior on lead transforms
# --------------------------------------------------------------------------
@dataclass
class TransformPrior:
    mean: np.ndarray       # (p,)
    cov: np.ndarray        # (p, p)


def coefficient_prior(covs: np.ndarray, target: int, preds: tuple[int, ...],
                      eps: float = 1e-8) -> TransformPrior:
    """Population distribution of per-record regression coefficients of lead
    ``target`` on leads ``preds``, from per-record covariances ``(N, 8, 8)``.

    Robust: median as the prior mean, covariance over records whose
    coefficients lie within 5 MADs of the median.
    """
    p = list(preds)
    Scc = covs[:, p][:, :, p]                                       # (N, p, p)
    Scj = covs[:, p, target]                                        # (N, p)
    tr = np.trace(Scc, axis1=1, axis2=2) / len(p)
    ok = tr > 0
    reg = Scc[ok] + eps * tr[ok, None, None] * np.eye(len(p))
    W = np.linalg.solve(reg, Scj[ok][..., None])[..., 0]
    med = np.median(W, axis=0)
    mad = 1.4826 * np.median(np.abs(W - med), axis=0) + 1e-9
    keep = (np.abs(W - med) <= 5 * mad).all(axis=1)
    Wk = W[keep] if keep.sum() > len(p) + 2 else W
    cov = np.cov(Wk.T).reshape(len(p), len(p))
    cov += 1e-6 * np.eye(len(p)) * max(np.trace(cov) / len(p), 1e-6)
    return TransformPrior(med, cov)


# --------------------------------------------------------------------------
# method
# --------------------------------------------------------------------------
class BeatTemplate(Reconstructor):
    """Patient-specific lead transform (step 1) + beat-locked residual (step 2).

    Parameters
    ----------
    corr_time : float
        Residual decorrelation time in seconds; the record's data term counts
        ``duration / corr_time`` effective samples against the population prior
        on the lead transform. Larger = closer to the population transform.
    residual : bool
        Enable step 2 (beat-synchronous residual templates).
    corr_threshold : float
        Minimum rhythm-lead morphology correlation for a source beat to lend
        its residual to a target beat.
    n_match : int
        Use at most this many best-matching source beats (averaged).
    residual_gain : float
        Shrinkage of the transferred residual (1 = full template).
    warp : float
        Exponent of the RR scaling of the post-QRS window part (stretch
        anchored at the QRS end; see module docstring).
    taper : float
        Length (s) of the cosine taper at each window edge.
    edge_tau : float
        Decay time (s) of the continuity correction at boundaries to observed
        samples of the same lead (0 disables).
    base_ridge : float
        Shrinkage of the baseline-wander regression (relative ridge; large
        values give a constant level per lead).
    fs : int
        Sampling rate (Hz).
    """

    name = "beat_template"

    def __init__(self, corr_time: float = 1.0, residual: bool = True,
                 corr_threshold: float = 0.9, n_match: int = 3,
                 residual_gain: float = 1.0, warp: float = 0.5,
                 taper: float = 0.04, edge_tau: float = 1.0,
                 base_ridge: float = 1.0, fs: int = FS_DEFAULT):
        self.corr_time = float(corr_time)
        self.residual = bool(residual)
        self.corr_threshold = float(corr_threshold)
        self.n_match = int(n_match)
        self.residual_gain = float(residual_gain)
        self.warp = float(warp)
        self.taper = float(taper)
        self.edge_tau = float(edge_tau)
        self.base_ridge = float(base_ridge)
        self.fs = int(fs)
        self.covs: np.ndarray | None = None
        self._priors: dict[tuple[int, tuple[int, ...]], TransformPrior] = {}

    # ---------------------------------------------------------------- fit
    def fit(self, train_ecgs: np.ndarray | None = None) -> "BeatTemplate":
        """Per-record covariances of the baseline-corrected independent leads
        (used to build the population prior on lead transforms on demand)."""
        if train_ecgs is None:
            self.covs = None
            return self
        covs = []
        for ecg in np.asarray(train_ecgs):
            x = independent_from_12(np.asarray(ecg, dtype=np.float64))
            if not np.isfinite(x).all():
                continue
            x = x - median_baseline(x, self.fs)
            covs.append(x.T @ x / len(x))
        self.covs = np.asarray(covs)
        self._priors = {}
        return self

    def _prior(self, target: int, preds: tuple[int, ...]) -> TransformPrior | None:
        if self.covs is None or len(self.covs) == 0:
            return None
        key = (target, preds)
        if key not in self._priors:
            self._priors[key] = coefficient_prior(self.covs, target, preds)
        return self._priors[key]

    # ------------------------------------------------------- step 1: map
    def _fit_transform(self, X: np.ndarray, y: np.ndarray, prior: TransformPrior | None) -> np.ndarray:
        p = X.shape[1]
        XtX, Xty = X.T @ X, X.T @ y
        scale = np.trace(XtX) / p + 1e-12
        w_ols = np.linalg.solve(XtX + 1e-6 * scale * np.eye(p), Xty)
        s2 = max(float(np.mean((y - X @ w_ols) ** 2)), 1e-6 * float(np.mean(y ** 2)) + 1e-10)
        n_c = max(self.corr_time * self.fs, 1.0)
        if prior is None:  # weak isotropic shrinkage towards zero
            P0, w0 = 1e-3 * scale / (s2 * n_c) * np.eye(p), np.zeros(p)
        else:
            P0, w0 = np.linalg.inv(prior.cov), prior.mean
        A = XtX / (s2 * n_c) + P0
        return np.linalg.solve(A, Xty / (s2 * n_c) + P0 @ w0)

    def _fit_baseline(self, B: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Ridge regression of the target's baseline on the predictors'
        baselines (respiration and torso motion move all electrodes; part of
        the wander is shared). ``base_ridge`` is relative to the mean
        eigenvalue of ``B'B``, so 1 means strong shrinkage towards zero."""
        G = B.T @ B
        lam = self.base_ridge * np.trace(G) / len(G) + 1e-12
        return np.linalg.solve(G + lam * np.eye(len(G)), B.T @ yb)

    def _population_regression(self, hp, rows, j, known) -> np.ndarray:
        """Fallback: conditional mean of lead ``j`` given ``known`` leads under
        the pooled population covariance (zero if no training data)."""
        if self.covs is None or not known:
            return np.zeros(len(rows))
        S = self.covs.mean(axis=0)
        k = list(known)
        Skk = S[np.ix_(k, k)]
        w = np.linalg.solve(Skk + 1e-6 * np.trace(Skk) / len(k) * np.eye(len(k)), S[k, j])
        return hp[rows][:, k] @ w

    # ---------------------------------------------------- step 2: beats
    def _beats(self, hp: np.ndarray, full: np.ndarray):
        sig = hp[:, full]
        fid = detect_beats(sig, self.fs)
        fid = refine_fiducials(sig, fid, self.fs)
        if len(fid) == 0:
            return fid, None, None
        rr = np.diff(fid)
        med_rr = float(np.median(rr)) if len(rr) else float(self.fs)
        rr_prev = np.concatenate([[med_rr], rr]).astype(float)
        rr_next = np.concatenate([rr, [med_rr]]).astype(float)
        corr = correlation_matrix(beat_morphology(sig, fid, self.fs))
        return fid, (rr_prev, rr_next), corr

    def _residual_signal(self, res: np.ndarray, target_rows: np.ndarray, fid, rr, corr) -> np.ndarray:
        """Beat-synchronous residual for the samples ``target_rows`` (bool,
        length n) from the residual ``res`` (NaN where unavailable); see
        :func:`transplant_residual`."""
        return transplant_residual(res, target_rows, fid, rr, corr, fs=self.fs,
                                   corr_threshold=self.corr_threshold, n_match=self.n_match,
                                   warp=self.warp, taper=self.taper, gain=self.residual_gain)

    # ---------------------------------------------------------- predict
    def predict_independent(self, x8: np.ndarray, m8: np.ndarray) -> np.ndarray:
        """Estimate all 8 independent leads ``(n, 8)`` from ``x8`` (NaN where
        ``~m8``). For every missing lead and every distinct pattern of known
        leads on its missing samples: step 1 (lead transform) + baseline model,
        then step 2 (beat-locked residual), then boundary continuity."""
        n, L = x8.shape
        x8 = np.where(m8, x8, np.nan)
        hp, base = remove_baseline(x8, m8, self.fs)
        est = np.where(m8, x8, 0.0)
        full = m8.all(axis=0)

        beats = None
        if self.residual and full.any():
            fid, rr, corr = self._beats(hp, full)
            if len(fid) >= 2:
                beats = (fid, rr, corr)

        for j in range(L):
            miss = ~m8[:, j]
            if not miss.any():
                continue
            obs_j = m8[:, j]
            level = float(np.median(base[obs_j, j])) if obs_j.any() else 0.0
            pred = np.full(n, level)
            fitted = np.full(n, np.nan)      # baseline model where predictors exist
            # group missing samples by which leads are known there
            pats, inv = np.unique(m8[miss], axis=0, return_inverse=True)
            miss_idx = np.flatnonzero(miss)
            for pi, pat in enumerate(pats):
                rows = miss_idx[inv.reshape(-1) == pi]
                known_here = np.flatnonzero(pat)
                # predictors: known on these rows and on (all of) j's observed rows
                C = [int(k) for k in known_here if obs_j.any() and m8[obs_j, k].all()]
                train = obs_j & m8[:, C].all(axis=1) if C else np.zeros(n, bool)
                if not C or train.sum() < self.fs:
                    # nothing to learn from this record: population regression
                    pred[rows] += self._population_regression(hp, rows, j, tuple(known_here))
                    continue
                have = m8[:, C].all(axis=1)
                # step 1: patient-specific lead transform on baseline-free signals
                X, y = hp[train][:, C], hp[train, j]
                w = self._fit_transform(X, y, self._prior(j, tuple(C)))
                # low-frequency level: ridge map of the predictors' baselines
                bc = base[:, C] - np.median(base[train][:, C], axis=0)
                wb = self._fit_baseline(bc[train], base[train, j] - level)
                fitted[have] = level + bc[have] @ wb
                pred[rows] = fitted[rows] + hp[rows][:, C] @ w
                # step 2: beat-locked non-dipolar residual
                if beats is not None:
                    res = np.full(n, np.nan)
                    res[train] = y - X @ w
                    tgt = np.zeros(n, bool)
                    tgt[rows] = True
                    pred[rows] += self._residual_signal(res, tgt, *beats)[rows]
            if self.edge_tau > 0 and obs_j.any():
                pred = self._edge_blend(pred, base[:, j], fitted, obs_j)
            est[miss, j] = pred[miss]
        return np.nan_to_num(est)

    def _edge_blend(self, pred, base_j, fitted, obs):
        """Continuity of the baseline at quarter boundaries.

        At each boundary between observed and missing samples of the lead, the
        misfit between its observed baseline and the modelled baseline
        (``fitted``) over the adjacent 100 ms is added to the missing run with
        an exponential decay of time constant ``edge_tau``: wander is slow, so
        the level just across the boundary is the best local information.
        """
        n = len(pred)
        out = pred.copy()
        win = max(int(0.1 * self.fs), 1)
        tau = self.edge_tau * self.fs
        mis = base_j - fitted
        for a, b in contiguous_runs(~obs):
            t = np.arange(b - a, dtype=float)
            for side, seg, decay in ((a > 0, slice(max(a - win, 0), a), t),
                                     (b < n, slice(b, min(b + win, n)), b - a - 1 - t)):
                if not side:
                    continue
                d = mis[seg]
                d = d[np.isfinite(d)]
                if len(d):
                    out[a:b] += d.mean() * np.exp(-decay / tau)
        return out
