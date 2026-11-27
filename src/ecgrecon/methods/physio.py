"""Physiological reconstruction: patient lead transform + gated beat residual.

Model
-----
Every lead is a projection of the cardiac dipole plus a non-dipolar part::

    x_j(t) = l_j . h(t) + r_j(t) + b_j(t)

(``b_j``: baseline wander). The final method combines the two candidate
methods that model these terms best, with a data-driven gate in between:

1. **Core (dipolar part + wander)** -- the patient-specific Bayesian
   lead-transform regression of :mod:`.patient_regression`: each missing
   independent lead (I, V2, V3, V4, V6) is a patient-specific FIR map of the
   always-observed rhythm leads II, V1, V5, learned in the quarter where the
   lead is observed and shrunk to a population prior, plus its model of the
   lead's baseline (shared wander + Ornstein-Uhlenbeck extrapolation).

2. **Beat-synchronous non-dipolar residual** -- the part of the observed
   quarter that the core cannot explain, ``e_j = x_j^hp - F w_j`` (high-passed
   lead minus core cardiac prediction), contains near-field potentials that
   no linear map of three leads can produce (e.g. the V2/V3 morphology). It
   repeats from beat to beat, so -- as in :mod:`.beat_template` -- it is cut
   around R peaks found on the rhythm leads and transplanted onto matching
   beats of the missing quarters (QRS unchanged, ST-T time-scaled with RR).
   A missing beat only receives residual from observed beats whose
   rhythm-lead morphology correlates at least ``corr_threshold`` with its own
   (per-beat morphology gate: ectopic beats get ectopic templates or none).

3. **Data-driven gate (cross-validated shrinkage)** -- whether the residual
   is worth transplanting is decided per record and lead from the observed
   quarter alone, by leave-one-beat-out prediction: every observed beat's
   residual is predicted from the *other* observed beats (same matching
   rules), giving held-out pairs ``(e, p)``. With template = signal ``s`` +
   averaged beat noise, ``<e, p> ~ |s|^2`` and ``<p, p> ~ |s|^2 + N / m``
   (``m`` beats averaged). This gives moment estimates of the signal and
   noise powers, and the transplant is shrunk by the corresponding
   James-Stein / Wiener-type weight for the number of beats ``m'`` averaged
   in the missing quarters::

       alpha = S / (S + N / m'),   S = <e,p>,  N = m (<p,p> - <e,p>)

   clipped to [0, 1]. The estimates are only approximately unbiased: they
   assume independent beat noise and a constant ``m`` (``m`` is replaced by
   its mean over beats, although ``<p,p>`` involves ``mean(1/m)``), whereas
   neighbouring beats share respiration and leftover wander, and a 2.5 s
   quarter offers only about 2-4 held-out beats. On validation data the gate
   is systematically more conservative than the oracle weight (median 0.64
   vs 0.79); it hardly changes mean scores, and its value is mainly that it
   limits harm on records where the residual does not repeat. A residual
   that is not reproducible from beat to beat (noise, artefacts, irregular
   conduction) gets ``alpha ~ 0`` and the core prediction is returned
   unchanged. If the observed quarter contains no pair of matching beats
   (so nothing can be cross-validated) a neutral ``gate_prior = 0.5`` is
   used.

Validation notes (PTB-XL strat_fold 9, 500 records)
    Error analysis of the two parent methods showed that the beat template
    method lost to the regression core in pooled scores almost entirely
    through one artefact record (ecg_id 5712, 30 mV right-arm electrode pop
    in I/II) and through occasional large *harm* of the ungated residual
    (ectopy, noise; 9% of record-leads got >5% worse in shape SSE). With the
    CV gate, 81% of record-leads improve by >5% over the core and 0.2% get
    worse by >5% (worst record-lead change in shape r: -0.01 vs -0.34
    ungated). The residual hyperparameters (``corr_threshold`` 0.8-0.95,
    ``n_match`` 2-6, ``warp`` 0-1, ``gate_prior`` 0.5-1) change mean r by
    < 0.0005; the core settings are those tuned for
    :class:`.patient_regression.PatientRegression`. All of these settings
    were chosen on the same validation records, so validation scores are
    optimistic; the held-out test split is the unbiased check.

    Compared with :class:`.beat_template.BeatTemplate`, the advantage of this
    method is in the ``shape`` view and in record-median r. Its lead in the
    *pooled* ``missing`` scores rests on the single artefact record 5712:
    without it the two are about equal (see ``scripts/evaluate.py``, which
    reports artefact-excluded scores and paired-bootstrap intervals).

Everything uses only observed samples, is deterministic, and runs per record.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..beats import (
    beat_morphology, correlation_matrix, detect_beats, refine_fiducials, transplant_residual,
)
from ..signal import contiguous_runs, median_baseline
from .base import Reconstructor
from .patient_regression import PatientRegression

# Beat windows (s) used to count source beats for the gate. The source-QRS
# window is the one transplant_residual uses to accept a source beat. The
# target window is a fixed approximation of the transplant's RR-dependent
# window [-min(0.3, 0.4 RR_prev), +min(0.7, 0.6 RR_next)] s; it only decides
# which beats enter the mean count m (or m'), not what is transplanted.
_SRC_QRS_PRE, _SRC_QRS_POST = 0.06, 0.10
_TGT_PRE, _TGT_POST = 0.3, 0.5


class PhysioReconstructor(Reconstructor):
    """Patient lead-transform regression + CV-gated beat-locked residual.

    Parameters
    ----------
    lags, prior_scale, hp_cutoff, baseline_tau, wander_regression, wander_prior_scale
        Core settings, passed to :class:`.patient_regression.PatientRegression`
        (defaults = its tuned values).
    residual : bool
        Add the beat-synchronous non-dipolar residual (step 2).
    gate : {"cv", "none"}
        ``"cv"``: shrink the residual by the leave-one-beat-out James-Stein
        weight (step 3); ``"none"``: full residual.
    corr_threshold : float
        Minimum rhythm-lead morphology correlation between a missing beat and
        a source beat (per-beat gate).
    n_match : int
        At most this many best-matching source beats are averaged.
    warp : float
        Exponent of the RR scaling of the ST-T part of a transplanted window.
    taper : float
        Cosine taper (s) at the window edges.
    detrend : bool
        Remove the residual's own slow trend (two-stage moving median) before
        cutting beats, so that leftover baseline does not enter the templates.
    gate_prior : float
        Residual weight used when the observed quarter offers no
        cross-validation pair (e.g. its beats do not match each other but
        match beats of the missing quarters). Negative CV evidence still
        gives 0.
    gate_pseudo : float
        Pseudo-beats with which the CV weight is shrunk towards
        ``gate_prior`` (0 = pure CV estimate).
    fs : int
        Sampling rate (Hz).
    """

    name = "physio"

    def __init__(
        self,
        lags: Sequence[int] = (-16, -8, 0, 8, 16),
        prior_scale: float | str = "auto",
        hp_cutoff: float = 0.5,
        baseline_tau: float = 1.5,
        wander_regression: bool = True,
        wander_prior_scale: float | str = 0.3,
        residual: bool = True,
        gate: str = "cv",
        corr_threshold: float = 0.9,
        n_match: int = 3,
        warp: float = 0.5,
        taper: float = 0.04,
        detrend: bool = True,
        gate_prior: float = 0.5,
        gate_pseudo: float = 0.0,
        fs: int = 500,
    ):
        if gate not in ("cv", "none"):
            raise ValueError("gate must be 'cv' or 'none'")
        self.core = PatientRegression(
            lags=lags, prior_scale=prior_scale, hp_cutoff=hp_cutoff,
            baseline_tau=baseline_tau, wander_regression=wander_regression,
            wander_prior_scale=wander_prior_scale, residual_correction=False, fs=fs,
        )
        self.residual = bool(residual)
        self.gate = gate
        self.corr_threshold = float(corr_threshold)
        self.n_match = int(n_match)
        self.warp = float(warp)
        self.taper = float(taper)
        self.detrend = bool(detrend)
        self.gate_prior = float(gate_prior)
        self.gate_pseudo = float(gate_pseudo)
        self.fs = int(fs)
        #: per-lead gate weights of the last record (diagnostics)
        self.last_alpha: dict[int, float] = {}

    def fit(self, train_ecgs: np.ndarray | None = None) -> "PhysioReconstructor":
        """Population prior of the core lead transform (the residual and the
        gate are learned per record and need no training data)."""
        self.core.fit(train_ecgs)
        return self

    # ------------------------------------------------------------- beats
    def _beats(self, sig: np.ndarray):
        """Fiducials, RR intervals and the beat-by-beat rhythm-lead
        morphology correlation (``sig``: high-passed rhythm leads)."""
        fid = refine_fiducials(sig, detect_beats(sig, self.fs), self.fs)
        if len(fid) < 3:
            return None
        rr = np.diff(fid).astype(float)
        med = float(np.median(rr))
        rr_pair = (np.concatenate([[med], rr]), np.concatenate([rr, [med]]))
        corr = correlation_matrix(beat_morphology(sig, fid, self.fs))
        return fid, rr_pair, corr

    def _transplant(self, e, rows, beats):
        fid, rr, corr = beats
        return transplant_residual(e, rows, fid, rr, corr, fs=self.fs,
                                   corr_threshold=self.corr_threshold, n_match=self.n_match,
                                   warp=self.warp, taper=self.taper)

    def _n_sources(self, beats, avail: np.ndarray, rows: np.ndarray) -> float:
        """Mean number of source beats averaged for the beats touching
        ``rows`` (sources: QRS inside ``avail``; the beat itself excluded)."""
        fid, _, corr = beats
        n = len(avail)
        off = np.arange(-int(_SRC_QRS_PRE * self.fs), int(_SRC_QRS_POST * self.fs))
        src = np.array([f + off[0] >= 0 and f + off[-1] < n and avail[f + off].all() for f in fid])
        counts = []
        for k, f in enumerate(fid):
            lo, hi = max(f - int(_TGT_PRE * self.fs), 0), min(f + int(_TGT_POST * self.fs), n)
            if not rows[lo:hi].any():
                continue
            ok = src & (corr[k] >= self.corr_threshold)
            ok[k] = False
            if ok.any():
                counts.append(min(int(ok.sum()), self.n_match))
        return float(np.mean(counts)) if counts else 0.0

    def _gate(self, e, obs, miss, beats) -> float:
        """Leave-one-beat-out James-Stein weight of the transplant (step 3),
        shrunk towards ``gate_prior`` with ``gate_pseudo`` pseudo-beats."""
        if self.gate == "none":
            return 1.0
        avail = np.isfinite(e)
        m_out = self._n_sources(beats, avail, miss)
        if m_out <= 0:
            return 0.0
        p = self._transplant(e, obs, beats)          # each beat from the others
        v = obs & (p != 0) & avail
        fid = beats[0]
        n_cv = int(sum(1 for f in fid if f < len(p) and v[f]))
        alpha_cv = 0.0
        if v.sum() >= self.fs // 5 and n_cv > 0:
            ev, pv = e[v], p[v]
            S, pp = float(ev @ pv), float(pv @ pv)
            m_cv = self._n_sources(beats, avail, obs)
            if S > 0 and pp > 0 and m_cv > 0:
                N = max(m_cv * (pp - S), 0.0)
                alpha_cv = float(np.clip(S / (S + N / m_out), 0.0, 1.0))
        else:
            n_cv = 0
        if n_cv == 0:
            return self.gate_prior
        k0 = self.gate_pseudo
        return (n_cv * alpha_cv + k0 * self.gate_prior) / (n_cv + k0)

    # ----------------------------------------------------------- predict
    def predict_independent(self, x8: np.ndarray, m8: np.ndarray) -> np.ndarray:
        est, st = self.core.predict_with_state(x8, m8)
        self.last_alpha = {}
        if not self.residual or st is None:
            return est
        m8 = np.asarray(m8, dtype=bool)
        P, fits, F, xhp = st["P"], st["fits"], st["F"], st["xhp"]
        sig = xhp[:, list(P)]
        if not np.isfinite(sig).all() or sig.shape[0] < 2 * self.fs:
            return est
        beats = self._beats(sig)
        if beats is None:
            return est
        n = len(est)
        for j, fit in fits.items():
            obs = m8[:, j]
            miss = ~obs
            if obs.sum() < self.fs or not miss.any():
                continue
            e = np.full(n, np.nan)
            e[obs] = xhp[obs, j] - F[obs] @ fit.w
            if self.detrend:
                for a, b in contiguous_runs(obs):
                    if b - a > self.fs // 2:
                        e[a:b] -= median_baseline(e[a:b], self.fs)
            if not np.isfinite(e[obs]).all():
                continue
            alpha = self._gate(e, obs, miss, beats)
            self.last_alpha[j] = alpha
            if alpha <= 0:
                continue
            r = self._transplant(e, miss, beats)
            est[miss, j] += alpha * r[miss]
        return est
