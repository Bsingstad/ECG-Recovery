"""Physiologically consistent propagation of edits in a 12-lead ECG.

An edit ``d(t)`` in leads ``J`` is interpreted as a change in the cardiac
source: we look for the smallest change ``dh`` of the heart vector that
produces the edit::

    min  dh^T S^-1 dh   subject to   L[J] dh = d
    =>   dh = S L[J]^T (L[J] S L[J]^T)^+ d

(Gaussian conditioning / weighted minimum-norm). The change in all 12 leads is
``L @ dh``; the original signal and its non-dipolar residual are kept. Any part
of the edit that a dipole cannot explain (only possible when more than three
independent leads are edited) is added locally to the edited leads without
breaking Einthoven/Goldberger.

Input: ``ecg`` of shape (n_samples, 12) in the standard order
I, II, III, aVR, aVL, aVF, V1-V6. Linear and unit independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .leads import LEADS, LIMB_FROM_I_II, LIMB_LEADS, ecg_to_vcg, lead_matrix, normalize_lead_name


def _check(ecg: np.ndarray) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 2 or ecg.shape[1] != 12:
        raise ValueError(f"Expected shape (n_samples, 12), got {ecg.shape}")
    return ecg


@dataclass
class EditResult:
    ecg: np.ndarray            # edited ECG, (n, 12)
    delta_ecg: np.ndarray      # change in all leads, (n, 12)
    delta_vcg: np.ndarray      # change in the heart vector, (n, 3)
    non_dipolar_rms: float     # RMS of the edit the dipole could not explain
    max_edit_error: float      # max deviation between requested and achieved edit


def prior_covariance(prior, ecg, active) -> np.ndarray:
    if isinstance(prior, np.ndarray):
        S = np.asarray(prior, dtype=float)
        if S.shape != (3, 3):
            raise ValueError("A custom prior must have shape (3, 3)")
    elif prior == "identity":
        return np.eye(3)
    elif prior in ("signal", "local"):
        h = ecg_to_vcg(ecg)
        h = h - np.median(h, axis=0)
        if prior == "local" and active.sum() >= 10:
            h = h[active]
        S = h.T @ h / len(h)
    else:
        raise ValueError("prior must be 'identity', 'signal', 'local' or a 3x3 matrix")
    tr = np.trace(S)
    if tr <= 0:
        return np.eye(3)
    return S / tr * 3.0 + 0.05 * np.eye(3)


def propagate_edit(
    ecg: np.ndarray,
    edits: Mapping[str, np.ndarray],
    *,
    prior: str | np.ndarray = "identity",
    keep_non_dipolar: bool = True,
) -> EditResult:
    """Propagate edits in one or more leads to all 12 leads.

    ``edits`` maps lead name -> full new signal (n_samples,). See module doc.
    ``prior``: 'identity' (minimum dipole change), 'signal' (weighted by the
    record's heart-vector covariance), 'local' (covariance of the edited
    samples only) or a custom 3x3 covariance.
    """
    ecg = _check(ecg)
    n = ecg.shape[0]
    if not edits:
        raise ValueError("edits is empty")
    names = [normalize_lead_name(k) for k in edits]
    J = [LEADS.index(k) for k in names]
    delta = np.column_stack(
        [np.asarray(v, dtype=float).reshape(n) - ecg[:, j] for v, j in zip(edits.values(), J)]
    )
    active = np.abs(delta).max(axis=1) > 0

    L = lead_matrix()
    A = L[J]
    S = prior_covariance(prior, ecg, active)
    G = S @ A.T @ np.linalg.pinv(A @ S @ A.T, rcond=1e-10)
    delta_vcg = delta @ G.T
    delta_ecg = delta_vcg @ L.T

    resid = delta - delta_ecg[:, J]
    non_dipolar_rms = float(np.sqrt(np.mean(resid**2)))

    if keep_non_dipolar and non_dipolar_rms > 0:
        for c, (k, j) in enumerate(zip(names, J)):
            if k not in LIMB_LEADS:
                delta_ecg[:, j] += resid[:, c]
        limb_cols = [c for c, k in enumerate(names) if k in LIMB_LEADS]
        if limb_cols:
            M_J = np.array([LIMB_FROM_I_II[names[c]] for c in limb_cols])
            corr = resid[:, limb_cols] @ np.linalg.pinv(M_J).T
            for k in LIMB_LEADS:
                delta_ecg[:, LEADS.index(k)] += corr @ np.array(LIMB_FROM_I_II[k])

    return EditResult(
        ecg=ecg + delta_ecg,
        delta_ecg=delta_ecg,
        delta_vcg=delta_vcg,
        non_dipolar_rms=non_dipolar_rms,
        max_edit_error=float(np.abs(delta - delta_ecg[:, J]).max()),
    )


def perturb_lead(
    ecg: np.ndarray,
    perturbation: np.ndarray,
    lead: str,
    *,
    prior: str | np.ndarray = "identity",
    return_details: bool = False,
) -> np.ndarray | EditResult:
    """Add ``perturbation`` (n_samples,) to one lead and propagate it to the
    others. The selected lead becomes exactly ``ecg + perturbation``; where the
    perturbation is 0 all leads are unchanged (for non-'local' priors)."""
    ecg = _check(ecg)
    perturbation = np.asarray(perturbation, dtype=float)
    if perturbation.shape != (ecg.shape[0],):
        raise ValueError(f"perturbation must have shape ({ecg.shape[0]},), got {perturbation.shape}")
    name = normalize_lead_name(lead)
    j = LEADS.index(name)
    res = propagate_edit(ecg, {name: ecg[:, j] + perturbation}, prior=prior)
    return res if return_details else res.ecg


def windowed_offset(n: int, start: int, stop: int, amplitude: float, taper: int = 20) -> np.ndarray:
    """A plateau with cosine-tapered edges, e.g. to add ST elevation."""
    w = np.zeros(n)
    w[start:stop] = 1.0
    if taper > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper)))
        w[start:start + taper] = ramp
        w[stop - taper:stop] = ramp[::-1]
    return amplitude * w
