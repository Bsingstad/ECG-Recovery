"""Lead definitions and the physiological (dipole) lead model.

Two facts about the 12-lead ECG are used throughout the package:

1. The six limb leads have only two degrees of freedom (Einthoven/Goldberger)::

       III = II - I,  aVR = -(I + II)/2,  aVL = I - II/2,  aVF = II - I/2

   These relations are exact; they follow from how the leads are defined.

2. The heart's electrical activity is approximated by a time-varying dipole
   ``h(t) = [X, Y, Z]`` (Frank coordinates: X right->left, Y head->foot,
   Z front->back). Every lead is a projection of ``h`` on a lead vector::

       ecg(t) = L @ h(t) + r(t)

   ``L`` (12x3) is built from the Dower matrix (Dower et al., 1980) for
   I, II, V1-V6 and extended with the relations in (1). ``r(t)`` is the
   non-dipolar residual (local potentials, noise).

The 8 *independent* leads are I, II, V1-V6; the other four follow from (1).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
LIMB_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")
CHEST_LEADS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_LEADS: tuple[str, ...] = ("I", "II") + CHEST_LEADS

#: Indices of the independent leads in the standard 12-lead order.
INDEPENDENT_IDX: tuple[int, ...] = tuple(LEADS.index(k) for k in INDEPENDENT_LEADS)

#: Dower transform: lead = coefficients @ [X, Y, Z]
DOWER: dict[str, tuple[float, float, float]] = {
    "I":  (0.632, -0.235,  0.059),
    "II": (0.235,  1.066, -0.132),
    "V1": (-0.515, 0.157, -0.917),
    "V2": (0.044,  0.164, -1.387),
    "V3": (0.882,  0.098, -1.277),
    "V4": (1.213,  0.127, -0.601),
    "V5": (1.125,  0.127, -0.086),
    "V6": (0.831,  0.076,  0.230),
}

#: Limb leads expressed through (I, II).
LIMB_FROM_I_II: dict[str, tuple[float, float]] = {
    "I":   (1.0, 0.0),
    "II":  (0.0, 1.0),
    "III": (-1.0, 1.0),
    "aVR": (-0.5, -0.5),
    "aVL": (1.0, -0.5),
    "aVF": (-0.5, 1.0),
}


def expansion_matrix() -> np.ndarray:
    """``E`` (12x8) such that ``ecg12 = x8 @ E.T`` for independent leads ``x8``
    ordered as :data:`INDEPENDENT_LEADS`."""
    E = np.zeros((12, 8))
    for k, (a, b) in LIMB_FROM_I_II.items():
        E[LEADS.index(k), 0] = a
        E[LEADS.index(k), 1] = b
    for c, k in enumerate(CHEST_LEADS):
        E[LEADS.index(k), 2 + c] = 1.0
    return E


EXPANSION: np.ndarray = expansion_matrix()


def expand_independent(x8: np.ndarray) -> np.ndarray:
    """Map independent leads ``(..., 8)`` to all 12 leads ``(..., 12)``.
    The limb leads of the result satisfy Einthoven/Goldberger exactly."""
    return np.asarray(x8) @ EXPANSION.T


def independent_from_12(ecg: np.ndarray) -> np.ndarray:
    """Pick the independent leads ``(..., 8)`` from a 12-lead array."""
    return np.asarray(ecg)[..., list(INDEPENDENT_IDX)]


def lead_matrix(lead_order: Sequence[str] = LEADS) -> np.ndarray:
    """Lead-field matrix ``L`` (len(lead_order) x 3): ``ecg = h @ L.T`` for a
    pure dipole source, built from the Dower coefficients."""
    d_i, d_ii = np.array(DOWER["I"]), np.array(DOWER["II"])
    rows = []
    for name in lead_order:
        if name in LIMB_FROM_I_II:
            a, b = LIMB_FROM_I_II[name]
            rows.append(a * d_i + b * d_ii)
        else:
            rows.append(np.array(DOWER[name]))
    return np.vstack(rows)


def ecg_to_vcg(ecg: np.ndarray) -> np.ndarray:
    """Least-squares (inverse Dower) estimate of the heart vector ``(n, 3)``
    from the eight independent leads of a 12-lead ECG ``(n, 12)``."""
    L8 = lead_matrix(INDEPENDENT_LEADS)
    return independent_from_12(ecg) @ np.linalg.pinv(L8).T


def limb_consistency_error(ecg: np.ndarray) -> float:
    """Largest deviation from the Einthoven/Goldberger relations (0 = exact)."""
    ecg = np.asarray(ecg, dtype=float)
    i, ii = ecg[..., 0], ecg[..., 1]
    err = 0.0
    for name, (a, b) in LIMB_FROM_I_II.items():
        dev = ecg[..., LEADS.index(name)] - (a * i + b * ii)
        err = max(err, float(np.nanmax(np.abs(dev))))
    return err


def normalize_lead_name(lead: str) -> str:
    """Accepts e.g. 'II', 'Lead II', 'lead avl', 'AVL', 'v2'."""
    key = str(lead).strip().lower().replace("lead", "").replace(" ", "").replace("_", "")
    for name in LEADS:
        if name.lower() == key:
            return name
    raise ValueError(f"Unknown lead {lead!r}. Valid names: {LEADS}")
