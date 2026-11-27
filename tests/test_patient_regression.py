import numpy as np
import pytest

from ecgrecon.leads import INDEPENDENT_LEADS, expand_independent, lead_matrix, limb_consistency_error
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods.patient_regression import (
    PatientRegression, contiguous_runs, integrated_autocorr_time, nuisance_basis,
)


def dipole_ecg(seed=0, n=5000, fs=500, noise=0.003, wander=0.0, perturb=0.3):
    """Pure-dipole ECG with a *patient-specific* lead field (Dower perturbed),
    optional respiratory-like baseline wander per lead."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    h = np.zeros((n, 3))
    rr = 0.75 + 0.2 * rng.random()
    for beat in np.arange(0.4, n / fs, rr):
        for dt, width, vec in [(-0.16, 0.025, (0.10, 0.10, 0.00)),
                               (0.00, 0.012, (1.00, 0.90, 0.30)),
                               (0.03, 0.015, (-0.3, 0.20, -0.5)),
                               (0.28, 0.060, (0.30, 0.25, -0.10))]:
            h += np.exp(-0.5 * ((t - beat - dt) / width) ** 2)[:, None] * np.array(vec)
    L = lead_matrix(INDEPENDENT_LEADS)
    L = L * (1 + perturb * rng.standard_normal(L.shape))
    x8 = h @ L.T + noise * rng.standard_normal((n, 8))
    if wander:
        phase = rng.uniform(0, 2 * np.pi, 8)
        x8 += wander * np.sin(2 * np.pi * 0.2 * t[:, None] + phase)
    return expand_independent(x8)


@pytest.fixture(scope="module")
def fitted():
    train = np.stack([dipole_ecg(seed=s) for s in range(40)]).astype(np.float32)
    return PatientRegression(residual_correction=True).fit(train)


def test_helpers():
    assert contiguous_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    rows = np.r_[0:100, 300:400]
    Q = nuisance_basis(contiguous_runs(np.isin(np.arange(400), rows)), rows, 500, 0.5)
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
    rng = np.random.default_rng(0)
    assert integrated_autocorr_time(rng.standard_normal(5000)) < 1.5
    smooth = np.convolve(rng.standard_normal(5000), np.ones(50) / 50, mode="same")
    assert integrated_autocorr_time(smooth) > 20


def test_contract_and_accuracy(fitted):
    ecg = dipole_ecg(seed=123, wander=0.05)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    out = fitted.reconstruct(obs, mask)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[mask], ecg[mask])
    assert limb_consistency_error(out) < 1e-10
    miss = ~mask
    for j in range(12):
        if miss[:, j].any():
            r = np.corrcoef(out[miss[:, j], j], ecg[miss[:, j], j])[0, 1]
            assert r > 0.95, (j, r)
    # deterministic
    np.testing.assert_array_equal(out, fitted.reconstruct(obs, mask))


def test_without_fit_uses_dower_prior():
    ecg = dipole_ecg(seed=7)
    mask = paper_mask()
    out = PatientRegression(residual_correction=False).reconstruct(apply_mask(ecg, mask), mask)
    assert np.isfinite(out).all()
    miss = ~mask[:, 7]
    assert np.corrcoef(out[miss, 7], ecg[miss, 7])[0, 1] > 0.9


def test_flat_lead_is_safe(fitted):
    ecg = dipole_ecg(seed=5)
    ecg[:, 8] = 0.0                       # disconnected V3
    mask = paper_mask()
    out = fitted.reconstruct(apply_mask(ecg, mask), mask)
    assert np.isfinite(out).all()
