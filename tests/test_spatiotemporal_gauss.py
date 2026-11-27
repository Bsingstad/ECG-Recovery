import numpy as np
import pytest

from ecgrecon.leads import LEADS, lead_matrix, expand_independent, limb_consistency_error
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods.spatiotemporal_gauss import SpatioTemporalGaussian


def dipole_ecg(seed, n=5000, fs=500):
    """Synthetic dipole ECG with a random (patient-specific) lead field."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    h = np.zeros((n, 3))
    rr = rng.uniform(0.7, 1.1)
    for beat in np.arange(0.4, n / fs, rr):
        for dt, width, vec in [(-0.16, 0.025, (0.10, 0.10, 0.00)),
                               (0.00, 0.012, (1.00, 0.90, 0.30)),
                               (0.28, 0.060, (0.30, 0.25, -0.10))]:
            h += np.exp(-0.5 * ((t - beat - dt) / width) ** 2)[:, None] * np.array(vec)
    L = lead_matrix(("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"))
    L = L + 0.25 * rng.standard_normal(L.shape)          # patient-specific lead field
    x8 = h @ L.T + 0.005 * rng.standard_normal((n, 8))
    x8 += 0.05 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))[:, None]   # wander
    return expand_independent(x8)


@pytest.fixture(scope="module")
def train():
    return np.stack([dipole_ecg(s) for s in range(20)]).astype(np.float32)


@pytest.mark.parametrize("kw", [
    {"lags": [-4, -2, 0, 2, 4], "kappa": 100},
    {"lags": [0], "kappa": 100, "estimator": "pairwise"},
    {"lags": [0, 3], "kappa": float("inf"), "hp_cutoff": 0.0},
])
def test_contract(train, kw):
    ecg = dipole_ecg(123)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    m = SpatioTemporalGaussian(**kw).fit(train)
    out = m.reconstruct(obs, mask)
    assert out.shape == ecg.shape and np.isfinite(out).all()
    np.testing.assert_array_equal(out[mask], obs[mask])
    assert limb_consistency_error(out) < 1e-9
    np.testing.assert_array_equal(out, m.reconstruct(obs, mask))      # deterministic


def test_patient_adaptation_beats_population(train):
    """With a patient-specific lead field the EM-adapted model must clearly
    beat the population-shape-only model (kappa = inf)."""
    mask = paper_mask()
    adapted = SpatioTemporalGaussian(lags=[0], kappa=100).fit(train)
    pop = SpatioTemporalGaussian(lags=[0], kappa=float("inf")).fit(train)
    err_a, err_p = [], []
    for seed in (1001, 1002, 1003):
        ecg = dipole_ecg(seed)
        obs = apply_mask(ecg, mask)
        miss = ~mask
        err_a.append(np.mean((adapted.reconstruct(obs, mask) - ecg)[miss] ** 2))
        err_p.append(np.mean((pop.reconstruct(obs, mask) - ecg)[miss] ** 2))
    assert np.mean(err_a) < 0.8 * np.mean(err_p)


def test_nan_and_flat_leads(train):
    ecg = dipole_ecg(7)
    ecg[:, LEADS.index("V5")] = 0.0                       # flat rhythm lead
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    obs[100:200, LEADS.index("V1")] = np.nan              # NaN inside an observed run
    out = SpatioTemporalGaussian(lags=[-2, 0, 2]).fit(train).reconstruct(obs, mask)
    assert np.isfinite(out[~mask]).all()


def test_requires_zero_lag():
    with pytest.raises(ValueError):
        SpatioTemporalGaussian(lags=[-2, 2])
