import numpy as np
import pytest

from ecgrecon.leads import INDEPENDENT_LEADS, LEADS, expand_independent, lead_matrix, limb_consistency_error
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods import REGISTRY, load_method
from ecgrecon.methods.patient_regression import PatientRegression
from ecgrecon.methods.physio import PhysioReconstructor

FS = 500


def gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def synthetic(seed=0, n=5000, local=0.0, burst=0.0, wander=0.0, noise=0.004, perturb=0.25):
    """Dipole ECG with a patient-specific lead field, an optional beat-locked
    non-dipolar notch in V2/V3 120 ms after the QRS, outside the FIR
    context of the core (``local`` mV), optional non-beat-locked
    bursts in V2/V3 (``burst`` mV) and baseline wander."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    h = np.zeros((n, 3))
    r = np.zeros((n, 8))
    rr = 0.75 + 0.15 * rng.random()
    for b in np.arange(0.4, n / FS, rr):
        for dt, sd, vec in [(-0.16, 0.025, (0.10, 0.10, 0.00)),
                            (0.00, 0.012, (1.00, 0.90, 0.30)),
                            (0.03, 0.015, (-0.3, 0.20, -0.5)),
                            (0.28, 0.060, (0.30, 0.25, -0.10))]:
            h += gauss(t, b + dt, sd)[:, None] * np.array(vec)
        r[:, 3] += local * gauss(t, b + 0.12, 0.012)
        r[:, 4] += 0.7 * local * gauss(t, b + 0.12, 0.012)
    if burst:
        for c in rng.uniform(0, n / FS, 25):
            r[:, 3] += burst * rng.standard_normal() * gauss(t, c, 0.02)
            r[:, 4] += burst * rng.standard_normal() * gauss(t, c, 0.02)
    L = lead_matrix(INDEPENDENT_LEADS)
    L = L * (1 + perturb * rng.standard_normal(L.shape))
    x8 = h @ L.T + r
    if wander:
        x8 += wander * np.sin(2 * np.pi * 0.2 * t[:, None] + rng.uniform(0, 2 * np.pi, 8))
    x8 += noise * rng.standard_normal(x8.shape)
    return expand_independent(x8)


@pytest.fixture(scope="module")
def train():
    return np.stack([synthetic(seed=s, local=0.05) for s in range(30)]).astype(np.float32)


@pytest.fixture(scope="module")
def fitted(train):
    return PhysioReconstructor().fit(train)


def test_registry():
    for name in ("physio", "patient_regression", "beat_template", "spatiotemporal_gauss", "patient_fa"):
        assert name in REGISTRY
    assert isinstance(load_method("physio"), PhysioReconstructor)


def test_contract(fitted):
    ecg = synthetic(seed=101, local=0.15, wander=0.1)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    out = fitted.reconstruct(obs, mask)
    assert out.shape == ecg.shape and np.isfinite(out).all()
    np.testing.assert_array_equal(out[mask], ecg[mask])          # observations unchanged
    assert limb_consistency_error(out) < 1e-10                    # Einthoven/Goldberger exact
    np.testing.assert_array_equal(out, fitted.reconstruct(obs, mask))   # deterministic
    # a fresh instance gives the same result
    np.testing.assert_array_equal(out, PhysioReconstructor().fit(
        np.stack([synthetic(seed=s, local=0.05) for s in range(30)]).astype(np.float32)
    ).reconstruct(obs, mask))


def test_unobserved_values_are_ignored(fitted):
    ecg = synthetic(seed=102, local=0.1)
    mask = paper_mask()
    garbage = ecg.copy()
    garbage[~mask] = np.random.default_rng(0).normal(0, 50, (~mask).sum())
    np.testing.assert_array_equal(fitted.reconstruct(apply_mask(ecg, mask), mask),
                                  fitted.reconstruct(garbage, mask))


def test_flat_lead_zero_record_short_record(fitted):
    mask = paper_mask()
    ecg = synthetic(seed=103, local=0.1)
    ecg[:, LEADS.index("V3")] = 0.0                      # disconnected electrode
    out = fitted.reconstruct(apply_mask(ecg, mask), mask)
    assert np.isfinite(out).all()

    out = fitted.reconstruct(apply_mask(np.zeros((5000, 12)), mask), mask)
    assert np.isfinite(out).all() and np.abs(out).max() < 1e-9

    short = synthetic(seed=104, n=1000)
    m_short = paper_mask(1000, 250)
    out = fitted.reconstruct(apply_mask(short, m_short), m_short)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[m_short], short[m_short])

    out = PhysioReconstructor().reconstruct(apply_mask(ecg, mask), mask)   # no training data
    assert np.isfinite(out).all()


def _rmse_missing(out, ecg, mask, leads=("V2", "V3")):
    idx = [LEADS.index(k) for k in leads]
    miss = ~mask[:, idx]
    return float(np.sqrt(np.mean((out[:, idx][miss] - ecg[:, idx][miss]) ** 2)))


def test_beat_locked_residual_is_recovered(train):
    """A beat-locked non-dipolar notch in V2/V3 cannot be produced by any
    linear map of II, V1, V5; the residual transplant recovers it."""
    ecg = synthetic(seed=105, local=0.3)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    m = PhysioReconstructor().fit(train)
    core = PatientRegression().fit(train).reconstruct(obs, mask)
    out = m.reconstruct(obs, mask)
    e_core, e_phys = _rmse_missing(core, ecg, mask), _rmse_missing(out, ecg, mask)
    assert e_phys < 0.6 * e_core, (e_phys, e_core)
    assert m.last_alpha[INDEPENDENT_LEADS.index("V2")] > 0.5
    # without the residual the method equals its core
    np.testing.assert_allclose(PhysioReconstructor(residual=False).fit(train).reconstruct(obs, mask), core)


def test_gate_rejects_non_beat_locked_residual(train):
    """Bursts that are not locked to the beats are not reproducible across
    beats: the CV gate shrinks the transplant so that it does not hurt."""
    ecg = synthetic(seed=106, burst=0.3)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    core = PatientRegression().fit(train).reconstruct(obs, mask)
    gated = PhysioReconstructor().fit(train)
    out = gated.reconstruct(obs, mask)
    ungated = PhysioReconstructor(gate="none").fit(train).reconstruct(obs, mask)
    e_core = _rmse_missing(core, ecg, mask)
    assert _rmse_missing(out, ecg, mask) <= 1.02 * e_core
    assert _rmse_missing(out, ecg, mask) <= _rmse_missing(ungated, ecg, mask) + 1e-12
    assert gated.last_alpha[INDEPENDENT_LEADS.index("V2")] < 0.5
