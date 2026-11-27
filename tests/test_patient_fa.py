import numpy as np

from ecgrecon.leads import LEADS, limb_consistency_error
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods import load_method
from ecgrecon.methods.patient_fa import (
    PatientFactorAnalysis, fa_from_covariance, missing_levels, remove_baseline,
)

from test_core import synthetic_ecg


def _patient_ecg(seed=0):
    """Synthetic dipole ECG with a patient-specific (perturbed) lead field,
    a little non-dipolar noise and a per-lead offset."""
    rng = np.random.default_rng(seed)
    ecg = synthetic_ecg(seed=seed)
    # Perturb chest leads by mixing in rhythm leads (a different torso), keep limb consistency.
    ecg[:, 6:] = ecg[:, 6:] @ (np.eye(6) + 0.15 * rng.standard_normal((6, 6)))
    ecg[:, 6:] += 0.05 * rng.standard_normal(6)
    return ecg


def test_reconstructs_dipole_ecg_and_keeps_contract():
    ecg = _patient_ecg()
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    # The synthetic signal has no wander; the constant-median baseline keeps
    # all cardiac content, so this checks identifiability of the lead field.
    out = PatientFactorAnalysis(baseline="median").fit(None).reconstruct(obs, mask)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[mask], ecg[mask])
    assert limb_consistency_error(out) < 1e-9
    for lead in ("I", "V2", "V3", "V4", "V6"):
        j = LEADS.index(lead)
        mj = ~mask[:, j]
        r = np.corrcoef(out[mj, j], ecg[mj, j])[0, 1]
        assert r > 0.98, (lead, r)
    # Default (moving-median) baseline still gives a close reconstruction.
    out = PatientFactorAnalysis().fit(None).reconstruct(obs, mask)
    for lead in ("I", "V2", "V3", "V4", "V6"):
        j = LEADS.index(lead)
        assert np.corrcoef(out[~mask[:, j], j], ecg[~mask[:, j], j])[0, 1] > 0.9


def test_deterministic_and_population_fit():
    train = np.stack([_patient_ecg(s) for s in range(4)]).astype(np.float32)
    m = load_method("ecgrecon.methods.patient_fa:PatientFactorAnalysis", tau=20.0).fit(train)
    assert m.W_pop.shape == (8, 3) and np.all(m.psi_ratio_pop > 0)
    ecg = _patient_ecg(7)
    mask = paper_mask()
    a = m.reconstruct(apply_mask(ecg, mask), mask)
    b = m.reconstruct(apply_mask(ecg, mask), mask)
    np.testing.assert_array_equal(a, b)


def test_robust_to_flat_lead_and_extreme_settings():
    ecg = _patient_ecg(1)
    ecg[:, LEADS.index("V5")] = 0.0                      # disconnected rhythm lead
    mask = paper_mask()
    for kw in ({}, {"tau": 0.0, "nu": 0.0}, {"n_factors": 4}, {"align": "none"},
               {"baseline": "median", "level_tc": 1.0}, {"baseline": "highpass0.5"}):
        out = PatientFactorAnalysis(**kw).fit(None).reconstruct(apply_mask(ecg, mask), mask)
        assert np.isfinite(out).all(), kw


def test_never_observed_lead_and_full_observation():
    ecg = _patient_ecg(2)
    mask = paper_mask()
    mask[:, LEADS.index("V4")] = False                   # V4 never observed
    out = PatientFactorAnalysis(tau=0.0, nu=0.0).fit(None).reconstruct(apply_mask(ecg, mask), mask)
    assert np.isfinite(out).all()
    full = np.ones_like(mask)
    out = PatientFactorAnalysis().fit(None).reconstruct(ecg, full)
    np.testing.assert_array_equal(out, ecg)


def test_helpers():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 3))
    psi = rng.uniform(0.1, 0.3, 8)
    C = W @ W.T + np.diag(psi)
    W_hat, psi_hat = fa_from_covariance(C, 3)
    np.testing.assert_allclose(W_hat @ W_hat.T + np.diag(psi_hat), C, atol=1e-3)

    x = np.linspace(0, 1, 5000)[:, None] * np.ones((1, 8))
    m = np.ones_like(x, dtype=bool)
    m[:1250, 3] = False
    z = remove_baseline(x, m, "movmedian")
    assert np.abs(z).max() < 0.05 and np.all(z[~m] == 0)
    lev = missing_levels(x, x - z, m, 0.0)
    assert np.allclose(lev[:, 3], np.median(x[m[:, 3], 3]))
