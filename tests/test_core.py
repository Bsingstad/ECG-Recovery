import numpy as np
import pytest

from ecgrecon.leads import (
    LEADS, expand_independent, independent_from_12, lead_matrix, limb_consistency_error,
)
from ecgrecon.masking import (
    from_unet_input, independent_observations, paper_mask, to_unet_input, apply_mask,
)
from ecgrecon.methods import DowerDipole, PopulationGaussian, load_method
from ecgrecon.metrics import evaluate
from ecgrecon.propagate import perturb_lead, windowed_offset


def synthetic_ecg(n=5000, fs=500, seed=0, noise=0.005):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    h = np.zeros((n, 3))
    for beat in np.arange(0.5, n / fs, 0.9):
        for dt, width, vec in [(-0.16, 0.025, (0.10, 0.10, 0.00)),
                               (0.00, 0.012, (1.00, 0.90, 0.30)),
                               (0.28, 0.060, (0.30, 0.25, -0.10))]:
            h += np.exp(-0.5 * ((t - beat - dt) / width) ** 2)[:, None] * np.array(vec)
    x8 = h @ lead_matrix(("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")).T
    x8 += noise * rng.standard_normal(x8.shape)
    return expand_independent(x8)


def test_expansion_is_einthoven_consistent():
    ecg = synthetic_ecg()
    assert limb_consistency_error(ecg) < 1e-12
    np.testing.assert_allclose(expand_independent(independent_from_12(ecg)), ecg, atol=1e-12)


def test_lead_matrix_limb_rows():
    L = lead_matrix()
    np.testing.assert_allclose(L[LEADS.index("III")], L[1] - L[0])
    np.testing.assert_allclose(L[LEADS.index("aVR")], -(L[0] + L[1]) / 2)


def test_paper_mask_matches_notebook_process_ecg():
    ecg = synthetic_ecg()
    x15 = to_unet_input(ecg)
    obs, mask = from_unet_input(x15)
    np.testing.assert_array_equal(mask, paper_mask())
    np.testing.assert_allclose(obs[mask], ecg[mask])
    # observed counts: 3 groups x 1250 + rhythm strips II, V1, V5 full
    assert mask[:, LEADS.index("II")].all() and mask[:, LEADS.index("V5")].all()
    assert mask[:1250, LEADS.index("I")].all() and not mask[1250:, LEADS.index("I")].any()
    assert mask[3750:, LEADS.index("V6")].all() and not mask[:3750, LEADS.index("V6")].any()


def test_limb_basis_recovered_in_second_quarter():
    ecg = synthetic_ecg()
    mask = paper_mask()
    x8, m8 = independent_observations(apply_mask(ecg, mask), mask)
    assert m8[:2500, 0].all() and not m8[2500:, 0].any()      # lead I known in Q1, Q2
    np.testing.assert_allclose(x8[1250:2500, 0], ecg[1250:2500, 0], atol=1e-12)


@pytest.mark.parametrize("method", [DowerDipole(), PopulationGaussian().fit(np.stack([synthetic_ecg(seed=s) for s in range(3)]))])
def test_reconstruction_contract(method):
    ecg = synthetic_ecg(seed=7)
    mask = paper_mask()
    rec = method.reconstruct(apply_mask(ecg, mask), mask)
    assert rec.shape == ecg.shape and np.isfinite(rec).all()
    np.testing.assert_allclose(rec[mask], ecg[mask])                 # observations kept
    assert limb_consistency_error(rec) < 1e-9                        # Einthoven holds
    m = evaluate(ecg[None], rec[None], mask)
    assert m["missing"]["mean_pcc"] > 0.9                            # pure-dipole signal


def test_load_method_registry():
    assert isinstance(load_method("dower_dipole"), DowerDipole)
    assert isinstance(load_method("ecgrecon.methods.baselines:DowerDipole"), DowerDipole)


def test_perturb_lead_hits_target_and_keeps_einthoven():
    ecg = synthetic_ecg()
    p = windowed_offset(len(ecg), 280, 350, 0.2)
    res = perturb_lead(ecg, p, "Lead II", return_details=True)
    assert res.max_edit_error < 1e-9
    assert limb_consistency_error(res.ecg) < 1e-9
    np.testing.assert_allclose(res.delta_ecg[p == 0], 0, atol=1e-12)
