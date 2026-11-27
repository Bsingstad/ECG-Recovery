"""Tests for the shared helpers, the evaluation views and the gap-robust
predictor selection (review follow-up)."""

import numpy as np
import pytest

from ecgrecon import beats, signal
from ecgrecon.leads import INDEPENDENT_LEADS, LEADS
from ecgrecon.masking import apply_mask, determined_mask, independent_observations, paper_mask
from ecgrecon.methods import beat_template, patient_regression
from ecgrecon.methods.patient_regression import PatientRegression
from ecgrecon.methods.physio import PhysioReconstructor
from ecgrecon.metrics import bootstrap_record_median, evaluate, flag_artefacts

from test_physio import synthetic


@pytest.fixture(scope="module")
def train():
    return np.stack([synthetic(seed=s, local=0.05) for s in range(20)]).astype(np.float32)


@pytest.fixture(scope="module")
def core(train):
    return PatientRegression().fit(train)


# ------------------------------------------------------------ shared helpers
def test_shared_helpers_are_single_implementations():
    # the candidate modules re-export the shared helpers (no private copies)
    assert beat_template.contiguous_runs is signal.contiguous_runs
    assert patient_regression.contiguous_runs is signal.contiguous_runs
    assert beat_template.median_baseline is signal.median_baseline
    assert patient_regression.highpass_observed is signal.highpass_observed
    for name in ("detect_beats", "refine_fiducials", "beat_morphology", "correlation_matrix",
                 "transplant_residual"):
        assert getattr(beat_template, name) is getattr(beats, name)
    runs = signal.contiguous_runs(np.array([0, 1, 1, 0, 1], bool))
    assert runs == [(1, 3), (4, 5)] and all(type(v) is int for r in runs for v in r)


def test_fill_gaps():
    x = np.array([0.0, 1.0, np.nan, 3.0, np.nan])
    m = np.isfinite(x)
    np.testing.assert_allclose(signal.fill_gaps(x, m), [0, 1, 2, 3, 3])
    np.testing.assert_array_equal(signal.fill_gaps(x[:2], m[:2]), x[:2])
    np.testing.assert_array_equal(signal.fill_gaps(x, np.zeros(5, bool)), np.zeros(5))


def test_predict_independent_equals_state_estimate(core):
    ecg = synthetic(seed=201, local=0.1, wander=0.05)
    mask = paper_mask()
    x8, m8 = independent_observations(apply_mask(ecg, mask), mask)
    est, state = core.predict_with_state(x8, m8)
    np.testing.assert_array_equal(core.predict_independent(x8, m8), est)
    assert state["P"] == (1, 2, 6)                       # II, V1, V5


def test_physio_without_residual_is_the_core(train, core):
    ecg = synthetic(seed=202, local=0.1)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    phys = PhysioReconstructor(residual=False).fit(train)
    np.testing.assert_array_equal(phys.reconstruct(obs, mask), core.reconstruct(obs, mask))


# --------------------------------------------------- gaps in rhythm leads
@pytest.mark.parametrize("lead", ["V1", "V5"])
def test_short_gap_in_rhythm_lead_keeps_it_as_predictor(train, core, lead):
    ecg = synthetic(seed=203, local=0.1)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    gap = obs.copy()
    gap[2600:2610, LEADS.index(lead)] = np.nan
    x8, m8 = independent_observations(gap, mask)
    _, state = core.predict_with_state(x8, m8)
    assert INDEPENDENT_LEADS.index(lead) in state["P"]
    phys = PhysioReconstructor().fit(train)
    ref, out = phys.reconstruct(obs, mask), phys.reconstruct(gap, mask)
    assert np.isfinite(out).all()
    ok = mask & np.isfinite(gap)
    np.testing.assert_array_equal(out[ok], ecg[ok])       # observations unchanged
    miss = ~mask
    err_ref = np.sqrt(np.mean((ref[miss] - ecg[miss]) ** 2))
    err_gap = np.sqrt(np.mean((out[miss] - ecg[miss]) ** 2))
    assert err_gap < 1.1 * err_ref + 1e-3


# ------------------------------------------------------------- metrics
def test_determined_mask_paper_layout():
    mask = paper_mask()
    unknown = ~determined_mask(mask)
    n_unknown = dict(zip(LEADS, unknown.sum(axis=0).tolist()))
    # limb leads: only 5-10 s is unknown (2.5-5 s / 0-2.5 s follow from Einthoven)
    for lead in ("I", "III", "aVR", "aVL", "aVF"):
        assert n_unknown[lead] == 2500
    assert n_unknown["II"] == n_unknown["V1"] == n_unknown["V5"] == 0
    for lead in ("V2", "V3", "V4", "V6"):
        assert n_unknown[lead] == 3750
    assert determined_mask(np.broadcast_to(mask, (2,) + mask.shape)).shape == (2,) + mask.shape


def test_unknown_view_weights_independent_leads_equally():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((4, 5000, 12))
    mask = paper_mask()
    res, rec = evaluate(truth, truth, mask, return_records=True)
    u = res["unknown"]
    scored = [LEADS[j] for j in range(12) if np.isfinite(u["pcc_pooled"][j])]
    assert scored == ["I", "V2", "V3", "V4", "V6"]
    assert u["mean_pcc"] == pytest.approx(1.0)
    assert rec["unknown"].shape == (4, 12)


def test_flag_artefacts_and_bootstrap():
    truth = np.zeros((3, 10, 12))
    truth[1, 5, 0] = 30.0
    np.testing.assert_array_equal(flag_artefacts(truth), [False, True, False])
    rng = np.random.default_rng(1)
    a = np.clip(0.9 + 0.05 * rng.standard_normal((200, 12)), -1, 1)
    b = a - 0.01
    d = bootstrap_record_median(a, b, n_boot=200)
    assert d["estimate"] == pytest.approx(0.01)
    assert d["lo"] == pytest.approx(0.01) and d["hi"] == pytest.approx(0.01)
    s = bootstrap_record_median(a, n_boot=200)
    assert s["lo"] <= s["estimate"] <= s["hi"]
