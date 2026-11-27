import numpy as np
import pytest

from ecgrecon.leads import LEADS, expand_independent, lead_matrix, limb_consistency_error
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods import load_method
from ecgrecon.methods.beat_template import (
    BeatTemplate, contiguous_runs, detect_beats, median_baseline,
)
from ecgrecon.metrics import evaluate

FS = 500
BEAT_TIMES = np.arange(0.4, 10.0, 0.85)


def gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def synthetic(seed=0, local=0.0, wander=0.0, noise=0.005):
    """Dipole ECG with optional beat-locked non-dipolar potential in V2/V3
    (``local`` mV) and baseline wander (``wander`` mV)."""
    rng = np.random.default_rng(seed)
    n = 5000
    t = np.arange(n) / FS
    h = np.zeros((n, 3))
    r = np.zeros((n, 8))
    for b in BEAT_TIMES:
        for dt, sd, vec in [(-0.16, 0.025, (0.10, 0.10, 0.00)),
                            (0.00, 0.012, (1.00, 0.90, 0.30)),
                            (0.28, 0.060, (0.30, 0.25, -0.10))]:
            h += gauss(t, b + dt, sd)[:, None] * np.array(vec)
        # local (non-dipolar) notch after the QRS, only seen by V2 and V3
        r[:, 3] += local * gauss(t, b + 0.05, 0.015)
        r[:, 4] += 0.7 * local * gauss(t, b + 0.05, 0.015)
    x8 = h @ lead_matrix(("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")).T + r
    x8 += wander * np.sin(2 * np.pi * 0.15 * t[:, None] + rng.uniform(0, 6.28, 8)) * rng.uniform(0.5, 1, 8)
    x8 += noise * rng.standard_normal(x8.shape)
    return expand_independent(x8)


@pytest.fixture(scope="module")
def train():
    return np.stack([synthetic(seed=s, local=0.1) for s in range(6)])


def test_contiguous_runs():
    assert contiguous_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert contiguous_runs(np.zeros(3, bool)) == []


def test_median_baseline_tracks_slow_wander():
    t = np.arange(5000) / FS
    slow = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    x = synthetic(local=0.0)[:, 1]
    # the added wander must show up in the baseline estimate almost one-to-one
    d = median_baseline(x + slow, FS) - median_baseline(x, FS)
    assert np.sqrt(np.mean((d - slow) ** 2)) < 0.06      # ~10% of the 0.5 mV wander


def test_detect_beats_finds_all_qrs():
    ecg = synthetic()
    sig = ecg[:, [1, 6, 10]] - median_baseline(ecg[:, [1, 6, 10]], FS)
    fid = detect_beats(sig, FS)
    assert len(fid) == len(BEAT_TIMES)
    assert np.max(np.abs(fid / FS - BEAT_TIMES)) < 0.03


@pytest.mark.parametrize("kw", [{"residual": False}, {"residual": True}])
def test_contract(train, kw):
    ecg = synthetic(seed=11, local=0.1, wander=0.2)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    m = BeatTemplate(**kw).fit(train)
    rec = m.reconstruct(obs, mask)
    assert rec.shape == ecg.shape and np.isfinite(rec).all()
    np.testing.assert_allclose(rec[mask], ecg[mask])
    assert limb_consistency_error(rec) < 1e-9
    assert evaluate(ecg[None], rec[None], mask)["missing"]["mean_pcc"] > 0.9
    # deterministic
    np.testing.assert_array_equal(rec, m.reconstruct(obs, mask))


def test_residual_template_recovers_local_potential(train):
    ecg = synthetic(seed=12, local=0.25)
    mask = paper_mask()
    obs = apply_mask(ecg, mask)
    j = LEADS.index("V2")
    miss = ~mask[:, j]
    err = {}
    for flag in (False, True):
        rec = BeatTemplate(residual=flag).fit(train).reconstruct(obs, mask)
        err[flag] = np.sqrt(np.mean((rec[miss, j] - ecg[miss, j]) ** 2))
    assert err[True] < 0.6 * err[False]


def test_works_without_training_data():
    ecg = synthetic(seed=3, local=0.1)
    mask = paper_mask()
    rec = BeatTemplate().fit(None).reconstruct(apply_mask(ecg, mask), mask)
    assert np.isfinite(rec).all()


def test_flat_record_is_finite():
    mask = paper_mask()
    rec = BeatTemplate().reconstruct(apply_mask(np.zeros((5000, 12)), mask), mask)
    assert np.isfinite(rec).all() and np.abs(rec).max() < 1e-9


def test_load_by_spec():
    m = load_method("ecgrecon.methods.beat_template:BeatTemplate", corr_time=0.1)
    assert isinstance(m, BeatTemplate) and m.corr_time == 0.1
