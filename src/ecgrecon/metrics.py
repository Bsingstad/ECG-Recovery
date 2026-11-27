"""Evaluation metrics.

Five views are reported:

* ``full``    - over the whole 10 s signal, exactly as in the original notebook
                (pooled Pearson r per lead, pooled RMSE per lead, per-record R2).
                Observed samples are included, which inflates scores.
* ``missing`` - only over samples that were *not* observed, for all 12 leads.
                Two distortions: (a) on the paper layout 1250 of the 3750
                missing samples of every limb lead (2.5-5 s for I/III, 0-2.5 s
                for aVR/aVL/aVF) follow exactly from observed limb leads via
                Einthoven/Goldberger, so any method ending in ``finalize`` gets
                them almost free; (b) I, III, aVR, aVL, aVF at 5-10 s are all
                functions of one predicted quantity (lead I), so the 12-lead
                mean gives lead I's error 5/9 of the weight.
* ``shape``   - as ``missing`` but after a 0.67 Hz zero-phase high-pass of both
                truth and prediction: waveform accuracy without baseline
                wander, which is largely unpredictable and otherwise dominates
                pooled scores through a few artefact records.
* ``unknown`` - the main summary: only the samples of the 8 independent leads
                (I, II, V1-V6) that are neither observed nor determined by
                observed leads (:func:`ecgrecon.masking.determined_mask`), each
                lead weighted equally. On the paper layout these are I at
                5-10 s and V2, V3, V4, V6 in their three missing quarters.
* ``unknown_shape`` - ``unknown`` after the 0.67 Hz high-pass.

Pooled scores are sensitive to single artefact records (see
:func:`flag_artefacts`); record-median r and paired bootstrap intervals over
records (:func:`bootstrap_record_median`) are more robust summaries.
"""

from __future__ import annotations

import warnings

import numpy as np

from .leads import INDEPENDENT_IDX, LEADS
from .masking import determined_mask

#: views whose per-record Pearson r can be returned by :func:`evaluate`
RECORD_VIEWS = ("missing", "shape", "unknown", "unknown_shape")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def evaluate(
    truth: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    fs: float = 500.0,
    return_records: bool = False,
):
    """Compute metrics for a batch.

    Parameters
    ----------
    truth, pred : (N, n_samples, 12)
    mask : (n_samples, 12) or (N, n_samples, 12), True = observed.
    fs : sampling rate, used for the ``shape`` views (0.67 Hz high-pass of
        truth and prediction before scoring, i.e. accuracy of the waveform
        without unpredictable baseline wander).
    return_records : also return ``{view: (N, 12) per-record Pearson r}``
        for the views in :data:`RECORD_VIEWS` (NaN where a lead has no
        scored samples), e.g. for :func:`bootstrap_record_median`.

    Returns
    -------
    dict, or ``(dict, records)`` if ``return_records``.
    """
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    independent = np.zeros(12, dtype=bool)
    independent[list(INDEPENDENT_IDX)] = True
    unknown = ~determined_mask(mask) & independent
    if mask.ndim == 2:
        mask = np.broadcast_to(mask, truth.shape)
        unknown = np.broadcast_to(unknown, truth.shape)
    miss = ~mask
    res: dict = {"n_records": int(truth.shape[0]), "leads": list(LEADS)}

    # --- notebook-compatible, full signal --------------------------------
    pcc_full = [_pearson(truth[..., j].ravel(), pred[..., j].ravel()) for j in range(12)]
    rmse_full = [float(np.sqrt(np.mean((truth[..., j] - pred[..., j]) ** 2))) for j in range(12)]
    ss_res = ((truth - pred) ** 2).sum(axis=1)
    ss_tot = ((truth - truth.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_full = 1.0 - ss_res / ss_tot                                # (N, 12)
    res["full"] = {
        "pcc_pooled": pcc_full,
        "rmse": rmse_full,
        "r2_median": np.nanmedian(r2_full, axis=0).tolist(),
        "mean_pcc": float(np.nanmean(pcc_full)),
        "mean_rmse": float(np.mean(rmse_full)),
    }

    records = {}
    hp_t, hp_p = highpass(truth, fs), highpass(pred, fs)
    views = {
        "missing": (truth, pred, miss),
        "shape": (hp_t, hp_p, miss),
        "unknown": (truth, pred, unknown),
        "unknown_shape": (hp_t, hp_p, unknown),
    }
    for view, (t, p, m) in views.items():
        res[view], records[view] = _masked_view(t, p, m)
    return (res, records) if return_records else res


def highpass(x: np.ndarray, fs: float = 500.0, cutoff: float = 0.67) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth high-pass along the time axis (-2).

    AHA/ACC/HRS (2007) recommend a 0.05 Hz lower cut-off for diagnostic ECG
    and accept up to 0.67 Hz with zero-phase (linear-phase) filtering, which
    is what is used here."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-2)


def _masked_view(truth: np.ndarray, pred: np.ndarray, miss: np.ndarray) -> tuple[dict, np.ndarray]:
    """Scores over the samples where ``miss`` is True; also returns the
    per-record Pearson r ``(N, 12)`` (NaN where a record has no such sample)."""
    N = truth.shape[0]
    pcc_m, rmse_m, rec_pcc_m, r2_m = [], [], [], []
    rec = np.full((N, 12), np.nan)
    for j in range(12):
        mj = miss[..., j]
        if not mj.any():
            pcc_m.append(float("nan")); rmse_m.append(float("nan"))
            rec_pcc_m.append(float("nan")); r2_m.append(float("nan"))
            continue
        t, p = truth[..., j][mj], pred[..., j][mj]
        pcc_m.append(_pearson(t, p))
        rmse_m.append(float(np.sqrt(np.mean((t - p) ** 2))))
        per_r2 = np.full(N, np.nan)
        for n in range(N):
            mm = mj[n]
            if not mm.any():
                continue
            tn, pn = truth[n, mm, j], pred[n, mm, j]
            rec[n, j] = _pearson(tn, pn)
            sst = ((tn - tn.mean()) ** 2).sum()
            if sst > 0:
                per_r2[n] = 1 - ((tn - pn) ** 2).sum() / sst
        rec_pcc_m.append(float(np.nanmedian(rec[:, j])))
        r2_m.append(float(np.nanmedian(per_r2)))
    out = {
        "pcc_pooled": pcc_m,
        "rmse": rmse_m,
        "pcc_record_median": rec_pcc_m,
        "r2_record_median": r2_m,
        "mean_pcc": float(np.nanmean(pcc_m)),
        "mean_rmse": float(np.nanmean(rmse_m)),
        "mean_pcc_record_median": float(np.nanmean(rec_pcc_m)),
        "mean_r2_record_median": float(np.nanmean(r2_m)),
    }
    return out, rec


def flag_artefacts(truth: np.ndarray, max_abs_mv: float = 10.0) -> np.ndarray:
    """Boolean ``(N,)``: records whose true signal exceeds ``max_abs_mv`` in
    any lead and sample. Surface ECG amplitudes above about 10 mV are not
    physiological (electrode pops, saturation), and one such record can
    dominate pooled scores. For sensitivity analyses only, not for tuning."""
    truth = np.asarray(truth)
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.abs(truth), axis=(1, 2)) > max_abs_mv


def _record_median_stat(rec: np.ndarray) -> np.ndarray:
    """Mean over leads of the median over records; ``rec`` is ``(..., N, 12)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.nanmedian(rec, axis=-2), axis=-1)


def record_median_score(rec: np.ndarray) -> float:
    """Mean over leads of the median over records of per-record r
    (``mean_pcc_record_median`` of a view, from its ``(N, 12)`` records)."""
    return float(_record_median_stat(np.asarray(rec, dtype=float)))


def bootstrap_record_median(
    rec_a: np.ndarray,
    rec_b: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Percentile bootstrap over records of :func:`record_median_score`.

    ``rec_a`` (and ``rec_b``) are per-record r matrices ``(N, 12)`` from
    :func:`evaluate` with ``return_records=True``. With ``rec_b`` the
    statistic is the *paired* difference ``score(a) - score(b)`` (the same
    resampled records for both methods). Returns
    ``{"estimate", "lo", "hi", "n_boot", "level"}``.
    """
    rec_a = np.asarray(rec_a, dtype=float)
    N = rec_a.shape[0]
    idx = np.random.default_rng(seed).integers(0, N, size=(n_boot, N))
    chunks = [idx[k:k + 200] for k in range(0, n_boot, 200)]
    est = record_median_score(rec_a)
    boots = np.concatenate([_record_median_stat(rec_a[c]) for c in chunks])
    if rec_b is not None:
        rec_b = np.asarray(rec_b, dtype=float)
        est -= record_median_score(rec_b)
        boots = boots - np.concatenate([_record_median_stat(rec_b[c]) for c in chunks])
    a = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(boots, [a, 1.0 - a])
    return {"estimate": float(est), "lo": float(lo), "hi": float(hi),
            "n_boot": int(n_boot), "level": float(level)}


def format_table(results: dict[str, dict], view: str = "missing") -> str:
    """Plain-text comparison table of several methods' results."""
    names = list(results)
    lines = []
    head = f"{'lead':>5} " + " ".join(f"{n[:22]:>22}" for n in names)
    key = "pcc_pooled"
    lines.append(f"[{view}] pooled Pearson r / RMSE (mV) per lead")
    lines.append(head)
    for j, lead in enumerate(LEADS):
        cells = []
        for n in names:
            r = results[n][view]
            cells.append(f"{r[key][j]:>9.4f} / {r['rmse'][j]:<9.4f}"[:22].rjust(22))
        lines.append(f"{lead:>5} " + " ".join(cells))
    cells = []
    for n in names:
        r = results[n][view]
        cells.append(f"{r['mean_pcc']:>9.4f} / {r['mean_rmse']:<9.4f}"[:22].rjust(22))
    lines.append(f"{'mean':>5} " + " ".join(cells))
    return "\n".join(lines)
