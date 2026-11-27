"""Beat detection, beat morphology and beat-synchronous residual transplant.

Shared by :mod:`ecgrecon.methods.beat_template` and
:mod:`ecgrecon.methods.physio`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from .signal import FS_DEFAULT


def _robust_scale(x: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def detect_beats(sig: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """QRS fiducials from one or more baseline-corrected leads ``(n, k)``.

    Energy detector: 8-30 Hz band-pass, square, normalise each lead by its
    robust scale, sum over leads, 80 ms moving average, then
    :func:`scipy.signal.find_peaks` with a 300 ms refractory period and an
    adaptive height threshold.
    """
    sig = np.atleast_2d(np.asarray(sig, float).T).T
    n = sig.shape[0]
    if n < int(fs):
        return np.zeros(0, dtype=int)
    sos = butter(2, [8.0, 30.0], btype="band", fs=fs, output="sos")
    f = sosfiltfilt(sos, sig, axis=0)
    sc = _robust_scale(f)
    ok = sc > 1e-6
    if not ok.any():
        return np.zeros(0, dtype=int)
    e = ((f[:, ok] / sc[ok]) ** 2).sum(axis=1)
    e = uniform_filter1d(e, max(1, int(0.08 * fs)))
    hi = np.percentile(e, 99)
    peaks, props = find_peaks(e, height=0.25 * hi, distance=int(0.3 * fs))
    if len(peaks) > 2:
        # drop weak detections (T waves, noise) relative to the typical beat
        h = props["peak_heights"]
        peaks = peaks[h > 0.3 * np.median(h)]
    return peaks.astype(int)


def refine_fiducials(sig: np.ndarray, peaks: np.ndarray, fs: int = FS_DEFAULT,
                     max_shift: float = 0.04) -> np.ndarray:
    """Align fiducials by cross-correlating each beat's QRS (all leads in
    ``sig``) with the median QRS template; reduces envelope-peak jitter."""
    if len(peaks) < 2:
        return peaks
    n = sig.shape[0]
    pre, post, sh = int(0.08 * fs), int(0.10 * fs), int(max_shift * fs)
    pad = pre + post + sh
    sp = np.pad(sig, ((pad, pad), (0, 0)))
    offs = np.arange(-pre, post)
    segs = np.stack([sp[p + pad + offs] for p in peaks])          # (B, w, k)
    tmpl = np.median(segs, axis=0)
    out = peaks.copy()
    for b, p in enumerate(peaks):
        best, best_s = -np.inf, 0
        for s in range(-sh, sh + 1):
            seg = sp[p + s + pad + offs]
            c = float((seg * tmpl).sum())
            if c > best:
                best, best_s = c, s
        out[b] = min(max(p + best_s, 0), n - 1)
    return out


def beat_morphology(sig: np.ndarray, fid: np.ndarray, fs: int = FS_DEFAULT,
                    pre: float = 0.15, post: float = 0.35) -> np.ndarray:
    """Per-beat feature matrix ``(B, w, k)``: the leads of ``sig`` in
    ``[-pre, post]`` s around each fiducial (NaN beyond the record edges)."""
    a, b = int(pre * fs), int(post * fs)
    sp = np.pad(np.asarray(sig, float), ((a, b), (0, 0)), constant_values=np.nan)
    offs = np.arange(-a, b)
    if len(fid) == 0:
        return np.zeros((0, a + b, sig.shape[1]))
    return np.stack([sp[f + a + offs] for f in fid])


def correlation_matrix(feat: np.ndarray) -> np.ndarray:
    """Pearson correlation between beats ``(B, w, k)``, each pair evaluated
    over the samples valid in both beats (edge beats are truncated)."""
    B = feat.shape[0]
    f = feat.reshape(B, -1)
    valid = np.isfinite(f)
    out = np.eye(B)
    for i in range(B):
        for k in range(i + 1, B):
            v = valid[i] & valid[k]
            if v.sum() < 10:
                c = 0.0
            else:
                x, y = f[i, v] - f[i, v].mean(), f[k, v] - f[k, v].mean()
                den = np.sqrt((x * x).sum() * (y * y).sum())
                c = float((x * y).sum() / den) if den > 0 else 0.0
            out[i, k] = out[k, i] = c
    return out


def transplant_residual(res: np.ndarray, target_rows: np.ndarray, fid: np.ndarray, rr,
                        corr: np.ndarray, fs: int = FS_DEFAULT, corr_threshold: float = 0.9,
                        n_match: int = 3, warp: float = 0.5, taper: float = 0.04,
                        gain: float = 1.0) -> np.ndarray:
    """Beat-synchronous transplant of a residual.

    ``res`` (length n) is a residual signal, NaN where unavailable. Every beat
    ``k`` (fiducials ``fid``, RR intervals ``rr = (rr_prev, rr_next)``) whose
    window ``[-min(0.3 s, 0.4 RR_prev), min(0.7 s, 0.6 RR_next)]`` touches
    ``target_rows`` (bool, length n) receives the average residual of up to
    ``n_match`` *other* beats whose QRS lies where ``res`` is available and
    whose rhythm-lead morphology correlation ``corr[k, s]`` is at least
    ``corr_threshold``. The post-QRS part is time-scaled by
    ``(RR_source / RR_target) ** warp``; window edges get a cosine taper of
    ``taper`` s. The window is multiplied by ``gain``. Returns the summed
    windows (length n, zero elsewhere); windows of adjacent beats do not
    overlap.

    The post-QRS stretch is anchored at the QRS end (fiducial + 100 ms), so it
    is an RR-dependent ST-T stretch, not an exact QT correction formula
    (``warp=0.5`` is not Bazett's correction, which scales from QRS onset).
    """
    n = len(res)
    out = np.zeros(n)
    if len(fid) == 0 or gain == 0:
        return out
    rr_prev, rr_next = rr
    qrs_end = int(0.10 * fs)
    tap = max(int(taper * fs), 1)
    avail = np.isfinite(res)
    # source beats: QRS window fully inside the region where res exists
    qrs_off = np.arange(-int(0.06 * fs), qrs_end)
    src = [b for b, f in enumerate(fid)
           if f + qrs_off[0] >= 0 and f + qrs_off[-1] < n and avail[f + qrs_off].all()]
    if not src:
        return out
    src = np.asarray(src)
    for k, f in enumerate(fid):
        a = int(min(0.30 * fs, 0.4 * rr_prev[k]))
        b = int(min(0.70 * fs, 0.6 * rr_next[k]))
        lo, hi = max(f - a, 0), min(f + b, n)
        if not target_rows[lo:hi].any():
            continue
        c = corr[k, src]
        cand = src[(c >= corr_threshold) & (src != k)]
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(-corr[k, cand])][:n_match]
        tau = np.arange(lo - f, hi - f).astype(float)
        acc = np.zeros(len(tau))
        cnt = np.zeros(len(tau))
        for s in cand:
            scale = (rr_prev[s] / rr_prev[k]) ** warp
            ts = np.where(tau > qrs_end, qrs_end + (tau - qrs_end) * scale, tau)
            idx = fid[s] + ts
            i0 = np.floor(idx).astype(int)
            fr = idx - i0
            ok = (i0 >= 0) & (i0 + 1 < n)
            i0c = np.clip(i0, 0, n - 2)
            v0, v1 = res[i0c], res[i0c + 1]
            ok &= np.isfinite(v0) & np.isfinite(v1)
            val = np.where(ok, (1 - fr) * np.nan_to_num(v0) + fr * np.nan_to_num(v1), 0.0)
            acc += val
            cnt += ok
        seg = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        # cosine taper at both window edges
        win = np.ones(len(tau))
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(tap) + 0.5) / tap)
        m = min(tap, len(win))
        win[:m] *= ramp[:m]
        win[len(win) - m:] *= ramp[:m][::-1]
        out[lo:hi] += gain * win * seg
    return out
