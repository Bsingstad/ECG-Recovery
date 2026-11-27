"""Compare a method against the U-Net notebook's saved test-set predictions.

The notebook saved ``original`` (ground truth) and ``reconstructed`` (U-Net
output) for its own random test split. This script

1. identifies the PTB-XL ``ecg_id`` of every test record by signal
   fingerprint (needed to check patient overlap and to keep our population
   prior free of the U-Net test records);
2. reports how many U-Net test patients also have records in the U-Net
   train/val part (the notebook split record-wise, not patient-wise);
3. reconstructs the same records with ``--method`` (prior fitted only on
   training-cache records that are NOT in the U-Net test set) using the same
   paper-layout mask the U-Net saw;
4. scores both with the repository metrics, streaming in chunks so that the
   ~2 GB arrays never need to be held in memory twice, and also reports
   Einthoven/Goldberger consistency of each output.

Example
-------
    python scripts/compare_unet.py --original original.npy --reconstructed reconstructed.npy -m physio --jobs 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ecgrecon.data import load_cached, load_metadata, ptbxl_root
from ecgrecon.leads import LEADS, LIMB_FROM_I_II
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods import load_method
from ecgrecon.metrics import highpass

FP_SAMPLES = 500
FP_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


def fingerprint(sig: np.ndarray) -> str:
    q = np.round(np.asarray(sig[:FP_SAMPLES, FP_LEADS], dtype=np.float64) * 1000).astype(np.int64)
    return hashlib.sha1(q.tobytes()).hexdigest()


def _fp_record(path: str) -> str:
    import wfdb

    sig, _ = wfdb.rdsamp(path, sampto=FP_SAMPLES)
    return fingerprint(sig)


def match_ids(original: np.ndarray, jobs: int, cache_file: Path) -> np.ndarray:
    """ecg_id for each row of ``original`` (-1 if not found)."""
    if cache_file.exists():
        ids = np.loadtxt(cache_file, dtype=np.int64, delimiter=",", skiprows=1, usecols=1)
        if len(ids) == len(original):
            return ids
    meta = load_metadata()
    root = ptbxl_root()
    paths = [str(root / p) for p in meta.filename_hr]
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        fps = list(ex.map(_fp_record, paths, chunksize=64))
    table: dict[str, list[int]] = {}
    for eid, fp in zip(meta.index, fps):
        table.setdefault(fp, []).append(int(eid))
    ids = np.full(len(original), -1, dtype=np.int64)
    from ecgrecon.data import load_record

    ambiguous, used = [], set()
    for k in range(len(original)):
        cand = table.get(fingerprint(original[k]), [])
        if len(cand) == 1:
            ids[k] = cand[0]
        elif len(cand) > 1:
            # Resolve on the full signal. PTB-XL contains a few records with
            # identical signals; give each test row a not-yet-used candidate.
            for eid in cand:
                if eid not in used and np.allclose(load_record(eid, root, meta), original[k], atol=1e-6):
                    ids[k] = eid
                    break
            ambiguous.append((k, cand, int(ids[k])))
        if ids[k] >= 0:
            used.add(int(ids[k]))
    print(f"matched {np.sum(ids >= 0)}/{len(ids)} records in {time.time() - t0:.0f} s; "
          f"ambiguous fingerprints (row, candidates, chosen): {ambiguous}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write("row,ecg_id\n")
        f.writelines(f"{k},{e}\n" for k, e in enumerate(ids))
    return ids


def split_overlap(test_ids: np.ndarray) -> dict:
    meta = load_metadata()
    test_set = set(int(i) for i in test_ids if i >= 0)
    rest = meta.loc[~meta.index.isin(test_set)]
    rest_patients = set(rest.patient_id.dropna().astype(int))
    tmeta = meta.loc[sorted(test_set)]
    pats = tmeta.patient_id.dropna().astype(int)
    shared = pats.isin(rest_patients)
    folds = tmeta.strat_fold.value_counts().sort_index()
    return {
        "test_records": len(test_set),
        "test_patients": int(pats.nunique()),
        "test_records_whose_patient_also_in_unet_train_or_val": int(shared.sum()),
        "fraction_records_with_patient_overlap": float(shared.mean()),
        "test_records_by_ptbxl_strat_fold": {int(k): int(v) for k, v in folds.items()},
    }


def patient_disjoint_rows(test_ids: np.ndarray) -> np.ndarray:
    """Boolean per row: True if no other record of the same patient lies
    outside the U-Net test set (i.e. the U-Net never saw this patient)."""
    meta = load_metadata()
    test_set = set(int(i) for i in test_ids if i >= 0)
    seen = set(meta.loc[~meta.index.isin(test_set)].patient_id.dropna().astype(int))
    pid = meta.patient_id
    return np.array([i >= 0 and int(pid.loc[int(i)]) not in seen for i in test_ids])


class Stream:
    """Streaming accumulator for pooled r / RMSE and per-record r (missing,
    shape and full views) plus Einthoven consistency."""

    def __init__(self):
        z = lambda: np.zeros(12)
        self.s = {v: dict(n=z(), t=z(), p=z(), tt=z(), pp=z(), tp=z(), se=z()) for v in ("missing", "shape", "full")}
        self.rec = {v: [] for v in ("missing", "shape")}
        self.einthoven = []

    def _acc(self, view, t, p, m):
        a = self.s[view]
        for j in range(12):
            tj, pj = t[..., j][m[..., j]], p[..., j][m[..., j]]
            a["n"][j] += tj.size; a["t"][j] += tj.sum(); a["p"][j] += pj.sum()
            a["tt"][j] += (tj * tj).sum(); a["pp"][j] += (pj * pj).sum(); a["tp"][j] += (tj * pj).sum()
            a["se"][j] += ((tj - pj) ** 2).sum()

    @staticmethod
    def _rec_r(t, p, miss):
        out = np.full((t.shape[0], 12), np.nan)
        for n in range(t.shape[0]):
            for j in range(12):
                mm = miss[:, j]
                if mm.any():
                    a = t[n, mm, j] - t[n, mm, j].mean()
                    b = p[n, mm, j] - p[n, mm, j].mean()
                    d = np.sqrt((a * a).sum() * (b * b).sum())
                    out[n, j] = (a * b).sum() / d if d > 0 else np.nan
        return out

    def add(self, truth, pred, mask):
        miss = ~mask
        mb = np.broadcast_to(miss, truth.shape)
        self._acc("full", truth, pred, np.ones_like(mb))
        self._acc("missing", truth, pred, mb)
        self.rec["missing"].append(self._rec_r(truth, pred, miss))
        ht, hp = highpass(truth), highpass(pred)
        self._acc("shape", ht, hp, mb)
        self.rec["shape"].append(self._rec_r(ht, hp, miss))
        i, ii = pred[..., 0], pred[..., 1]
        dev = [pred[..., LEADS.index(k)] - (a * i + b * ii) for k, (a, b) in LIMB_FROM_I_II.items()]
        self.einthoven.append(np.sqrt(np.mean(np.stack(dev, -1) ** 2, axis=(1, 2))))

    def result(self) -> dict:
        res = {}
        for view, a in self.s.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                cov = a["tp"] / a["n"] - a["t"] * a["p"] / a["n"] ** 2
                vt = a["tt"] / a["n"] - (a["t"] / a["n"]) ** 2
                vp = a["pp"] / a["n"] - (a["p"] / a["n"]) ** 2
                r = cov / np.sqrt(vt * vp)
                rmse = np.sqrt(a["se"] / a["n"])
            if view != "full":
                r[a["n"] == 0] = np.nan
            res[view] = {"pcc_pooled": r.tolist(), "rmse": rmse.tolist(),
                         "mean_pcc": float(np.nanmean(r[np.isfinite(r)])),
                         "mean_rmse": float(np.nanmean(rmse[np.isfinite(r)]))}
            if view in self.rec:
                rr = np.concatenate(self.rec[view])
                med = np.nanmedian(rr, axis=0)
                res[view]["pcc_record_median"] = med.tolist()
                res[view]["mean_pcc_record_median"] = float(np.nanmean(med))
        e = np.concatenate(self.einthoven)
        res["einthoven_rms_mV"] = {"median": float(np.median(e)), "mean": float(e.mean()), "max": float(e.max())}
        return res


_M = None


def _init(m):
    global _M
    _M = m


def _run(o):
    return _M.reconstruct(o, paper_mask(o.shape[0], o.shape[0] // 4)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--reconstructed", required=True)
    ap.add_argument("-m", "--method", default="physio")
    ap.add_argument("-k", "--kwargs", default="{}")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default="results/unet_test_ecg_ids.csv")
    ap.add_argument("--pred-out", default=None, help="save our predictions (.npy, float32 memmap)")
    ap.add_argument("--out", default="results/unet_comparison.json")
    args = ap.parse_args()

    orig = np.load(args.original, mmap_mode="r")
    unet = np.load(args.reconstructed, mmap_mode="r")
    N = len(orig) if args.limit is None else min(args.limit, len(orig))
    n_samples = orig.shape[1]
    mask = paper_mask(n_samples, n_samples // 4)

    ids = match_ids(orig, args.jobs, Path(args.ids_file))
    overlap = split_overlap(ids)
    print(json.dumps(overlap, indent=1))
    fold10 = set(load_metadata().query("strat_fold == 10").index)

    train, train_ids = load_cached("train", args.train_n)
    keep = ~np.isin(train_ids, ids)
    print(f"population prior: {keep.sum()}/{len(keep)} train-cache records (excluded {np.sum(~keep)} in U-Net test set)")
    method = load_method(args.method, **json.loads(args.kwargs)).fit(train[keep])
    del train

    pred_mm = None
    if args.pred_out:
        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        pred_mm = np.lib.format.open_memmap(args.pred_out, mode="w+", dtype=np.float32, shape=(N,) + orig.shape[1:])

    disjoint = patient_disjoint_rows(ids)
    groups = {
        "all": np.arange(N),
        # U-Net test records whose patient has no other record in the U-Net train/val part
        "patient_disjoint": np.array([k for k in range(N) if disjoint[k]]),
        # records that are also in the official PTB-XL test fold (our held-out split)
        "ptbxl_fold10": np.array([k for k in range(N) if ids[k] in fold10]),
    }
    names = ("unet", "unet+obs", args.method)
    streams = {g: {n: Stream() for n in names} for g in groups}
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(method,)) as ex:
        for s in range(0, N, args.chunk):
            rows = np.arange(s, min(s + args.chunk, N))
            truth = np.asarray(orig[rows], dtype=np.float64)
            u = np.asarray(unet[rows], dtype=np.float64)
            ours = np.stack(list(ex.map(_run, [apply_mask(t, mask) for t in truth], chunksize=4))).astype(np.float64)
            if pred_mm is not None:
                pred_mm[rows] = ours
            preds = {"unet": u, "unet+obs": np.where(mask, truth, u), args.method: ours}
            for g, gi in groups.items():
                sel = np.isin(rows, gi)
                if sel.any():
                    for n in names:
                        streams[g][n].add(truth[sel], preds[n][sel], mask)
            print(f"{rows[-1] + 1}/{N} records ({time.time() - t0:.0f} s)", flush=True)

    out = {"overlap": overlap, "n_records": int(N), "method": args.method, "kwargs": json.loads(args.kwargs),
           "groups": {g: {"n": int(len(gi)), **{n: streams[g][n].result() for n in names}} for g, gi in groups.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    for g in groups:
        print(f"\n=== {g} (n={out['groups'][g]['n']}) ===")
        print(f"{'':>10} {'missing r':>10} {'rmse':>8} {'rec-med r':>10} | {'shape r':>8} {'rmse':>8} {'rec-med r':>10} | {'full r':>7} | Einthoven RMS")
        for n in names:
            r = out["groups"][g][n]
            print(f"{n[:10]:>10} {r['missing']['mean_pcc']:>10.4f} {r['missing']['mean_rmse']:>8.4f} "
                  f"{r['missing']['mean_pcc_record_median']:>10.4f} | {r['shape']['mean_pcc']:>8.4f} "
                  f"{r['shape']['mean_rmse']:>8.4f} {r['shape']['mean_pcc_record_median']:>10.4f} | "
                  f"{r['full']['mean_pcc']:>7.4f} | {r['einthoven_rms_mV']['median']:.4f} mV")
        print("per-lead missing r:", "  ".join(
            f"{LEADS[j]}: U {out['groups'][g]['unet']['missing']['pcc_pooled'][j]:.3f} / P {out['groups'][g][args.method]['missing']['pcc_pooled'][j]:.3f}"
            for j in range(12) if np.isfinite(out['groups'][g]['unet']['missing']['pcc_pooled'][j])))


if __name__ == "__main__":
    main()
