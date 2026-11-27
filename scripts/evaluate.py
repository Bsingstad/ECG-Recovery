"""Evaluate reconstruction methods on cached PTB-XL data (paper-layout mask).

Examples
--------
    python scripts/evaluate.py --split val --n 500 -m physio -m patient_regression --jobs 12
    python scripts/evaluate.py --split val --n 500 -m mypkg.module:MyMethod -k '{"alpha": 0.1}'
    # compare with the U-Net notebook's saved test set (data.npz: original, reconstructed)
    python scripts/evaluate.py --from-npz data.npz -m physio

Methods are fitted on ``cache/ptbxl_train_<train-n>.npz`` (strat_fold 1-8).
If that cache is missing the script stops; pass ``--no-train`` to evaluate
methods without population priors on purpose.

Warning about ``--from-npz``: the notebook's ``data.npz`` is a random
*record-wise* 20% of all of PTB-XL, so some of its records (and patients) can
also be in the strat_fold 1-8 training cache used to fit the methods here,
and the U-Net's own record-wise split mixes patients between its training and
test sets. ``data.npz`` has no ecg_ids, so overlapping records cannot be
removed. Use this mode for a quick sanity check only; the unbiased comparison
is on strat_fold 10 (``--split test``) with every method, including the
U-Net, trained on folds 1-8 only.

Output (``--out``): per method the metric views of :mod:`ecgrecon.metrics`,
the same views with artefact records excluded (``excl_artefacts``), paired
bootstrap intervals of record-median r (``bootstrap``) and a ``meta`` block
that records how the numbers were produced.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import ecgrecon
from ecgrecon.data import load_cached
from ecgrecon.masking import apply_mask, paper_mask
from ecgrecon.methods import REGISTRY, load_method
from ecgrecon.metrics import bootstrap_record_median, evaluate, flag_artefacts, format_table

_METHOD = None
_SUMMARY_VIEWS = ("unknown", "unknown_shape", "missing", "shape")


def _init(method):
    global _METHOD
    _METHOD = method


def _run(args):
    ecg_obs, mask = args
    return _METHOD.reconstruct(ecg_obs, mask)


def run_method(spec, kwargs, train, truth, mask, jobs):
    """Fit once in the parent process, then ship the *fitted* method (small)
    to the workers instead of the training data."""
    t0 = time.time()
    method = load_method(spec, **kwargs).fit(train)
    t_fit = time.time() - t0
    obs = [apply_mask(t, mask) for t in truth]
    t0 = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(jobs, initializer=_init, initargs=(method,)) as ex:
            preds = list(ex.map(_run, [(o, mask) for o in obs], chunksize=8))
    else:
        _init(method)
        preds = [_run((o, mask)) for o in obs]
    return method, np.stack(preds), time.time() - t0, t_fit


def _jsonable(v):
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def effective_kwargs(spec: str, kwargs: dict) -> dict:
    """Constructor defaults of the method's class merged with ``kwargs``."""
    mod, cls = REGISTRY.get(spec, spec).split(":", 1)
    import importlib

    sig = inspect.signature(getattr(importlib.import_module(mod), cls).__init__)
    eff = {k: p.default for k, p in sig.parameters.items()
           if k != "self" and p.default is not inspect.Parameter.empty}
    eff.update(kwargs)
    return {k: _jsonable(v) for k, v in eff.items()}


def _accepts_fs(spec: str) -> bool:
    return "fs" in effective_kwargs(spec, {})


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--n", type=int, default=None, help="cache subset size (as built)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    ap.add_argument("--from-npz", default=None,
                    help="evaluate on 'original' array of a notebook data.npz (see warning above)")
    ap.add_argument("--train-n", type=int, default=2000, help="train cache used for fit()")
    ap.add_argument("--no-train", action="store_true",
                    help="fit methods without training data (fallback priors)")
    ap.add_argument("--fs", type=float, default=None,
                    help="sampling rate in Hz (default: 'fs' in the npz, else 500)")
    ap.add_argument("--artefact-mv", type=float, default=10.0,
                    help="records with |truth| above this (mV) are excluded in 'excl_artefacts'")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (0 = off)")
    ap.add_argument("-m", "--method", action="append", required=True)
    ap.add_argument("-k", "--kwargs", action="append", default=None,
                    help="JSON kwargs per method (same order as -m)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--save-pred", default=None, help="save predictions of the LAST method (.npy)")
    args = ap.parse_args()

    ref, ids, fs = None, None, args.fs
    if args.from_npz:
        print("WARNING: --from-npz data is a record-wise random split of PTB-XL that can overlap "
              "the training cache; not a clean held-out comparison (see --help).", file=sys.stderr)
        z = np.load(args.from_npz)
        truth = z["original"].astype(np.float64)
        if "reconstructed" in z:
            ref = z["reconstructed"]
        if "ecg_ids" in z:
            ids = z["ecg_ids"]
        if fs is None and "fs" in z:
            fs = float(z["fs"])
    else:
        truth, ids = load_cached(args.split, args.n)
        truth = truth.astype(np.float64)
    fs = 500.0 if fs is None else float(fs)
    if args.limit:
        truth = truth[: args.limit]
        ids = None if ids is None else ids[: args.limit]
    mask = paper_mask(truth.shape[1], truth.shape[1] // 4)

    if args.no_train:
        train = None
    else:
        try:
            train, _ = load_cached("train", args.train_n)
        except FileNotFoundError as err:
            sys.exit(f"error: {err}\n(pass --no-train to evaluate without training data)")

    kw_list = [json.loads(k) for k in (args.kwargs or [])]
    kw_list += [{}] * (len(args.method) - len(kw_list))
    if fs != 500.0:
        for spec, kw in zip(args.method, kw_list):
            if not _accepts_fs(spec):
                sys.exit(f"error: method {spec!r} has no fs parameter (assumes 500 Hz)")
            kw.setdefault("fs", fs)

    commit = _git_commit()
    flagged = flag_artefacts(truth, args.artefact_mv)
    keep = ~flagged
    meta = {
        "argv": sys.argv,
        "split": None if args.from_npz else args.split,
        "from_npz": args.from_npz,
        "n": args.n,
        "limit": args.limit,
        "n_records": int(len(truth)),
        "fs": fs,
        "train_n": None if train is None else int(len(train)),
        "train_cache_n": None if train is None else args.train_n,
        "ecg_ids_sha1": None if ids is None else hashlib.sha1(
            np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
        "artefact_threshold_mv": args.artefact_mv,
        "artefact_records": (np.asarray(ids)[flagged].tolist() if ids is not None
                             else np.flatnonzero(flagged).tolist()),
        "artefact_records_are": "ecg_ids" if ids is not None else "row indices",
        "ecgrecon_version": ecgrecon.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": _git_dirty() if commit else None,
        "methods": {},
    }

    results: dict = {}
    records: dict = {}
    if ref is not None:
        results["unet(notebook)"] = evaluate(truth, ref[: len(truth)], mask, fs=fs)
    for spec, kw in zip(args.method, kw_list):
        method, pred, dt, t_fit = run_method(spec, kw, train, truth, mask, args.jobs)
        label = spec if not kw else f"{spec} {json.dumps(kw)}"
        meta["methods"][label] = {"spec": spec, "class": f"{type(method).__module__}:{type(method).__name__}",
                                  "kwargs": effective_kwargs(spec, kw)}
        r, records[label] = evaluate(truth, pred, mask, fs=fs, return_records=True)
        r["seconds"] = dt
        r["fit_seconds"] = t_fit
        if flagged.any() and keep.any():
            ex = evaluate(truth[keep], pred[keep], mask, fs=fs)
            r["excl_artefacts"] = {v: ex[v] for v in ("missing", "shape", "unknown", "unknown_shape")}
        results[label] = r
        u, us, ms, s = (r[v] for v in _SUMMARY_VIEWS)
        print(f"{label}: unknown r={u['mean_pcc']:.4f} rmse={u['mean_rmse']:.4f} "
              f"rec-median r={u['mean_pcc_record_median']:.4f} | unknown-shape r={us['mean_pcc']:.4f} "
              f"rec-median r={us['mean_pcc_record_median']:.4f}\n"
              f"    missing r={ms['mean_pcc']:.4f} rmse={ms['mean_rmse']:.4f} "
              f"rec-median r={ms['mean_pcc_record_median']:.4f} | shape r={s['mean_pcc']:.4f} "
              f"rmse={s['mean_rmse']:.4f} rec-median r={s['mean_pcc_record_median']:.4f} "
              f"(fit {t_fit:.1f}s, predict {dt:.1f}s)", flush=True)
        if "excl_artefacts" in r:
            e = r["excl_artefacts"]
            print(f"    excl. {int(flagged.sum())} artefact record(s): "
                  f"unknown r={e['unknown']['mean_pcc']:.4f} rmse={e['unknown']['mean_rmse']:.4f} | "
                  f"missing r={e['missing']['mean_pcc']:.4f} rmse={e['missing']['mean_rmse']:.4f} | "
                  f"shape r={e['shape']['mean_pcc']:.4f} rmse={e['shape']['mean_rmse']:.4f}", flush=True)

    if args.n_boot > 0 and records:
        labels = list(records)
        boot = {"statistic": "mean over leads of record-median Pearson r (percentile bootstrap "
                             "over records)", "per_method": {}, "paired_vs_first": {}}
        for lab in labels:
            boot["per_method"][lab] = {v: bootstrap_record_median(records[lab][v], n_boot=args.n_boot)
                                       for v in _SUMMARY_VIEWS}
        for lab in labels[1:]:
            key = f"{labels[0]} - {lab}"
            boot["paired_vs_first"][key] = {
                v: bootstrap_record_median(records[labels[0]][v], records[lab][v], n_boot=args.n_boot)
                for v in _SUMMARY_VIEWS}
            d = boot["paired_vs_first"][key]
            print(f"paired bootstrap {key}: " + ", ".join(
                f"{v} {d[v]['estimate']:+.4f} [{d[v]['lo']:+.4f}, {d[v]['hi']:+.4f}]"
                for v in _SUMMARY_VIEWS), flush=True)
        results["bootstrap"] = boot

    tables = {k: v for k, v in results.items() if k != "bootstrap"}
    for view in ("unknown", "unknown_shape", "missing", "shape", "full"):
        print()
        print(format_table(tables, view))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, **results}, indent=1))
    if args.save_pred:
        np.save(args.save_pred, pred.astype(np.float32))


if __name__ == "__main__":
    main()
