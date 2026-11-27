"""PTB-XL loading and splits.

Default split follows the PTB-XL authors' recommendation (Wagner et al., 2020):
``strat_fold`` 1-8 = train, 9 = validation, 10 = test. Folds are patient-wise,
so no patient appears in more than one split. (The original notebook used a
random *record-wise* split, which can put records from the same patient in
both train and test.)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

#: PTB-XL root used when neither ``root`` nor ``$PTBXL_ROOT`` is given
DEFAULT_ROOT = _REPO / "ptb_xl" / \
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"


def default_cache_dir() -> Path:
    """``$ECGRECON_CACHE`` if set, else ``<repo>/cache`` (independent of the
    current working directory)."""
    return Path(os.environ.get("ECGRECON_CACHE", _REPO / "cache"))


SPLIT_FOLDS = {"train": tuple(range(1, 9)), "val": (9,), "test": (10,)}


def ptbxl_root(root: str | os.PathLike | None = None) -> Path:
    """PTB-XL root directory: ``root``, else ``$PTBXL_ROOT``, else
    :data:`DEFAULT_ROOT`; must contain ``ptbxl_database.csv``."""
    root = Path(root or os.environ.get("PTBXL_ROOT", DEFAULT_ROOT))
    if not (root / "ptbxl_database.csv").exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found in {root}")
    return root


def load_metadata(root: str | os.PathLike | None = None):
    import pandas as pd

    root = ptbxl_root(root)
    return pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")


def split_ids(split: str, root=None) -> np.ndarray:
    """``ecg_id`` values in a split, sorted ascending (deterministic)."""
    meta = load_metadata(root)
    folds = SPLIT_FOLDS[split]
    return np.sort(meta.index[meta.strat_fold.isin(folds)].to_numpy())


def load_record(ecg_id: int, root=None, meta=None) -> np.ndarray:
    """Load one 500 Hz record as ``(5000, 12)`` float array in mV."""
    import wfdb

    root = ptbxl_root(root)
    if meta is None:
        meta = load_metadata(root)
    sig, _ = wfdb.rdsamp(str(root / meta.loc[ecg_id, "filename_hr"]))
    return sig


def load_records(ecg_ids, root=None, dtype=np.float32) -> np.ndarray:
    """Load several 500 Hz records as ``(N, n_samples, 12)`` (mV); all
    records must have the same length (5000 samples in PTB-XL)."""
    root = ptbxl_root(root)
    meta = load_metadata(root)
    out = None
    for k, eid in enumerate(ecg_ids):
        sig = load_record(int(eid), root, meta)
        if out is None:
            out = np.empty((len(ecg_ids),) + sig.shape, dtype=dtype)
        out[k] = sig
    return np.empty((0, 5000, 12), dtype=dtype) if out is None else out


def subsample_ids(ids: np.ndarray, n: int | None, seed: int = 0) -> np.ndarray:
    """Deterministic random subset of ``n`` ids (sorted); all if n is None."""
    if n is None or n >= len(ids):
        return np.asarray(ids)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=n, replace=False))


def cache_path(split: str, n: int | None, cache_dir: str | os.PathLike | None = None) -> Path:
    """Path of the cache file of ``split`` with ``n`` records (None = all) in
    ``cache_dir`` (default :func:`default_cache_dir`)."""
    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return cache_dir / f"ptbxl_{split}_{'all' if n is None else n}.npz"


def load_cached(split: str, n: int | None = None,
                cache_dir: str | os.PathLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(ecg, ecg_ids)`` from a cache built by ``scripts/build_cache.py``."""
    p = cache_path(split, n, cache_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; run: python scripts/build_cache.py --split {split}"
            + ("" if n is None else f" --n {n}")
        )
    z = np.load(p)
    return z["ecg"], z["ecg_ids"]
