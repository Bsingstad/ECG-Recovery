"""Load PTB-XL records of a split into an (uncompressed) .npz cache for fast experiments.

The cache goes to ``<repo>/cache`` unless ``--cache-dir`` or ``$ECGRECON_CACHE`` is set.

Examples
--------
    python scripts/build_cache.py --split val --n 500
    python scripts/build_cache.py --split train --n 2000
    python scripts/build_cache.py --split test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ecgrecon.data import cache_path, load_records, split_ids, subsample_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--n", type=int, default=None, help="random subset size (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None, help="PTB-XL root (default: ./ptb_xl/... or $PTBXL_ROOT)")
    ap.add_argument("--cache-dir", default=None, help="default: $ECGRECON_CACHE or <repo>/cache")
    args = ap.parse_args()

    ids = subsample_ids(split_ids(args.split, args.root), args.n, args.seed)
    t0 = time.time()
    ecg = load_records(ids, args.root)
    out = cache_path(args.split, args.n, args.cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ecg=ecg, ecg_ids=ids)
    print(f"{args.split}: {len(ids)} records -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
