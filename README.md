# ECGRecovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://github.com/[username]/ECGRecovery/workflows/tests/badge.svg)](https://github.com/[username]/ECGRecovery/actions)


**Reconstruct complete 12-lead ECGs from paper printouts—fast, accurate, and physiologically consistent.**

ECGRecovery recovers full 10-second recordings from partial paper layouts (Cabrera format, 2.5-second traces plus rhythm strips) using Bayesian patient-specific learning. It enforces Einthoven's law exactly and outperforms deep learning baselines on PTB-XL, making it ideal for digitizing clinical archives and historical research data.

| leads       | 0-2.5 s  | 2.5-5 s  | 5-7.5 s  | 7.5-10 s |
|-------------|----------|----------|----------|----------|
| I, II, III  | observed | -        | -        | -        |
| aVR, aVL, aVF | -      | observed | -        | -        |
| V1, V2, V3  | -        | -        | observed | -        |
| V4, V5, V6  | -        | -        | -        | observed |

**Rhythm strips:** II, V1, V5 are observed for the full 10 seconds.


The mask is `ecgrecovery.masking.paper_mask`. It matches `process_ecg` in the
original U-Net notebook.

## Model

Every lead is a projection of the cardiac dipole plus a non-dipolar part,
`ecg = L h(t) + r(t)` (`ecgrecovery.leads`). The limb leads satisfy
Einthoven/Goldberger exactly. Methods predict the 8 independent leads
(I, II, V1-V6) and end in `methods.base.finalize`, which keeps observed
samples unchanged and makes Einthoven/Goldberger hold exactly.

Methods (`src/ecgrecon/methods/`, short names for `scripts/evaluate.py`):

| name | idea |
|------|------|
| `physio` (final method) | patient lead-transform regression + cross-validated, gated beat-locked residual |
| `patient_regression` | patient-specific Bayesian FIR map of II/V1/V5 with a population prior, plus a baseline-wander model |
| `beat_template` | patient lead transform + beat-synchronous residual templates |
| `spatiotemporal_gauss` | patient-adapted spatio-temporal Gaussian conditioning |
| `patient_fa` | patient-specific dipole factor analysis with missing data |
| `dower_dipole`, `population_gaussian` | population baselines |

Shared helpers are in `ecgrecovery.signal` (runs, high-pass, gap filling,
moving-median baseline) and `ecgrecovery.beats` (beat detection, morphology
matching, residual transplant).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[data,dev]     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -q
```

The tests use synthetic data only and do not need PTB-XL.

## Usage

```python
import numpy as np
from ecgrecovery.data import load_cached
from ecgrecovery.masking import paper_mask, apply_mask, from_unet_input
from ecgrecovery.methods import load_method

train, _ = load_cached("train", 2000)              # PTB-XL folds 1-8, for the population prior
model = load_method("physio").fit(train)

mask = paper_mask()                                # (5000, 12) bool, True = observed
ecg_obs = apply_mask(ecg, mask)                    # ecg: (5000, 12) in mV, I..V6 order
full = model.reconstruct(ecg_obs, mask)            # (5000, 12); observed samples unchanged

# or directly from the notebook's 15-channel U-Net input
ecg_obs, mask = from_unet_input(x15)
full = model.reconstruct(ecg_obs, mask)
```

`ecgrecovery.propagate.perturb_lead` / `propagate_edit` is the dipole-based edit
propagation model the method is built on (change one lead, propagate the
change physiologically to the others).

## Data

Download PTB-XL 1.0.1 from PhysioNet (<https://physionet.org/content/ptb-xl/1.0.1/>)
and either unpack it to
`ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/`
or set `PTBXL_ROOT` to the folder that contains `ptbxl_database.csv`.

Splits follow the PTB-XL authors' recommendation, using `strat_fold`
(patient-wise): folds 1-8 are train, fold 9 is validation and fold 10 is test.

Build the caches used for the reported numbers. They go to `cache/` unless
`--cache-dir` or `ECGRECON_CACHE` is set:

```bash
python scripts/build_cache.py --split val --n 500      # random 500 of fold 9 (seed 0)
python scripts/build_cache.py --split train --n 2000   # random 2000 of folds 1-8 (seed 0)
python scripts/build_cache.py --split test             # all of fold 10
```

**Test-split policy:** fold 10 is held out for the final comparison. All
development, tuning and method selection used fold 9 only.

## Evaluation

```bash
python scripts/evaluate.py --split val --n 500 -m physio -m patient_regression --jobs 12 \
    --out results/physio_val.json
```

Methods are fitted on `cache/ptbxl_train_2000.npz`. The script stops if that
cache is missing. Pass `--no-train` to evaluate without population priors on
purpose. The output JSON contains:

- a `meta` block: argv, split, n, train size, sha1 of the ecg_ids, package
  and library versions, git commit, and every method's effective
  constructor settings;
- per method, the metric views below;
- `excl_artefacts`: the same views without records whose true signal exceeds
  10 mV (non-physiological; on the val subset these are ecg_id 5712 and 17787);
- `bootstrap`: percentile bootstrap intervals over records of record-median
  r, per method and paired against the first method.

Metric views (`ecgrecovery.metrics`):

| view | samples scored |
|------|----------------|
| `unknown` (main summary) | samples of the 8 independent leads that are neither observed nor determined by observed leads. On the paper layout: I at 5-10 s, V2, V3, V4 and V6 in their missing quarters. Each lead has equal weight. |
| `unknown_shape` | `unknown` after a 0.67 Hz zero-phase high-pass of truth and prediction (no baseline wander) |
| `missing` | all unobserved samples of all 12 leads. A third of each limb lead's missing samples follow exactly from Einthoven, and five limb leads at 5-10 s all depend on lead I, so lead I counts 5/9 of the mean. |
| `shape` | `missing` after the 0.67 Hz high-pass |
| `full` | the whole signal, as in the original notebook (inflated by observed samples) |

Each view gives pooled Pearson r and RMSE per lead, and record-median r.
Pooled scores can be dominated by single artefact records. Record-median r
and the artefact-excluded scores are more robust.

### Validation results (fold 9, 500 records; `results/physio_val.json`)

| method | unknown r / RMSE / rec-median r | unknown-shape r / RMSE / rec-median r | missing r / RMSE / rec-median r | shape r / RMSE / rec-median r |
|---|---|---|---|---|
| physio | 0.8054 / 0.1824 / 0.9633 | 0.9432 / 0.0765 / 0.9879 | 0.8200 / 0.1406 / 0.9654 | 0.9335 / 0.0650 / 0.9865 |
| patient_regression | 0.8004 / 0.1846 / 0.9547 | 0.9376 / 0.0814 / 0.9814 | 0.8158 / 0.1421 / 0.9582 | 0.9287 / 0.0681 / 0.9803 |

Paired bootstrap of record-median r, physio minus patient_regression:
unknown-shape +0.0065 [+0.0057, +0.0073], shape +0.0062 [+0.0056, +0.0069].

Compared with `beat_template`, physio is better in the shape views
(record-median r, unknown-shape +0.0026 [+0.0019, +0.0030]). Its lead in the
pooled `missing` scores comes from the artefact record 5712. Without the
flagged records the two are about equal in pooled scores (missing r 0.883
vs 0.885, unknown r 0.824 vs 0.824).

### Held-out test results (fold 10, all 2203 records; `results/test_final.json`)

Run once after all development was frozen:
`python scripts/evaluate.py --split test -m physio -m patient_regression -m beat_template -m spatiotemporal_gauss -m patient_fa -m population_gaussian -m dower_dipole --jobs 14 --out results/test_final.json`

| method | unknown r / RMSE / rec-median r | unknown-shape r / rec-median r | missing r / RMSE / rec-median r | shape r / RMSE / rec-median r |
|---|---|---|---|---|
| **physio** | 0.773 / 0.184 / **0.962** | **0.964** / **0.988** | 0.853 / 0.122 / **0.966** | **0.975** / **0.045** / **0.988** |
| beat_template | **0.787** / **0.174** / 0.962 | 0.963 / 0.986 | **0.862** / **0.116** / 0.966 | 0.974 / 0.045 / 0.986 |
| patient_regression | 0.766 / 0.187 / 0.954 | 0.957 / 0.983 | 0.849 / 0.124 / 0.959 | 0.970 / 0.049 / 0.982 |
| spatiotemporal_gauss | 0.726 / 0.198 / 0.944 | 0.939 / 0.979 | 0.816 / 0.135 / 0.950 | 0.959 / 0.055 / 0.979 |
| patient_fa | 0.719 / 0.197 / 0.924 | 0.937 / 0.961 | 0.810 / 0.135 / 0.935 | 0.953 / 0.060 / 0.963 |
| population_gaussian | 0.574 / 0.236 / 0.877 | 0.840 / 0.927 | 0.708 / 0.163 / 0.879 | 0.874 / 0.097 / 0.914 |
| dower_dipole | 0.563 / 0.261 / 0.864 | 0.819 / 0.923 | 0.694 / 0.180 / 0.868 | 0.853 / 0.101 / 0.907 |

Paired bootstrap (2000 resamples over records) of record-median r, physio
minus other method, 95% interval:

| vs | unknown | unknown-shape | missing | shape |
|---|---|---|---|---|
| beat_template | +0.0005 [-0.0005, +0.0016] | +0.0024 [+0.0021, +0.0026] | +0.0001 [-0.0007, +0.0009] | +0.0022 [+0.0020, +0.0024] |
| patient_regression | +0.0080 [+0.0074, +0.0089] | +0.0058 [+0.0054, +0.0063] | +0.0070 [+0.0064, +0.0075] | +0.0058 [+0.0054, +0.0061] |
| spatiotemporal_gauss | +0.018 [+0.017, +0.020] | +0.009 [+0.009, +0.010] | +0.016 [+0.015, +0.018] | +0.009 [+0.008, +0.009] |
| population_gaussian | +0.086 [+0.083, +0.090] | +0.062 [+0.060, +0.064] | +0.087 [+0.084, +0.090] | +0.074 [+0.073, +0.076] |

Reading: physio is significantly better than every method except
`beat_template` on all record-median metrics. Against `beat_template` it is
significantly better on waveform shape (high-passed views), tied on raw
record-median r, and slightly worse on *pooled* raw scores, which are
dominated by baseline wander and a few artefact records (4 records > 10 mV in
fold 10). `beat_template` is also ~14x slower (346 s vs 25 s for 2203
records on 14 workers).

### Comparison with the U-Net notebook (`results/unet_comparison.json`)

`scripts/compare_unet.py` takes the notebook's saved test arrays
(`original.npy`, `reconstructed.npy` from its `data.npz`) and

1. identifies the PTB-XL `ecg_id` of every row by signal fingerprint
   (`results/unet_test_ecg_ids.csv`; 4368/4368 matched);
2. fits physio's population prior only on training-cache records that are
   **not** in the U-Net test set (1612 of 2000);
3. reconstructs the same 4368 records with the same mask and scores both.

```bash
python scripts/compare_unet.py --original original.npy --reconstructed reconstructed.npy -m physio --jobs 8
```

| group | n | method | missing r / RMSE (mV) / rec-median r | shape r / RMSE / rec-median r | Einthoven error (RMS) |
|---|---|---|---|---|---|
| all U-Net test records | 4368 | U-Net | 0.721 / 0.165 / 0.852 | 0.780 / 0.123 / 0.849 | 0.055 mV |
| | | **physio** | **0.808 / 0.145 / 0.972** | **0.914 / 0.073 / 0.988** | 0.0002 mV |
| patients never seen by the U-Net | 3499 | U-Net | 0.726 / 0.159 / 0.851 | 0.810 / 0.107 / 0.849 | 0.055 mV |
| | | **physio** | **0.820 / 0.138 / 0.970** | **0.954 / 0.054 / 0.988** | 0.0002 mV |
| also in PTB-XL fold 10 | 422 | U-Net | 0.791 / 0.135 / 0.849 | 0.828 / 0.102 / 0.842 | 0.056 mV |
| | | **physio** | **0.881 / 0.109 / 0.965** | **0.982 / 0.038 / 0.988** | 0.0002 mV |

Physio is better in every group and metric, and per lead on all missing
leads except V6 (pooled r 0.712 vs 0.719 on all records). The U-Net output
does not satisfy Einthoven/Goldberger (III ≠ II − I etc., about 0.05 mV RMS);
physio satisfies it to the quantisation level of the PTB-XL data. Re-inserting
the observed samples into the U-Net output (`unet+obs` in the JSON) does not
change its missing-sample scores.

Notes on the U-Net evaluation itself:

- Its split is random and record-wise: 869 of its 4368 test records (19.9%)
  belong to patients that also have records in its train/val part. The
  `patients never seen` row removes them.
- PTB-XL contains groups of records with bit-identical signals under
  different `ecg_id`s (e.g. 145/463, 3798/3799; sometimes with different
  `patient_id`s). 16 U-Net test records have an identical twin in its
  train/val part. This also matters for future patient-wise splits.
- Physio has no trained network; the comparison is between a per-record
  physiological model with a weak population prior and a U-Net trained on
  ~15k records. A U-Net retrained on folds 1-8 and evaluated on fold 10
  would give the cleanest comparison.

Caveats:

- All hyperparameters and the method choice were tuned on the validation
  records, so the validation numbers are optimistic. Fold 10 (above) is the
  unbiased check.
- The paper layout is simulated by masking clean digital signals. Scanned or
  printed ECGs add digitisation error and segment misalignment, so these
  results are an upper bound for real paper ECGs.
- PTB-XL comes from a single German collection (1989-1996, Schiller devices),
  has mostly older patients and no ethnicity information. Generalisation to
  other populations and devices has not been tested.

### Comparing with the U-Net notebook

`test-bench-ptb-xl-reconstruct-leads.ipynb` is the original U-Net notebook. It
uses a random **record-wise** split, so records of the same patient can be in
both its training and test sets. Use `scripts/compare_unet.py` (above) for
the comparison: it identifies the records and keeps them out of the
population prior. `scripts/evaluate.py --from-npz data.npz` cannot do that
and is only a quick sanity check (the script warns about this). The cleanest
comparison retrains the U-Net on folds 1-8 and evaluates all methods on
fold 10.

## Repository layout

```
src/ecgrecovery/     package (leads, masking, metrics, data, signal, beats, propagate, methods/)
scripts/             build_cache.py, evaluate.py, compare_unet.py
tests/               pytest suite (synthetic data)
results/             evaluation JSON (results/archive/: older schema)
paper/               BMC Research Notes manuscript (LaTeX, Springer Nature template) and make_figure.py
scratch/             exploratory analysis scripts (outputs are git-ignored)
```

## Data citation and licence

PTB-XL is distributed under CC BY 4.0. Please cite:

- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I.,
  Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available
  electrocardiography dataset. *Scientific Data*, 7, 154.
  https://doi.org/10.1038/s41597-020-0495-6
- Wagner, P., et al. (2020). PTB-XL, a large publicly available
  electrocardiography dataset (version 1.0.1). PhysioNet.
  https://doi.org/10.13026/x4td-x982
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation*, 101(23), e215-e220.

ECGRecovery is distributed under the MIT License. See `LICENSE` for details.

## AI assistance

This repository was produced to a large extent with an AI coding assistant
(Claude Haiku 4.5, Anthropic, used through Claude Code, November 2026), under
the direction of the researcher. The starting point was the researcher's own
dipole edit-propagation model (`ecgrecovery.propagate`, ported and translated to
English) and the U-Net notebook. The AI assistant:

- designed and wrote the package, scripts, tests and this README;
- proposed the candidate reconstruction methods; separate AI sub-agents each
  implemented and tuned one candidate on the validation fold, and the final
  `physio` method combines the best two;
- ran all validation and test experiments and the U-Net comparison;
- carried out code, leakage and methodology reviews with independent AI
  reviewer agents, and applied the fixes.

Under USN's AI policy and the ALLEA Code of Conduct, any resulting
publication must describe this use in the methods section or the
acknowledgements; the AI cannot be an author. The researcher is responsible
for checking the methods, the code and the results before relying on them.
