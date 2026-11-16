# Full training and evaluation pipeline

These scripts implement the manuscript analysis from the source CSM arrays. The public demo in
`../src/predict.py` is standalone; the scripts in this directory require the complete data.

## Data boundary

Patient imaging, patient-derived geometries, and patient simulation inputs are not distributed.
They may be shared only under a data-use or protocol agreement and subject to the governing ethics
approval. The large synthetic training arrays are available from the corresponding author on
request. Do not place restricted data inside a public checkout.

## Setup

```bash
python -m pip install -r ../requirements-pipeline.txt
export ATAA_DATA_ROOT=/path/to/csm/dataset
export ATAA_WORK_ROOT=/path/to/scratch
export ATAA_APPROVE_EXTERNAL_WORK_ROOT=1
export ATAA_RUN_ID=replace_with_an_immutable_run_id
```

PowerShell uses `$env:ATAA_DATA_ROOT = 'D:\path\to\dataset'` (and likewise for the other
variables). On Windows, use a short virtual-environment and scratch path if the checkout is deeply
nested; some PyTorch installations otherwise encounter the legacy path-length limit.

`ATAA_DATA_ROOT` must contain `3.Inputs_Outputs/`, `7.PSdatabase/`, and `1.Meshes/`. All generated
arrays, split files, model checkpoints, and metric outputs are written under
`ATAA_WORK_ROOT/runs/ATAA_RUN_ID`. A non-canonical scratch root must be outside the repository and
requires the explicit approval flag shown above.

## Hardened run order

Inspect the complete dependency graph without creating files:

```bash
python scripts/run_pipeline.py --dry-run
```

The evidence rebuild is deliberately gated. Conversion, diagnosis and split creation may run first:

```bash
python scripts/run_pipeline.py --through-step splits
```

Training remains blocked until `audit/data_audit.json` records the source-layout/unit audit and
`audit/loss_policy.json` records the author's validation-only loss decision. The frozen manuscript
run selected `ataa_group_balanced_ghazi_v1`; a new evidence run must record its own prospective
decision rather than inheriting that choice silently. After those gates, resume with `--from-step
train-grouped-seed42`. The command is fail-fast, holds one exclusive run lock, refuses output
collisions, pins the CPU/thread policy and ends with result, privacy and completion audits.

Direct training commands create non-evidence development artefacts unless
`ATAA_AUDITED_EVIDENCE_MODE=1` is set and both gate documents already exist. Finalisation rejects
every model not permanently recorded as an audited evidence run; use the orchestrator for evidence.

Equivalent individual commands, for inspection only, are:

```bash
python scripts/convert_outputs.py
python scripts/diagnose_data.py
python scripts/build_splits.py
python scripts/train_cardiac_cycle.py --split grouped
python scripts/train_cardiac_cycle.py --split random
python scripts/stage4_metrics.py --split grouped
python scripts/stage4_metrics.py --split random
python scripts/ps_transfer.py
python scripts/stage4_principal.py
python scripts/mesh_prep.py
python scripts/boundary_pws.py
python scripts/pressure_dependence.py
python scripts/percentiles.py
python scripts/mc_dropout.py
python scripts/ensemble_uq.py
python scripts/baseline_meanfield.py
python scripts/meanfield_componentwise.py
python scripts/baselines.py
python scripts/make_figures.py
python scripts/make_figures2.py
python scripts/timing_benchmark.py
python scripts/build_results_report.py
python scripts/prepare_release.py
python scripts/audit_release.py
python scripts/finalize_run.py
```

The orchestrator, rather than this abbreviated inspection list, is authoritative for arguments and
ordering. It runs the five grouped seeds 42--46 and the seed-42 sample-shuffled comparator with
seed-explicit immutable filenames. That comparator is a rebuild diagnostic, not the historical
submission split. The released training script predicts `S` and the stored incomplete `C` block
directly; it does not expose an unimplemented constitutive or Cholesky variant.

## Script responsibilities

| Script | Purpose |
|---|---|
| `convert_outputs.py` | Stream the large text output array to a float32 NumPy array. |
| `diagnose_data.py` | Identify non-finite failed-simulation rows. |
| `build_splits.py` | Create geometry-grouped and sample-random comparison splits. |
| `train_cardiac_cycle.py` | Train the direct `S+C` multilayer perceptron. |
| `stage4_metrics.py` | Compute R2, CCC, relative L2, RMSE, MAE, and NMAE. |
| `stage4_principal.py` | Evaluate first-principal stress, peak `S1`, and stored-`C` admissibility. |
| `ps_transfer.py` | Evaluate transfer to imaging-derived geometries. |
| `mc_dropout.py`, `ensemble_uq.py` | Evaluate uncertainty discrimination and component--node coverage. |
| `pressure_dependence.py` | Quantify the descriptive association between absolute error and pressure without pseudo-replicated row-wise p-values. |
| `percentiles.py` | Persist error percentiles and empirical relative-L2 CDFs. |
| `mesh_prep.py`, `boundary_pws.py` | Prepare the boundary mask and common-template peak-node disagreement analysis. |
| `baseline_meanfield.py`, `meanfield_componentwise.py` | Evaluate input-independent mean-field diagnostics. |
| `baselines.py` | Evaluate fixed, untuned POD regressors; kernel ridge uses 6,000 training rows. |
| `make_figures.py`, `make_figures2.py` | Generate analysis figures from persisted outputs. |
| `timing_benchmark.py` | Benchmark one row, one 20-state sequence, and the held-out batch; compare sequence-to-sequence with the inherited CSM time. |
| `build_results_report.py` | Build manuscript values, the claim register, and results report. |
| `prepare_release.py` | Stage the patient-free public component. |
| `audit_release.py` | Scan staged release content for privacy/provenance violations. |
| `finalize_run.py` | Validate completion and write the immutable run marker/checksums. |

For the frozen manuscript run, Gate L compared the former rebuild loss and the group-balanced
Ghazi adaptation on validation data only, kept the grouped test set sealed, and selected
`ataa_group_balanced_ghazi_v1` before evidence training.

## Principal outputs

Each immutable run contains `data/`, `splits/`, `models/`, `outputs/`, `figures/`, `audit/`, and a
staged `release/` component. The final products are `outputs/manuscript_values.json`,
`outputs/claim_traceability.csv`, `outputs/RESULTS_REPORT.md`, `audit/privacy_audit.json`,
`RUN_COMPLETE.json`, and the run-level `SHA256SUMS`. Large arrays, checkpoints, and restricted data
remain outside Git.
