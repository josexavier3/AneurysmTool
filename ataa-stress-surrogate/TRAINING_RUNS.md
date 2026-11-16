# Training record

These values are read from the frozen evidence run `i07_convert_split_20260821`, whose artefacts are
hashed in that run's `SHA256SUMS`. They supersede the provisional 19--20 August 2026
timings that earlier versions of this file reported.

Host CPU: AMD Ryzen AI 9 HX 370, 12 cores and 24 logical processors. Training was
CPU-only; implicit CUDA use is refused by the pipeline. Native and PyTorch pools were
pinned before any numerical import.

## Primary frozen model

| Field | Value |
|---|---|
| Run name | `cardiac_grouped_free_seed42` |
| Split | grouped, geometry-grouped, seed 42 |
| Loss | `ataa_group_balanced_ghazi_v1` |
| Best epoch | 61 |
| Epochs completed | 77 |
| Best validation loss | 0.0076939 |
| Parameters | 17,353,144 |
| Checkpoint SHA-256 | `94cf2e561159119d7f267b43e6f82c09ba9597eddc4d35018d00e395a43f97b4` |
| Publicly distributed | No -- excluded from GitHub and Zenodo by the author release policy |

## Ensemble

Seeds 42--46 were trained under the identical protocol for the seed-spread and
deep-ensemble analyses. Across those five models the median first-principal-stress
`R2` was 0.9946 +/- 0.0003 and the median MAE
1.19 +/- 0.03 kPa (mean +/- sample standard deviation).
A seed-42 random-split model was trained as a leakage contrast. No trained checkpoint is
distributed publicly.

The random comparator is a seed-42 sample-shuffled diagnostic generated during the rebuild; it is
not a reconstruction of the historical submission split.

## Frozen-run audit notes

The immutable evidence seal is `RUN_COMPLETE.json` together with the run-level `SHA256SUMS`
(131/131 artefacts verified). The run's initial top-level manifest was written before execution
and still carries `status: initialised` and `audited: false`; it is retained unchanged because
editing it would invalidate the seal. Per-model manifests record audited evidence status.

A post-run source comparison found 30 of 34 pipeline-command files byte-identical to the frozen
copies. The four current files that differ -- `build_splits.py`, `split_validation.py`,
`mc_dropout.py`, and `train_cardiac_cycle.py` -- contain subsequent validation/hardening changes.
They do not alter the frozen private checkpoint or the frozen manuscript results. Reproducing those
exact results therefore requires the frozen source snapshots; the public pipeline represents the
maintained, hardened implementation and does not distribute trained weights.
