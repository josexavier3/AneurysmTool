# Model card: ATAA wall-mechanics surrogate

## Model and intended use

The model is a research surrogate for computational solid-mechanics simulations of ascending
thoracic aortic aneurysm wall mechanics. It predicts nodal Second Piola--Kirchhoff stress `S` and
Right Cauchy--Green deformation tensor `C` fields from a compact synthetic geometry, material,
and pressure description. It is intended for methodological research only.

It is not validated against physical wall-stress measurements or clinical outcomes and must not
be used for diagnosis, intervention planning, or other patient-specific decisions.

## Inputs, outputs, and units

- Input: six statistical-shape-model PCA coefficients, wall thickness, Young's modulus,
  diastolic pressure, and instantaneous pressure (10 scalars).
- Output: the six stored entries of `S` and the six stored entries of `C` at 1,354 nodes
  (16,248 scalars). The `S` entries are the six independent components of a symmetric
  tensor. The `C` entries are **not**: the source arrays repeat `C12` in the position that
  should hold `C23`, so `C23` is absent. The defect is inherited from the original
  finite-element extraction, every `C` result is diagnostic only, and the objective pays
  twice for the duplicated entry. The primary endpoint, first principal `S`, is unaffected.
- Stress is stored in dyne/cm2; multiply by `1e-4` for kPa.

Architecture: `10 -> 64 -> 128 -> 256 -> 512 -> 1024 -> dropout(0.2) -> 16248`, with ReLU
activations and a linear output layer.

## Weights and availability

Trained checkpoints are intentionally excluded from GitHub and Zenodo. The primary frozen
checkpoint remains in the private evidence bundle with SHA-256
`94cf2e561159119d7f267b43e6f82c09ba9597eddc4d35018d00e395a43f97b4`; the training
record documents its parameters and provenance. A compatible checkpoint can be reproduced with
the released pipeline when the governed source data are available. The patient-free interface is
then run with `python src/predict.py --checkpoint /path/to/checkpoint.pth`.

## Training-data accounting

The stored synthetic array has 79,626 rows. Exactly 1,400 non-finite rows were excluded. The
78,226 finite rows comprise 3,911 complete 20-state sequences (78,220 rows) and one incomplete
finite six-state sequence. Those six rows occur only in the grouped training partition and do not
affect validation or held-out metrics.

The geometry-grouped split (seed 42) contains 56,466 training rows from 359 geometries, 6,280
validation rows from 40 geometries, and 15,480 test rows from 100 geometries, with zero geometry
overlap. Input and output scaling parameters were fitted to training data only.

## Evaluation summary

Median first-principal-stress results in kPa:

| Evaluation set | R2 | Relative L2 | MAE |
|---|---:|---:|---:|
| Held-out synthetic geometries | 0.995 | 0.048 | 1.16 |
| Imaging-derived geometries | 0.897 | 0.220 | 5.17 |

On imaging-derived rows, peak first-principal Second Piola--Kirchhoff stress (peak `S1`) was
under-predicted by 7.54 kPa on average. The 13.9 mm median is the Euclidean chord distance between
predicted and reference peak-node coordinates on one common template; it is not a patient-specific
anatomical or geodesic localisation error. MC dropout and a five-model ensemble did not reliably
signal this domain shift. Their reported interval coverage pools all stored stress
component--node values, not first-principal stress values. A fixed, untuned POD-regression
baseline was nearly as accurate in the imaging-derived domain; the comparison does not isolate
network depth or regressor class.

## Limitations

- The CSM targets assume a single-layer, isotropic, nearly incompressible neo-Hookean wall,
  idealised loading, uniform thickness, and fixed ends.
- The imaging-derived set is a linked transfer test from the same shape-analysis framework, not an
  independent external cohort.
- Only six shape modes are provided to the network.
- Predictive uncertainty is overconfident under the observed domain shift.
- The `C` field is near identity in these data and is much easier to predict than stress.
- The sample-shuffled comparator was generated during the rebuild and is not a reconstruction of
  the historical submission split.

## Provenance

Inherited statistical shape model, CSM database, and original workflow: André Mourato. Current
geometry-grouped retraining, expanded evaluation, and release preparation: manuscript authors.
