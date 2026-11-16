# ATAA stress surrogate

This is the `ataa-stress-surrogate` component of the
[AneurysmTool](https://github.com/josexavier3/AneurysmTool)
repository. It accompanies the
manuscript *A deep neural network surrogate for ascending thoracic aortic aneurysm wall
stress: accuracy, uncertainty and domain transfer*.

The component contains the multilayer-perceptron definition, fitted scalers, a patient-free
synthetic example, and the code used for geometry-grouped training and evaluation. The trained
checkpoint is deliberately not distributed in GitHub or Zenodo; it remains in the frozen private
evidence bundle. The model maps 10 shape, material, and loading variables to a complete nodal
Second Piola--Kirchhoff stress field and a stored, incomplete deformation-tensor block. The inherited arrays
duplicate `C12` where `C23` should appear; it is retained for reproducibility and is diagnostic
only.

## Important limitation

This is a research artefact, not a medical device. Agreement is high on held-out synthetic
geometries, but it degrades on imaging-derived anatomy. It must not be used for patient-specific
clinical decisions.

## Repository contents

```text
ataa-stress-surrogate/
|-- README.md
|-- MODELCARD.md
|-- CITATION.cff
|-- LICENSE
|-- SHA256SUMS
|-- TRAINING_RUNS.md
|-- requirements.txt
|-- requirements-pipeline.txt
|-- data/
|   |-- LICENSE
|   `-- example_synthetic.npz
|-- models/
|   |-- cardiac_grouped_free_input_scaler.npz
|   `-- cardiac_grouped_free_output_scaler.npz
|-- src/
|   |-- model.py
|   |-- metrics.py
|   `-- predict.py
|-- tests/
`-- pipeline/
```

## Use the patient-free example

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/predict.py --checkpoint /path/to/cardiac_grouped_free.pth
```

The command requires a compatible checkpoint produced by the released training pipeline or held
privately by an authorised user. For the frozen private checkpoint (SHA-256
`94cf2e561159119d7f267b43e6f82c09ba9597eddc4d35018d00e395a43f97b4`), the expected
first-principal-stress output for the 50 bundled held-out synthetic loading-state rows was:

```text
R2 = 0.994   relative L2 = 0.053   MAE = 1.16 kPa   Pearson = 0.998
```

Run the deterministic tests with:

```bash
python -m unittest discover -s tests -v
```

Verify the three binary release assets with `SHA256SUMS` (for example,
`sha256sum -c SHA256SUMS` on Linux/macOS).

On Windows, create the virtual environment in a short path if your checkout is deeply nested.
Some PyTorch wheels exceed the legacy path-length limit during installation even though the
component itself is portable.

## Full analysis pipeline

The `pipeline/` directory contains the scripts for the leakage-free split, training, metric
battery, imaging-derived transfer test, percentile/CDF reports, Monte Carlo-dropout and ensemble
uncertainty, pressure dependence, boundary sensitivity, peak first-principal Second
Piola--Kirchhoff stress, fixed reduced-order baselines, figure generation, results bundling,
release staging, privacy audit, and finalisation. Install its dependencies
with `python -m pip install -r requirements-pipeline.txt` and follow `pipeline/README.md`.

The complete CSM arrays, patient-derived inputs, and trained weights are not distributed. The
public example validates the data and tensor interfaces. Model inference can be evaluated only
when the user supplies a compatible checkpoint; reproducing the full manuscript analysis requires
controlled access to the source data and frozen private evidence.

## Data availability and privacy

No clinical images, patient-derived geometries, subject identifiers, or patient boundary
conditions are included. The bundled `.npz` file contains 50 held-out synthetic loading-state
rows only; it does not represent 50 independent geometries or finite-element runs.
Requests for de-identified restricted data may be directed to the corresponding author and are
considered subject to the governing ethics approval.

## Provenance

André Mourato developed the inherited statistical shape model, CSM database, and original
analysis workflow. The current authors performed the geometry-grouped retraining, expanded
evaluation, and preparation of this public component.

## Licence and citation

Code is Apache-2.0 (`LICENSE`). The synthetic example is CC-BY-4.0 (`data/LICENSE`). Citation
metadata are in `CITATION.cff`.
