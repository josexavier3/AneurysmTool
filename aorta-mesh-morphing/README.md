# Aorta mesh-morphing haemodynamics pipeline

Methodological code for constructing image-derived moving-wall boundary conditions and calibrating
outlet and wall parameters for thoracic-aorta haemodynamics. This component underpins the manuscript
being prepared for *Computer Methods and Programs in Biomedicine*.

> **Released.** `v1.0.0` is archived at
> [doi:10.5281/zenodo.21925126](https://doi.org/10.5281/zenodo.21925126). Cite that version DOI, or
> the one for whichever release the results came from, rather than the `main` branch, which changes.

## Public release boundary

The reusable algorithms are public. Patient imaging and all patient-derived inputs are not. The
public tree therefore contains:

- the implementation of every methodological stage;
- a neutral configuration schema for authorised private or user-supplied cases;
- non-runnable solver templates containing placeholders rather than study values;
- regression tests for critical numerical and release-safety behaviour; and
- a wholly synthetic, quantitatively checked Stage-2 example.

It does not contain clinical images, patient-derived geometries, physiological boundary values,
case transformations, manual case selections, continuation files or study simulation results. The
archive consequently does not reproduce the patient-specific results in the paper. See
[PUBLIC_RELEASE_SCOPE.md](PUBLIC_RELEASE_SCOPE.md) for the stage-by-stage boundary.

## Pipeline and dependencies

The stage numbers follow the research workflow, but their computational dependencies are not a
simple numerical sequence:

| Stage | Code | Required inputs | Principal output | Public execution status |
|---|---|---|---|---|
| 0a: control planes | `select_cut.py` | Phase-0 surface; saved SimVascular centreline or arch plane | `cuts_posit.txt` | Code retained; no supported public run |
| 0b: cine-CTA tracking | `deformation.py` | Phase surfaces, control planes and ascending centreline | `dispm/mw_<n>.vtp` | Code retained; no supported public run |
| 2: wall morphing | `wall_def_temp_def.ipynb` | Phase surfaces, full centreline, mesh boundaries and optional Stage-0b controls | `def_mesh/` and six wall-motion files | **Synthetic example supported** |
| 1: inlet velocity | `inlet_velocity_MRI_segmentation.ipynb` | 4D-flow data and the deformed meshes created by Stage 2 | `inlet_velocity_vectors.txt` | Code retained; private/user input required |
| 3: RCR calibration | `RCR_calibration.ipynb` | User 1-D network, pressure/flow targets and `svOneDSolver` | Four RCR triples | Code and template retained; not publicly executed |
| 4: stiffness calibration | `elastic_cycle_calib.ipynb` | User solid meshes, prestress/cycle decks and `svFSI` | Effective elastic modulus | Code and templates retained; not publicly executed |
| 3-D simulations | [`solver-inputs/`](solver-inputs/) | User fluid/solid meshes, conditions and solver state | Moving-wall or FSI solution | Placeholder templates only |

For a new acquired case, the effective dependency order is **0a → 0b → 2 → 1**. Stages 3 and 4
are calibration branches whose outputs feed the final solver configurations. Lumen segmentation and
SimVascular mesh/centreline generation are upstream and are not implemented here.

## Requirements

Create the pinned Python environment from this directory:

```bash
conda env create -f environment.yml
conda activate d_view
```

Stages 0–2 and the synthetic example are Python workflows. Stages 3, 4 and the final simulations
also require separately installed SimVascular solvers. The supplied calibration drivers invoke
`svOneDSolver` or `svFSI` through WSL using paths and MPI ranks from private configuration.

Versions used in the study were:

| Tool | Version | Use |
|---|---|---|
| SimVascular | 2023.03.27 | Geometry, meshing and centrelines |
| `svFSI` | 2022.09.26 | Structural calibration and 3-D simulations |
| `svOneDSolver` | 2022-07-22 | RCR calibration |

The solvers are not redistributed. `environment-windows-study.yml` records the original Windows
environment and is not intended as a cross-platform installer.

## Synthetic reference example

The supported public workflow constructs an idealised thoracic aorta from code and executes the
released Stage-2 notebook on it:

```bash
cd examples/phantom
python make_phantom.py
python run_stage2.py
```

No `config/local.py`, clinical dataset, WSL installation or external solver is needed. The first
command runs 59 geometry, motion and interface checks. The second writes all six moving-wall
boundary files and scores the reconstructed displacement against its analytic ground truth.

The most recent verified run reported a 9.451 mm peak true displacement, 0.300 mm median error,
1.110 mm mean error, 4.663 mm 95th percentile and 27.917 mm maximum local error over the scored
phases. The large local error and incomplete recovery of ascending translation/rotation are
important limitations, not validation successes. Read
[`examples/phantom/README.md`](examples/phantom/README.md) before interpreting the figures.

This example proves that the released Stage-2 code path executes on synthetic inputs and provides a
quantitative implementation check. It does not demonstrate the entire pipeline or establish
physiological accuracy.

## Private or user-supplied cases

Copy the neutral template only on an authorised machine:

```bash
cp config/local_example.py config/local.py
```

Fill `CASES`, `DATA_ROOT`, `PATHS` and `N_PROCS`, then select the label explicitly:

```bash
export AORTA_CASE=MY_CASE
```

`config/local.py` is Git-ignored and rejected by the pre-commit hook. It is where every identifier,
clinical value, geometry-dependent selection and machine path belongs. The tracked
`config/cases.py` contains validation only; it contains no study cases.

Stage 0a requires either `arc_cut_posit.txt` or a saved SimVascular ascending centreline at
`<segmentation_dir>/centerline.vtk`. The repository does not silently fall back to unavailable VMTK
code. Stage 1 requires the deformed meshes made by Stage 2, which is why it follows Stage 2 in the
dependency order.

## Solver templates

[`solver-inputs/`](solver-inputs/) contains five `*.template` files. They preserve the interfaces
between the Python stages and the solvers, but they are intentionally non-runnable and contain
`{{PLACEHOLDER}}` tokens. Supplying authorised or synthetic meshes and boundary conditions is the
user's responsibility. The former study decks are excluded because they mixed generic controls
with patient-derived geometry and boundary values.

## Tests

Run the regression suite from this directory:

```bash
python -m unittest discover -s tests -v
```

The tests cover the public/private configuration boundary, centreline arclength, inlet Boolean-mask
handling, explicit ray-intersection failure, notebook output stripping and the solver-template
contracts. They do not replace execution with `svOneDSolver` or `svFSI`.

## Known limitations

- Only Stage 2 currently has a complete public synthetic execution example.
- Stage 0b's ray misses are now explicit hard errors. A future automatic move/drop rule requires a
  documented methodological decision.
- Exact clinical reproduction is unavailable without the restricted study inputs; some historical
  manual selections for two cases were not retained.
- The effect of corrected centreline-arclength and inlet-mask defects on historical simulations must
  be assessed within the authorised private environment.
- Generic post-processing for the final haemodynamic comparison is not yet a supported public
  workflow.

## Licence and citation

Code is licensed under Apache License 2.0; see [`../LICENSE`](../LICENSE) and
[`../NOTICE`](../NOTICE). Citation metadata is in [`../CITATION.cff`](../CITATION.cff), which records
the released version, its date and its DOI. Cite the version DOI corresponding to the code used, not
the mutable `main` branch.
