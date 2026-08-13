# Verification of the tracking step

The synthetic phantom that verifies stage 0 — `select_cut.py` and `deformation.py` — as published
in the earlier paper:

> R. Valente, A. Mourato, A. Carvalho, J. Xavier, M. Brito, S. Avril, A. Tomás, J. Fragata.
> *Patient-Specific In-vivo Dynamic Motion of Ascending Thoracic Aortic Aneurysms from Cine CTA.*
> IEEE Transactions on Biomedical Engineering (2026).
> [doi:10.1109/TBME.2026.3681119](https://doi.org/10.1109/TBME.2026.3681119)

No code was released with that paper. These are the author's notebooks, deposited here for the
first time with the agreement of the co-authors, given in August 2026.

| Notebook | What it does |
|---|---|
| `1_build_phantom.ipynb` | Builds the phantom: a tube swept along a spline through three control points, in a reference state and three deformed ones, and writes the control planes |
| `2_reconstruct.ipynb` | Runs the tracking on it — registration, ring matching, the two RBF interpolations |
| `3_score.ipynb` | Scores the reconstruction: Dice against the prescribed shape, and measured against prescribed radius |

Run them in that order, in one working directory, with `STATE` set to the same value in the second
and the third. `STATE = 'def1'`, `'def2'` or `'def3'` selects which deformed phantom is
reconstructed; the three published Dice and radius values come from running all three.

## The phantom

Parametric, with no patient input at any stage. `create_body()` sweeps a tube of prescribed radius
along a spline through three control points; the three deformed states move those points, change
the radius from 15 mm to 16, 17 and 18, and apply a rotation of −0.22, +0.22 and −0.52 rad.

The first notebook formerly opened a restricted study centreline and displacement surface in a
display-only cell. That cell was removed; none of its variables was read downstream. Nothing
patient-derived remains.

## Two changes made in depositing them

The three notebooks did not run in sequence. Both gaps are closed, and neither required inventing
anything.

**`1_build_phantom.ipynb` now writes `cuts_posit.txt`.** It always computed both control planes and
printed them, but never saved them, and the second notebook reads that file — so the author had
transcribed the printed numbers by hand. The added cell writes the same two planes from the same
variables. Checked against the `cuts_posit.txt` he supplied with the notebooks: the two data rows
are character for character identical.

**`2_reconstruct.ipynb` now reads the surfaces the first notebook writes.** It named
`ATAA_sint1.stl` and `ATAA_sint_def11.stl`, which nothing here produces; his copies of those were
the same shapes at about forty times the point count. He confirmed in August 2026 that this was a
test of the effect of point density, that density does not affect the reconstruction, and that only
the reference mesh really matters. The names now follow `STATE`, and `3_score.ipynb` follows it
too — it had been left pointing at a third state while the second notebook reconstructed the first.

## What runs, and what it reproduces

Verified sequentially on a clean directory under `environment.yml`:

- `1_build_phantom.ipynb` — runs. Two historical preview cells that referenced outputs before they
  existed were converted to explanatory no-ops so that a clean Run All completes.
- `2_reconstruct.ipynb` — runs, and writes the two `.vtp` files the third notebook consumes.
- `3_score.ipynb` — the radius half runs. On `def1` it measures a reconstructed radius of
  **15.986 ± 0.020 mm** against the prescribed 16.0. The author's published literal for the same
  state is 15.976 ± 0.014, so the measurement reproduces to about 0.01 mm on a mesh forty times
  coarser than the one he used.
- The Dice calculation is **skipped** at this mesh density. `trimesh` reports one of the two
  surfaces as not a closed volume after repair; the notebook records that limitation and continues
  to the radius score rather than failing the run.

Pinning `manifold3d` in `environment.yml` was needed to get that far: `trimesh` has no boolean
engine of its own, and without one the Dice cell fails with "No boolean backend" whatever the mesh.

## What it establishes, and one thing it mislabels

The scoring notebook carries its published results as literals, which is how these can be checked.

**Dice**, over the three deformed states: 0.99695, 0.99646, 0.99545 — mean 0.9963, sample standard
deviation 0.0008. The manuscript's 0.996 ± 0.0006 is the same quantity; the smaller figure is the
population standard deviation, which the notebook prints.

**Radius.** The notebook holds two different measures, and they are easy to confuse:

| | Mean | SD |
|---|---|---|
| \|measured − prescribed\| radius, over the three states | 0.023 | 0.015 |
| Within-state scatter of the measured radius | 0.030 | 0.015 |

Measured radii are 15.976, 16.963 and 17.992 mm against prescribed 16, 17 and 18 — an underestimate
in all three states. The accuracy figure is the first row, and it is what a sentence about *the
difference between measured and prescribed* promises. An earlier draft of the manuscript quoted
0.030 ± 0.015 for that difference, which is the second row; it now quotes 0.023 ± 0.015.

## The registration, and the manuscript

`2_reconstruct.ipynb` is the only place in this repository where RANSAC appears:
`registration_ransac_based_on_feature_matching` with FPFH features, `ransac_n=4`, a point-to-*point*
estimator, then `registration_icp` with a point-to-plane estimator initialised from it.

The pipeline does not do this. `deformation.py` calls `point_to_points_reg`, which is
`registration_icp` with a point-to-plane estimator initialised from the identity, and no RANSAC
stage. So RANSAC-then-ICP is a property of this verification, not of the step that produced the
published displacement fields. The manuscript attributed it to the latter; that has been corrected.

## What it does not establish

It verifies the tracking of an ascending segment on a three-point spline, in four states. It has no
arch, no supra-aortic branches and no descending limb, so it exercises neither the four-outlet
topology nor the ten cardiac phases the pipeline assumes, and it cannot drive stage 2. As with the
other example: it verifies reconstruction under controlled conditions, and says nothing about
accuracy in the three clinical datasets.
