# Solver-input templates

This directory documents the solver interfaces used by the paper code without distributing the
study's patient-derived geometries, boundary values or continuation files.

The original solver decks are not public inputs. They combined reusable solver configuration with
patient-derived vascular geometry, measured/calibrated boundary conditions and private result
paths. Pseudonymising their filenames would not make those contents synthetic.

The tracked `*.template` files therefore preserve only the configuration structure needed to
understand how the pipeline connects to SimVascular solvers:

| Template | Purpose | Relationship to released code |
|---|---|---|
| `rigid_wall_svFSI.inp.template` | Rigid-wall CFD comparison | Generic interface only; the unavailable study deck is not reconstructed |
| `moving_wall_svFSI.inp.template` | Moving-wall CFD used by the paper's method | Consumes the inlet vectors from Stage 1 and six wall-motion files from Stage 2 |
| `fsi_svFSI.inp.template` | Fully coupled FSI comparison | Shows the fluid/solid interface and prestress inputs |
| `solver_1d.in.template` | RCR calibration interface | Contains the four blocks and solver-options line rewritten by Stage 3 |
| `prest_CSM.inp.template` | Structural prestress interface | Its elasticity field is rewritten by Stage 4 |
| `CSM.inp.template` | Structural-cycle interface | Its elasticity field is rewritten by Stage 4 |

## Safety and execution scope

The placeholder tokens use double braces, for example `{{FLUID_VOLUME_MESH}}`. The templates are
**not runnable as distributed** and must not be presented as a synthetic solver validation. A user
must supply their own authorised geometry, physiological boundary conditions and initial/continuation
state, then review every numerical setting for that model.

The rigid-wall template deliberately leaves its solver controls as placeholders because no
authoritative rigid-wall study deck was available. It must not be cited as the historical input.

The runnable public demonstration is [`../examples/phantom/`](../examples/phantom/). It executes the
released Stage-2 notebook and creates all six moving-wall boundary files. It does not claim to run
the 1-D calibration, structural calibration or 3-D solver.

## Moving-wall connection

Stage 2 writes:

```text
wall_motion_inlet.vtp_<case>.txt
wall_motion_out.vtp_<case>.txt
wall_motion_out1.vtp_<case>.txt
wall_motion_out2.vtp_<case>.txt
wall_motion_out3.vtp_<case>.txt
wall_motion_interface.vtp_<case>.txt
```

The moving-wall template applies these six files as general time-dependent Dirichlet conditions
for the mesh equation. Stage 1 writes `inlet_velocity_vectors.txt`, consumed by the fluid inlet.
The four RCR triples are private case inputs rather than public example values.

## Calibration substitutions

`RCR_calibration.ipynb` replaces:

- four `DATATABLE RCR_0` through `RCR_3` blocks; and
- the `SOLVEROPTIONS` line.

`elastic_cycle_calib.ipynb` replaces the `Elasticity modulus:` line in the prestress and structural
cycle inputs. Tests verify that these substitution targets remain present in the templates.

## Reproducing the paper's patient-specific results

That cannot be done from this public archive alone. Clinical images, derived meshes, centrelines,
case transformations, physiological inputs, calibrated boundary values and solver continuation
files are restricted and are not distributed. The repository releases the methodological code
base underpinning the paper and a synthetic execution example, not the patient inputs.
