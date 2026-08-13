# Public release scope

## Purpose

This repository will preserve the methodological code base underpinning the paper without
distributing patient data or patient-derived inputs. A synthetic example is provided to exercise
public code paths. It is not intended to reproduce the patient-specific numerical results in the
paper.

The public/private boundary is based on the source of a value, not only on whether a direct patient
identifier is present. Patient-labelled geometries, coordinates, transformations, physiological
boundary conditions and manually selected case parameters remain private even when pseudonymised.

## What can remain public

All reusable algorithms may remain in the public repository after case-specific literals and paths
have been removed. This includes the stage notebooks and modules, configuration validation,
non-patient solver templates, test code and generic post-processing routines.

The public release separates three kinds of material:

| Material | Public treatment |
|---|---|
| Reusable stage algorithms | Retain and test |
| Parameters needed by any new user | Document through a neutral configuration schema |
| Solver input structure | Retain as clearly marked, non-runnable templates with placeholders |
| Fully synthetic inputs and outputs | Retain or generate through public scripts |
| Patient images and patient-derived geometry | Exclude |
| Study-specific flows, transforms, pressures and manual selections | Exclude |
| Private continuation files and simulation results | Exclude |

Keeping a stage's source code public does not require publishing the clinical inputs used in the
study. Conversely, the presence of source code does not mean that the stage is reproducible from
the public release. Documentation must state that distinction for each stage.

## Current and intended stage coverage

| Stage | Code preserved publicly | Current public execution coverage | Intended release treatment |
|---|---:|---|---|
| 0a: control-plane selection | Yes | Fully runnable and quantitatively checked | Use alongside Stage 2 as a supported example |
| 0b: cine-CTA wall tracking | Yes | Runs on the phantom and is quantitatively checked, on analytic surfaces with generator-placed control planes | Use alongside Stages 0a and 2 as a supported example, stating that the inputs are analytic rather than segmented |
| 1: 4D-flow inlet velocity | Yes | Not covered | Retain generic code and a neutral configuration interface; a runnable example requires wholly synthetic velocity/DICOM-like data |
| 2: wall-motion morphing | Yes | Fully runnable and quantitatively checked | Use as the supported reference example |
| 3: RCR calibration | Yes | Not covered | Retain calibration code plus a synthetic/template 1-D network; execution additionally requires `svOneDSolver` |
| 4: wall-stiffness calibration | Yes | Not covered | Retain calibration code plus non-patient solid/prestress templates; execution additionally requires `svFSI` |
| 3-D moving-wall and FSI solves | Input structure only | Not covered | Publish non-patient templates; do not call them runnable until synthetic meshes and solver prerequisites exist |
| Result post-processing | Where generic | Not currently released as a supported workflow | Add only routines that accept public/synthetic results and have no private paths or case literals |

## Claims supported by the first release

The first release may state that it contains:

- the methodological code base underpinning the paper;
- a configuration interface for private or user-supplied cases;
- non-patient templates showing the required solver-input structure; and
- a deterministic synthetic example that executes and checks Stages 0a, 0b and 2; and
- a measurement, on that example, of how much the Stage-0b tracking contributes to the Stage-2
  reconstruction, obtained by running Stage 2 twice on identical geometry.

It must not state that the public archive:

- reproduces the patient-specific results in the paper;
- contains complete study solver inputs;
- demonstrates the complete end-to-end pipeline;
- validates Stage 1, 3, 4 or the final 3-D simulations unless separate public tests are added; or
- validates Stage 0b against acquired imaging. The phantom exercises that stage on analytic phase
  surfaces, with both control planes placed by the generator at stations chosen to meet the method's
  conventions. What is demonstrated is that the code runs and what it recovers on those inputs, not
  that it behaves so on segmented cine-CTA.

## Route to a complete synthetic demonstration

A future end-to-end synthetic demonstrator is possible, but each input must be generated
independently of the study cohort and its provenance documented. It would require:

1. synthetic reference and time-resolved aortic surfaces, centreline and control planes;
2. a synthetic time-resolved inlet velocity dataset in the coordinate system expected by Stage 1;
3. synthetic fluid and solid volume meshes with named boundary surfaces;
4. a synthetic 1-D outlet network and non-clinical RCR targets;
5. tested `svOneDSolver` and `svFSI` installations; and
6. generic post-processing with expected checks for the synthetic outputs.

Until those assets and solver runs exist, Stage 2 remains the complete runnable public example and
the other stages remain preserved, documented methodological code. Placeholder or mocked solver
outputs may be used for software-interface tests, but must not be described as scientific
end-to-end validation.

## Private study reproduction

Authorised local study reruns use an ignored private configuration and restricted input storage.
Those files are outside the public Git history and the Zenodo archive. The public code and private
configuration should meet through the same documented schema so that removing clinical inputs does
not require maintaining a second implementation.
