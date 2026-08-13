# Synthetic phantom — a worked example with no patient data

This example gives the pipeline a case that can be run and inspected without any material from the
study cohort. The geometry and motion are generated entirely from explicit synthetic design
choices; no participant's anatomy, measurement or case parameter enters it at any stage.

That is a stronger position than de-identification, and a verifiable one: `make_phantom.py` has no
external input, so reading it establishes the provenance. It is also deterministic, which is why
the generator is distributed and its output is not — see [What is not committed](#what-is-not-committed).

## Running it

    conda env create -f ../../environment.yml
    conda activate d_view
    python make_phantom.py        # build the phantom            (~5 min, 62 checks)
    python run_stage0a.py         # pick the planes headlessly, scored (~1 min)
    python run_stage0b.py         # track the wall motion, scored      (~1 min)
    python run_stage2.py          # run the morphing on it       (~5 min)

`make_phantom.py` verifies as it goes and exits non-zero if any check fails. Add
`--check-reproducible` to rebuild everything a second time and compare the output byte for byte;
that doubles the run.

Run them in that order. Stage 0b writes the `dispm/` directory that stage 2 then reads, so running
0b changes what stage 2 produces — deliberately, and it is the comparison the example exists to
make. [Accuracy](#accuracy) gives both results and [Stage 0b](#stage-0b) explains why.

On Linux and macOS the four commands above are the whole procedure: the example needs no solvers,
no `config/local.py` and no WSL, so there is nothing platform-specific to arrange.

Windows is the one that needs instructions, because there are two routes and they behave
differently — inside WSL and native. Both are set out step by step in
[RUNNING-ON-WINDOWS.md](RUNNING-ON-WINDOWS.md), with what each has been tested on, and with the
`/mnt/c` performance trap that catches the first attempt.

`run_stage0a.py` runs `select_cut.py` — the script that picks the two control planes — headlessly
against the phantom, then scores its inlet and arch planes against the ones `make_phantom.py` wrote
directly from the same analytic geometry. See [Stage 0a](#stage-0a).

`run_stage0b.py` runs `deformation.py`'s own `main()`, which tracks the ascending-aorta wall motion
across the ten phases and writes `dispm/`. It then scores that tracked field against the analytic
one. See [Stage 0b](#stage-0b).

`run_stage2.py` executes the code cells of `../../wall_def_temp_def.ipynb` in order — it does not
reimplement the pipeline, so there is no second copy of the method to drift out of step. It selects
the case and renders off-screen, which is all a notebook cannot do for itself. It then scores the
reconstruction against the field the phantom was generated from; see [Accuracy](#accuracy).

To run interactively in Jupyter, set `os.environ["AORTA_CASE"] = "PHANTOM"` before executing the
configuration cell. The script above is the supported reproducible route.

Either way the example loads through the ordinary configuration layer, with no `config/local.py`:

```python
from config import load_case
cfg = load_case("PHANTOM")
```

## The phantom

An idealised thoracic aorta: an ascending limb, an arch turning posteriorly and to the left, three
supra-aortic branches and a descending limb — 377.5 mm of centreline, giving the four-outlet
topology (`out`, `out1`, `out2`, `out3`) the notebooks assume.

Every number below is chosen independently of the study cohort:

| Quantity | Value |
|---|---|
| Maximum ascending diameter | 45.0 mm |
| Heart rate | 70 bpm, cycle 0.857 s |
| Peak circumferential strain | 5 % |
| Ascending excursion | 5 mm translation, 3° rotation |
| Cardiac phases | 10, at 0 %…90 % |

Sinotubular, arch and descending diameters, branch sizes and positions, the shape of the temporal
waveform and the weighting of the motion along the vessel are chosen too. `make_phantom.py` marks
every one.

## The motion

The displacement field is analytic, so the exact position of every point at every phase is known in
closed form. It has the two components the manuscript describes:

- a **pressure-driven radial expansion**, greatest in the ascending segment and smaller in the
  stiffer descending aorta;
- a **non-radial motion of the ascending aorta**, a translation and rotation applied in full over
  the ascending segment and decaying to nothing past the arch, where the aorta is tethered.

Both are modulated by one waveform: an asymmetric raised cosine, rising over 0.257 s to the systolic
peak and decaying over the remaining 0.600 s. It is non-negative by construction, so the vessel
never contracts below its reference size, and its derivative vanishes at end-diastole and at peak
systole, as a volume extremum requires.

At 0 % the field is the identity, so that phase is the reference the rest is expressed against. At
the peak the enclosed volume is 8.1 % above it and the root has moved 9.45 mm.

Each phase is surfaced independently, on its own grid, so the ten surfaces share neither vertices
nor connectivity — as ten segmentations of a real acquisition do not. Establishing the
correspondence is the pipeline's job, not the generator's.

## What is generated

```
data/
├── segmentation/
│   ├── Segmentation_AI/     0%_phantom.stl … 90%_phantom.stl
│   ├── cuts_posit.txt       control-plane positions, written directly
│   ├── transformation_details.xlsx   three points on the root, for select_cut.py
│   ├── arc_cut_posit.txt    the arch plane, for select_cut.py
│   └── centerline.vtk       the tracked segment's centreline, for deformation.py
├── mesh-complete/
│   ├── mesh-surfaces/       inlet.vtp, out.vtp, out1-3.vtp, interface.vtp
│   └── mesh-complete.exterior.vtp
├── mesh/                    phantom_Centerlines.vtp
└── reference/
    └── displacement_truth.npz    the analytic field, at every vertex and phase
```

`run_stage0a.py` adds nothing here: it moves `cuts_posit.txt` aside, lets `select_cut.py` write its
own version from the two files above, scores it, then restores the original — so `data/` is exactly
as `make_phantom.py` left it whether or not `run_stage0a.py` has run.

`run_stage0b.py` adds `segmentation/dispm/mw_0.vtp … mw_9.vtp`, the tracked displacement of the
ascending segment at each phase. Stage 2 reads that directory when it is there, so this is the one
output of the three runners that changes a later stage's result.

`run_stage2.py` adds two more: `mesh_def_files/wall_motion_<surface>_phantom.txt`, the prescribed
boundary motion for each of the six surfaces, and `reference/reconstruction.npz`, the reconstructed
field kept beside the truth so the accuracy figures can be recomputed without running stage 2 again.

The boundary surfaces partition the exterior exactly — every triangle belongs to one of them — and
each carries a `GlobalNodeID` indexing the complete surface, as SimVascular writes it and as stage 2
requires.

## What is verified

Sixty-two checks, printed as it runs.

| | Checks | |
|---|---|---|
| 1 | 9 | Centreline: arc length and diameters to 0.1 mm, curvature continuity, no self-intersecting sweep |
| 2 | 14 | Surface: watertight, manifold, genus 0, one body, outward normals, branch diameters to 0.01 mm |
| 3 | 14 | Motion: volume tracks the prescribed strain to 0.001 %, every phase surface sound, phases distinct |
| 4 | 25 | Outputs: `load_case("PHANTOM")` returns, the pipeline's own reader parses `cuts_posit.txt`, boundaries partition the surface, cap radii within a voxel, both control planes sit where the stages that read them require, stage 0a's and 0b's inputs are written |

The cap radii come out 0.23–0.27 mm under specification, uniformly across all five. That is marching
cubes rounding a sharp rim at a 0.6 mm voxel, not an error in the geometry, which is why the check is
in millimetres rather than per cent: the same absolute error is 1.4 % of the inlet and 6.7 % of the
smallest branch.

## Stage 0a

`select_cut.py` picks the two control planes `cuts_posit.txt` holds. It is interactive only when it
has nothing to read: given `transformation_details.xlsx` it derives the inlet plane from three
points on the root instead of prompting for clicks, and given `arc_cut_posit.txt` it takes the arch
plane from the file instead of picking it on a centreline. `make_phantom.py` writes both, so the
script runs headless on this phantom.

`run_stage0a.py` is what actually exercises that path. `cuts_posit.txt` itself is written directly
by `make_phantom.py` from the same analytic geometry, so it cannot be used to demonstrate
`select_cut.py` — the script would just see the file already exists and stop. The runner moves it
aside, lets `select_cut.py` compute its own version, scores the result against the plane it moved
aside, and puts the analytic file back: `cuts_posit.txt` on disk is unaffected by whether
`run_stage0a.py` has ever been run.

The three root points are placed exactly on the inlet circle, so `select_cut.py`'s own cross-product
computation has an exact answer to be scored against, not merely a plausible one. It reproduces it
to floating-point precision:

|  | centre | normal |
|---|---|---|
| inlet | 3×10⁻¹⁶ mm | 0.00003° |
| arch | 0 mm | 0.00000° |

The arch row is not a computation — `select_cut.py` reads it from `arc_cut_posit.txt` and writes it
back unchanged — so its exact match checks the pass-through, not a derivation. The inlet row is the
one with something to get right, and it does.

## Stage 0b

`deformation.py` tracks the ascending-aorta wall across the ten phases: it registers each phase to
the first over a cube about the root, clips to the segment between the two control planes, matches
sixteen points around the inlet ring and sixteen around the arch ring, and interpolates twice — once
for the bodily motion of the segment, once for the radial expansion — before writing `dispm/`.

`run_stage0b.py` runs that, then scores the tracked field against the analytic one. The comparison
is not index-for-index as stage 2's is: `deformation.py` remeshes to a uniform clustering and clips,
so its vertices are not the generator's. The truth is evaluated directly at the tracked mesh's own
points instead, from the closed-form map that generated the phase surfaces.

Against a peak true displacement of 9.327 mm over the tracked segment, at peak systole the tracked
field recovers **59 % of the ascending motion** with a direction cosine of 0.791; the median nodal
error across all phases is 1.202 mm. Reruns reproduce those figures exactly.

**Where the two control planes sit is not a free choice**, and both had to move before this stage
would run at all:

- *The inlet plane cannot sit on the inlet cap.* The phantom is a closed surface, so its inlet is a
  filled disk of triangles; the 0.2 mm slab that isolates the inlet ring then contains that disk
  rather than a rim, and the ray matcher — which accepts a ray when the mean distance to the three
  nearest ring points falls under 10 % of the mean radius — is left deciding on point spacing rather
  than on geometry. On the 0 % phase fifteen of sixteen rays landed and the sixteenth missed by
  1.413 mm against a 1.391 mm tolerance. The plane is placed 3 mm distal instead, which makes the
  ring a true annulus; every ray then lands on every phase.
- *The arch plane has to sit where the vessel is still rising.* `deformation.py` forces that normal
  to point caudally and keeps the half-space it points into, which is the ascending segment only
  while the tangent still points cranially. Past the arch apex the same call keeps the descending
  limb: with the plane at 44 % of the arc length the stage tracked a region whose true motion peaks
  at 0.434 mm rather than the 9.33 mm of the ascending aorta. It sits at 30 % instead — before the
  apex at 37.9 %, and proximal to the brachiocephalic origin at 31.5 %, because a plane distal of
  that cuts the ostium and the ring stops being a circle.

Both are properties of this phantom meeting this method's conventions, and both were fixed by moving
the phantom's planes rather than by altering the method. A hand-picked plane on a patient's aortic
root would not be coplanar with the segmentation's cropped face either, so the first is arguably the
more faithful geometry as well as the workable one.

## Accuracy

`run_stage2.py` compares the reconstructed displacement with the analytic field, node for node.
The comparison is legitimate index-for-index: `mesh-complete.exterior.vtp` is written from the
reference phase surface and the ground truth is stored at those same vertices, in that order.

Because stage 0b can be run on this phantom, stage 2 can be scored twice on identical geometry —
once with the tracked `dispm/` control points and once without them. That is the only difference
between the two columns. Against a peak true displacement of **9.451 mm**, over phases 0–80 %:

| | without `dispm/` | with `dispm/` |
|---|---|---|
| median nodal error | 0.303 mm | 0.299 mm |
| mean nodal error | 1.108 mm (11.7 % of peak) | 0.960 mm (10.2 % of peak) |
| 95th percentile | 4.666 mm | 3.980 mm |
| maximum local error | 27.917 mm | 27.917 mm |

The final phase is excluded: cell 3 of the notebook sets `meshes[-1] = meshes[0]` to force the
cycle closed, so the pipeline never attempts to match it. Scoring it would measure the periodicity
assumption rather than the reconstruction.

**One number hides the result that matters.** The phantom moves in two ways — a radial expansion
everywhere, and a translation and rotation of the ascending aorta that decays past the arch. They
are not recovered alike. At peak systole:

| Region | Nodes | Recovered scale | Direction (cos) | Median error | Mean true motion |
|---|---|---|---|---|---|
| ascending + arch, without `dispm/` | 116 294 | 0.439 | 0.686 | 2.078 mm | 3.715 mm |
| ascending + arch, with `dispm/` | 116 294 | 0.582 | 0.796 | 1.771 mm | 3.715 mm |
| descending, without `dispm/` | 47 642 | 0.796 | 0.852 | 0.010 mm | 0.288 mm |
| descending, with `dispm/` | 47 642 | 0.782 | 0.839 | 0.010 mm | 0.288 mm |

The descending limb, which expands radially and does nothing else, comes back closely either way,
and slightly less well with `dispm/` — the tracked points cover the ascending segment, so they add
information there and only perturb the fit elsewhere.

The ascending aorta is where the difference is. Without the tracked points its displacement comes
back at **44 % of true**, near-constant across phases (0.432–0.480); with them, at **58 %**
(0.570–0.642). So the tracking step supplies about **a third of the ascending motion that ring
matching alone misses**, and a shortfall of roughly 40 % remains after it.

**What this does and does not establish.** Earlier versions of this document could only say that the
44 % shortfall was *consistent with* `dispm/` being what carries the tracked ascending-aorta motion,
because the phantom had no Stage-0b run and the point could not be tested. It can now be tested, and
the answer is that `dispm/` accounts for part of the gap but not the whole of it. Both figures are
properties of this synthetic example — a chosen motion, a chosen geometry, one case — and neither is
a measurement of the clinical results. Anyone comparing these numbers with the manuscript should
check which configuration is meant.

## What is not committed

`data/` is git-ignored: the generator writes 198 MB of it and stage 2 a further 484 MB, and the
generator reproduces its output exactly — so distributing the generator is both the smaller and
the clearer option. Regenerating costs about five minutes.

## Limitations

**This does not validate the pipeline against physiological reality, and must not be presented as
doing so.** It verifies that the implementation recovers a displacement field it was given, on a
geometry with no clinical provenance. The manuscript draws the same distinction for the phantom of
`valente_def_2025`: *"These phantom results verify reconstruction under controlled conditions; they
do not establish the accuracy of the displacement field in the three clinical datasets."*

**Stage 1 is out of scope.** It reads 4D flow DICOM; the phantom is geometry, not an acquisition.

**Stage 0b runs on the phantom, but on a phantom built to suit it.** The ten phase surfaces are
analytic, not segmented from images, and the two control planes are placed by the generator at
stations chosen so that the method's own conventions are met — see [Stage 0b](#stage-0b), where both
constraints and the reason for each are set out. It exercises the tracking code and scores what it
produces; it does not establish that the step behaves this way on acquired cine-CTA, where the
segmentations are noisy and the planes are picked by hand.

**Stage 2 can now be run both ways**, with the tracked `dispm/` and without it, and
[Accuracy](#accuracy) reports both. The gap between them is a measurement of what the tracking step
contributes on this phantom, not of what it contributes clinically.

**Stages 3 and 4 need the solvers** and are not attempted here.
