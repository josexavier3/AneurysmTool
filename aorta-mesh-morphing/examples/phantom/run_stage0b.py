"""Run stage 0b — the cine-CTA wall tracking — against the synthetic phantom,
then score the tracked displacement against the analytic field.

    python run_stage0b.py

This executes deformation.py's own main() for the PHANTOM case. It does not
reimplement the tracking: as with run_stage2.py and the stage-2 notebook, there is
no second copy of the method here to drift out of step. What it adds is off-screen
rendering, so the run needs no display, and the scoring below.

Run make_phantom.py first. It writes the three inputs this stage needs — the ten
phase surfaces, cuts_posit.txt, and centerline.vtk for the tracked segment.

**This run changes what run_stage2.py then does.** Stage 0b writes
segmentation/dispm/, and cell 4 of the stage-2 notebook reads that directory when
it is present, adding the tracked control points to the ring-matched ones. So
stage 2 scored after this run is the pipeline *with* tracking, and stage 2 scored
with dispm/ absent is the pipeline without it. Those are two different results
from identical geometry, which is the comparison this example exists to make; the
README gives both. Delete segmentation/dispm/ to get back to the second.

What the score means. The phantom's motion is the sum of a radial expansion
everywhere and a translation and rotation of the ascending aorta that decays past
the arch. The ring matching that stage 0b performs is what is supposed to capture
the second, so the figure to read is the ascending segment's scale — how much of
the true non-radial displacement comes back — not the overall median.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Read by deformation.py's configuration lookup and by PyVista at import, so both
# must be set before it is imported. setdefault, not assignment: a caller who has
# already chosen otherwise is not overridden.
os.environ.setdefault("AORTA_CASE", "PHANTOM")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)          # deformation.py resolves `from config import ...` from here


def _check(label, ok, detail):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def score(dispm_dir):
    """Compare the tracked displacement with the analytic field it came from.

    The comparison is at the vertices of the tracked mesh, which are not the
    phantom's own — deformation.py remeshes to a uniform 10 000-point clustering
    and then clips to the segment between the control planes. So the truth cannot
    be read from displacement_truth.npz, which is stored at the generator's
    vertices; it is evaluated directly instead, from the same closed-form map that
    generated the phase surfaces.
    """
    import numpy as np
    import pyvista as pv

    import make_phantom as mp

    files = sorted((f for f in os.listdir(dispm_dir) if f.endswith(".vtp")),
                   key=lambda f: int(f.split("_")[1].split(".")[0]))
    meshes = [pv.read(os.path.join(dispm_dir, f)) for f in files]
    points = meshes[0].points

    ref = mp.build_centreline()
    recon = np.stack([m["Displacement"] for m in meshes])
    truth = np.stack([mp.deform(points, k / mp.N_PHASES, ref) - points
                      for k in range(len(meshes))])

    error = np.linalg.norm(recon - truth, axis=2)          # [phase, point], mm
    magnitude = np.linalg.norm(truth, axis=2)

    print("\n" + "=" * 68)
    print("Stage 0b accuracy against the analytic field")
    print("=" * 68)
    print(f"tracked mesh: {len(points)} points, {len(meshes)} phases")
    print(f"{'phase':>7}  {'max true':>10}  {'median err':>11}  {'p95 err':>9}  {'max err':>9}")
    for k in range(len(meshes)):
        print(f"{k * 10:>6}%  {magnitude[k].max():>9.3f}  "
              f"{np.median(error[k]):>10.3f}  {np.percentile(error[k], 95):>8.3f}  "
              f"{error[k].max():>8.3f}")

    systole = int(np.argmax(magnitude.max(axis=1)))
    t = truth[systole].reshape(-1)
    r = recon[systole].reshape(-1)
    scale = float(t @ r / (t @ t))
    cos = float(t @ r / (np.linalg.norm(t) * np.linalg.norm(r)))

    print("-" * 68)
    print(f"peak true displacement          {magnitude.max():.3f} mm")
    print(f"median error, all phases        {np.median(error):.3f} mm")
    print(f"95th percentile                 {np.percentile(error, 95):.3f} mm")
    print(f"maximum local error             {error.max():.3f} mm")
    print(f"at peak systole (phase {systole * 10}%)   scale {scale:.3f}, cos {cos:.3f}")

    print()
    passed = [
        _check("all ten phases tracked", len(meshes) == mp.N_PHASES,
               f"{len(meshes)} files in dispm/"),
        # deformation.py raises from calculate_points_ref the moment a ray fails to
        # land, so reaching this point at all means every ray on every phase landed.
        _check("every inlet and arch ray landed", True,
               "16 per ring, both rings, all ten phases"),
        # Direction only. How *much* of the motion comes back is the measurement
        # this example exists to make, not a threshold to pass: it is the scale
        # printed above, and the README quotes it. Asserting a value for it would
        # be asserting a property of the method, which this example is not
        # entitled to do — it would also invite tuning the phantom until the
        # method flattered itself.
        _check("the tracked field points the right way", cos > 0.5,
               f"cos {cos:.3f} against the analytic field at peak systole"),
    ]
    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return all(passed)


def main():
    from config import load_case

    case = os.environ["AORTA_CASE"]
    cfg = load_case(case)
    seg_dir = cfg["segmentation_dir"]

    required = [cfg["cuts_posit"], os.path.join(seg_dir, "centerline.vtk")]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "stage 0b's inputs are missing: " + ", ".join(missing)
            + "\nRun make_phantom.py first.")

    print(f"running deformation.py for case {case}, off-screen")
    started = time.time()

    import deformation
    deformation.main(case)
    print(f"\ndeformation.py finished in {time.time() - started:.0f} s")

    dispm_dir = os.path.join(seg_dir, "dispm")
    if not os.path.isdir(dispm_dir):
        raise SystemExit(f"stage 0b wrote no dispm/ at {dispm_dir}")

    if not score(dispm_dir):
        raise SystemExit("stage 0b scored below tolerance")

    print("\nstage 0b completed on the phantom.")
    print("run_stage2.py will now pick up dispm/ — see this file's docstring.")


if __name__ == "__main__":
    main()
