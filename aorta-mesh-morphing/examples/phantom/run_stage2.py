"""Run stage 2 — the mesh morphing — against the synthetic phantom, then score it.

    python run_stage2.py

This executes the code cells of ../../wall_def_temp_def.ipynb in order, in one
namespace, exactly as "Run All" would. It does not reimplement the pipeline: there
is no second copy of the method here to drift out of step with the notebook. What it
adds is the two things a script can do and a notebook cannot — select the case
without the file being edited, and render off-screen so the run needs no display.

Working interactively is still the ordinary way to use the notebook. Open it, set
CASE = "PHANTOM" in the configuration cell, and run the cells; the plots are then
worth looking at, which is why they are there.

Because the phantom's displacement field is analytic, the reconstruction can be
compared against the field it was generated from. That comparison is the point of
the example: it makes this a quantitative check rather than a demonstration that
something ran to completion. See the limitations in README.md — it verifies the
implementation, not the physiology.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NOTEBOOK = os.path.join(REPO, "wall_def_temp_def.ipynb")

# Both are read by the notebook's own configuration cell and by PyVista at import,
# so they must be set before it is executed. setdefault, not assignment: a caller
# who has already chosen otherwise is not overridden.
os.environ.setdefault("AORTA_CASE", "PHANTOM")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

sys.path.insert(0, REPO)
os.chdir(REPO)          # the notebook resolves `from config import ...` from here


def run_notebook(path):
    """Execute every code cell of `path` in order, in a single shared namespace.

    Returns the namespace, with one addition: `_deformation_exterior`, a copy of
    `deformation` as it stood the first time the notebook bound it. That is the
    field over mesh-complete.exterior.vtp, which is the one the ground truth is
    stored at. The later cell that writes the per-boundary files rebinds the same
    name once per surface, so by the end of the run it holds whichever surface came
    last — the right array has to be taken while it is still there.
    """
    import numpy as np

    with open(path) as fh:
        cells = json.load(fh)["cells"]

    namespace = {"__name__": "__main__"}
    code_cells = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    exterior = None

    for n, (index, cell) in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        if not source.strip():
            continue
        print(f"\n--- cell {index} ({n}/{len(code_cells)}) "
              f"{'-' * max(0, 46 - len(str(index)))}")
        started = time.time()
        exec(compile(source, f"<{os.path.basename(path)} cell {index}>", "exec"),
             namespace)
        print(f"--- cell {index} done in {time.time() - started:.1f} s")

        if exterior is None and "deformation" in namespace:
            exterior = np.array(namespace["deformation"], copy=True)
            print(f"    captured the exterior-surface field: {exterior.shape}")

    namespace["_deformation_exterior"] = exterior
    return namespace


def score(namespace):
    """Compare the reconstructed displacement with the analytic field.

    The comparison is index-for-index, and legitimately so: the generator writes
    mesh-complete.exterior.vtp from the reference phase surface and stores the
    ground truth at those same vertices, in that order. Both arrays are therefore
    displacement from the reference phase, at one shared set of points.
    """
    import numpy as np

    cfg = namespace["cfg"]
    truth_path = os.path.join(cfg["reference_dir"], "displacement_truth.npz")
    if not os.path.isfile(truth_path):
        print(f"\nno ground truth at {truth_path}; scoring skipped")
        return None

    truth = np.load(truth_path)["displacement_mm"]
    recon = namespace.get("_deformation_exterior")

    if recon is None:
        print("\nthe notebook bound no `deformation`; scoring skipped")
        return None

    # Keep the reconstruction beside the truth it is scored against, so the numbers
    # below can be recomputed, plotted or checked without running stage 2 again.
    saved = os.path.join(cfg["reference_dir"], "reconstruction.npz")
    np.savez_compressed(saved, displacement_mm=recon.astype(np.float32))
    print(f"\nreconstruction saved to {saved}")

    if recon.shape != truth.shape:
        print(f"\nshapes differ: reconstruction {recon.shape}, truth {truth.shape}; "
              "scoring skipped")
        return None

    points = np.load(truth_path)["points_mm"]
    error = np.linalg.norm(recon - truth, axis=2)          # [phase, point], mm
    magnitude = np.linalg.norm(truth, axis=2)
    peak = magnitude.max()

    print("\n" + "=" * 68)
    print("Accuracy against the analytic field")
    print("=" * 68)
    print(f"{'phase':>7}  {'max true':>10}  {'median err':>11}  {'p95 err':>9}  {'max err':>9}")
    for k in range(truth.shape[0]):
        print(f"{k * 10:>6}%  {magnitude[k].max():>9.3f}  "
              f"{np.median(error[k]):>10.3f}  {np.percentile(error[k], 95):>8.3f}  "
              f"{error[k].max():>8.3f}")

    # The last phase is excluded from the summary. Cell 3 of the notebook sets
    # meshes[-1] = meshes[0] to force the cycle closed, so the pipeline never tries
    # to match that phase; scoring it measures the periodicity assumption, not the
    # reconstruction.
    body = error[:-1]
    print("-" * 68)
    print(f"peak true displacement          {peak:.3f} mm")
    print(f"median error, phases 0-80%      {np.median(body):.3f} mm")
    print(f"mean error, phases 0-80%        {body.mean():.3f} mm "
          f"({100.0 * body.mean() / peak:.1f} % of peak)")
    print(f"95th percentile                 {np.percentile(body, 95):.3f} mm")
    print(f"maximum local error             {body.max():.3f} mm")
    print("(the final phase is forced equal to the first, to close the cycle, "
          "and is not scored)")

    # The single number above hides the result that matters. The phantom's motion has
    # two parts: a radial expansion everywhere, and a translation and rotation of the
    # ascending aorta that decays past the arch. Split the surface between them.
    ascending = points[:, 1] > -25.0                       # the cell-5 split plane
    systole = int(np.argmax(magnitude.max(axis=1)))
    print("\nBy region, at peak systole "
          f"(phase {systole * 10}%) — the two motions are not recovered alike:")
    print(f"{'region':<20}{'nodes':>8}{'scale':>8}{'cos':>7}"
          f"{'median err':>12}{'mean |true|':>13}")
    for name, mask in (("ascending + arch", ascending), ("descending", ~ascending)):
        t = truth[systole][mask].reshape(-1)
        r = recon[systole][mask].reshape(-1)
        scale = float(t @ r / (t @ t))
        cos = float(t @ r / (np.linalg.norm(t) * np.linalg.norm(r)))
        e = error[systole][mask]
        print(f"{name:<20}{mask.sum():>8d}{scale:>8.3f}{cos:>7.3f}"
              f"{np.median(e):>12.3f}{np.linalg.norm(truth[systole][mask], axis=1).mean():>13.3f}")
    print("\nThe descending limb, which moves by radial expansion alone, is recovered\n"
          "closely. The ascending aorta, which also translates and rotates, is not:\n"
          "its displacement comes back at roughly 44 % of true. See README.md — this\n"
          "is the example running WITHOUT the dispm/ control points a study case has.")
    return error


def main():
    if not os.path.isfile(NOTEBOOK):
        raise SystemExit(f"stage 2 notebook not found: {NOTEBOOK}")

    case = os.environ["AORTA_CASE"]
    print(f"running {os.path.basename(NOTEBOOK)} for case {case}, off-screen")
    started = time.time()
    namespace = run_notebook(NOTEBOOK)
    print(f"\nnotebook finished in {time.time() - started:.0f} s")

    cfg = namespace["cfg"]
    out_dir = cfg["wall_motion_out_dir"]
    written = sorted(f for f in os.listdir(out_dir) if f.startswith("wall_motion_")) \
        if os.path.isdir(out_dir) else []
    print(f"\nwall-motion files in {out_dir}:")
    for name in written:
        size = os.path.getsize(os.path.join(out_dir, name)) / 1e6
        print(f"  {name}  ({size:.1f} MB)")
    if len(written) != 6:
        raise SystemExit(f"expected 6 wall-motion files, found {len(written)}")

    score(namespace)
    print("\nstage 2 completed on the phantom.")


if __name__ == "__main__":
    main()
