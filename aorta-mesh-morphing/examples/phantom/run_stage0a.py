"""Run stage 0a — pick the control planes headlessly — against the synthetic
phantom, then score select_cut.py's own plane against the analytic one.

    python run_stage0a.py

make_phantom.py writes two files select_cut.py needs to run without prompting —
transformation_details.xlsx, three points on the aortic root, and
arc_cut_posit.txt, the arch plane — and it also writes cuts_posit.txt directly
from the same analytic geometry, which is the file select_cut.py would otherwise
produce from those two inputs.

This script hides that analytic file, lets select_cut.py compute its own version
headlessly, scores the result against the analytic one, and restores the
analytic file exactly as make_phantom.py wrote it — so cuts_posit.txt as read by
run_stage2.py is unaffected by this script having run, whatever the score.

Run make_phantom.py first; this script does not build the phantom itself.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Both are read at import by select_cut.py's own dependencies (PyVista, and the
# interactive picker it imports but does not use on this path).
os.environ.setdefault("AORTA_CASE", "PHANTOM")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

sys.path.insert(0, REPO)
os.chdir(REPO)          # select_cut.py resolves `from config import ...` from here


def _check(label, ok, detail):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def score(truth, computed):
    """Compare select_cut.py's own inlet and arch planes with the analytic ones.

    Position is scored in millimetres and direction as the angle between the unit
    normals, in degrees, rather than as a raw vector difference — select_cut.py's
    sign convention already resolves which way the inlet normal points, so the
    angle is the meaningful quantity, not a signed component.
    """
    import numpy as np

    passed = []
    for name, (t_centre, t_normal), (c_centre, c_normal) in zip(
            ("inlet", "arch"), truth, computed):
        pos_err = float(np.linalg.norm(c_centre - t_centre))
        cos = float(np.clip(
            (t_normal @ c_normal) / (np.linalg.norm(t_normal) * np.linalg.norm(c_normal)),
            -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos)))
        passed.append(_check(f"{name} centre within 1e-3 mm of the analytic plane",
                             pos_err < 1e-3, f"{pos_err:.2e} mm"))
        passed.append(_check(f"{name} normal within 0.01 deg of the analytic plane",
                             angle_deg < 0.01, f"{angle_deg:.5f} deg"))
    return all(passed)


def main():
    import numpy as np

    from config import load_case
    from SUPORT_def_deformation import read_cuts_posit
    import select_cut

    cfg = load_case("PHANTOM")
    cuts_posit = cfg["cuts_posit"]
    if not os.path.isfile(cuts_posit):
        raise SystemExit(f"{cuts_posit} not found; run make_phantom.py first")

    truth = [(np.array(point), np.array(normal))
             for point, normal in read_cuts_posit(cuts_posit)]

    # select_cut.main() does nothing if cuts_posit.txt already exists, so the
    # analytic file has to be out of the way for the headless route to run at all.
    aside = cuts_posit + ".analytic"
    os.replace(cuts_posit, aside)
    try:
        select_cut.main("PHANTOM")
        if not os.path.isfile(cuts_posit):
            raise SystemExit("select_cut.py did not write cuts_posit.txt")
        computed = [(np.array(point), np.array(normal))
                    for point, normal in read_cuts_posit(cuts_posit)]
    finally:
        os.replace(aside, cuts_posit)

    print("\nStage 0a — select_cut.py run headless from transformation_details.xlsx "
          "and arc_cut_posit.txt, scored against the analytic plane")
    ok = score(truth, computed)
    print("\ncuts_posit.txt restored to the analytic file make_phantom.py wrote; "
          "unaffected by this script having run.")
    if not ok:
        raise SystemExit("\nstage 0a scoring failed")
    print("\nstage 0a completed on the phantom.")


if __name__ == "__main__":
    main()
