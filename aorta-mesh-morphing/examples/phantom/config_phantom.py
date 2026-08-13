"""Configuration for the synthetic phantom, in the shape load_case() returns.

Reached through the ordinary configuration layer, so the notebooks need no change:

    from config import load_case
    cfg = load_case("PHANTOM")

It is kept here rather than in private ``config/local.py`` deliberately. Nothing
here is measured or patient-derived.

Paths point inside this directory, so the example is self-contained and needs no
config/local.py. Run make_phantom.py to create the files they name.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

# Chosen, and matching make_phantom.py. The round 70-bpm value is synthetic.
# rings_ignored is empty because the phantom has no poorly segmented rings to
# exclude: it was generated rather than segmented.
CASE = {
    "heart_rate_bpm": 70,
    "rings_ignored": [],
    "wall_cycle_closure": "replace_last_with_reference",

    # Trimming planes for cell 5 of stage 2. They are derived from the generated
    # phantom coordinate system rather than copied from a study case.
    #
    # The phantom is built about the origin with the inlet at [0, 0, 0]; the
    # descending limb runs down at y = -34 mm. A split at y = -25 mm separates it
    # from the ascending limb and the arch, and the inlet plane is then applied to
    # the latter alone. Retains 96.2 % of the surface as one connected body.
    "trim_split_origin": [0, -25, 0],
    "trim_split_normal": [0, -1, 0],

    # No floor. The generator ends the descending limb at its outlet, so there is
    # no surplus to trim; the plane is placed clear of the geometry (which reaches
    # z = -95 mm) rather than pretending to cut something.
    "trim_floor_origin": [0, 0, -200],
    "trim_floor_normal": [0, 0, 1],
}

# The identifiers a real case takes from the imaging archive. Here they are just
# the word "phantom", which is what makes the centreline file name resolve.
DATASET_ID = "phantom"
SIM_FOLDER = "phantom"


def load_phantom_case():
    """Return the configuration for the synthetic example.

    The keys are those load_case() builds for a study case, less the ones that
    only mean something for acquired data: there is no imaging directory and no
    4D flow reference, so those are None rather than a path that cannot exist.
    """
    cfg = dict(CASE)
    cfg["case"] = "PHANTOM"
    cfg["dataset_id"] = DATASET_ID
    cfg["sim_folder"] = SIM_FOLDER
    cfg["data_root"] = DATA_ROOT
    cfg["paths"] = {}
    cfg["n_procs"] = 1

    mesh_complete = os.path.join(DATA_ROOT, "mesh-complete")
    cfg["segmentation_dir"] = os.path.join(DATA_ROOT, "segmentation")
    cfg["mesh_surfaces_dir"] = os.path.join(mesh_complete, "mesh-surfaces")
    cfg["mesh_exterior"] = os.path.join(mesh_complete, "mesh-complete.exterior.vtp")
    cfg["centerline"] = os.path.join(DATA_ROOT, "mesh", f"{DATASET_ID}_Centerlines.vtp")
    cfg["cuts_posit"] = os.path.join(cfg["segmentation_dir"], "cuts_posit.txt")
    cfg["deformed_mesh_dir"] = os.path.join(DATA_ROOT, "def_mesh")
    cfg["wall_motion_out_dir"] = os.path.join(DATA_ROOT, "mesh_def_files")
    cfg["inlet_velocity_out"] = None
    cfg["sim_dir"] = DATA_ROOT

    # No imaging: the phantom is geometry, not an acquisition. Stage 1 cannot run
    # against it, and says so rather than failing on a missing directory.
    cfg["imaging_dir"] = None
    cfg["mri_venc_reference"] = None

    cfg["cycle_duration_s"] = 60.0 / cfg["heart_rate_bpm"]

    # Where the analytic displacement is kept, for scoring a reconstruction
    # against the field it was generated from. A study case has no equivalent.
    cfg["reference_dir"] = os.path.join(DATA_ROOT, "reference")
    return cfg
