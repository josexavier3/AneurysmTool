"""Configuration loader for private study cases and the public synthetic case.

Paper algorithms use the stable interface::

    from config import load_case
    cfg = load_case("PRIVATE_CASE_LABEL")

Clinical paths, identifiers, geometry-derived selections and physiological values
belong in Git-ignored ``config/local.py``. The tracked repository contains their
schema but no clinical values. ``load_case("PHANTOM")`` selects the self-contained
synthetic example and needs no local configuration.
"""

import os
import subprocess
from collections.abc import Mapping

from .cases import contains_placeholder, validate_case


_NO_LOCAL = (
    "config/local.py not found. Copy config/local_example.py to config/local.py "
    "and fill its private case values and machine paths. The file is Git-ignored "
    "and must never be committed. The public PHANTOM case needs no local file."
)

try:
    from .local import CASES as PRIVATE_CASES
    from .local import DATA_ROOT, N_PROCS, PATHS
except ModuleNotFoundError as exc:  # pragma: no cover - exercised through subprocess test
    if exc.name != f"{__package__}.local":
        raise
    PRIVATE_CASES = DATA_ROOT = PATHS = N_PROCS = None


def to_wsl(windows_path):
    """Translate a Windows path to its WSL equivalent via ``wslpath``."""
    out = subprocess.check_output(["wsl", "wslpath", "-a", str(windows_path)])
    return out.decode().strip()


def _load_phantom():
    """Load the synthetic example from its own directory."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "examples", "phantom", "config_phantom.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"the synthetic example is missing: {path}")
    spec = importlib.util.spec_from_file_location("config_phantom", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_phantom_case()


def _validate_local_settings():
    if not isinstance(PRIVATE_CASES, Mapping) or not PRIVATE_CASES:
        raise ValueError("config/local.py: CASES must be a non-empty mapping")
    if (
        not isinstance(DATA_ROOT, (str, os.PathLike))
        or contains_placeholder(os.fspath(DATA_ROOT))
    ):
        raise ValueError("config/local.py: DATA_ROOT must be a resolved private path")
    if not isinstance(PATHS, Mapping):
        raise ValueError("config/local.py: PATHS must be a mapping")
    required_paths = {"svfsi_bin", "onedsolver_bin"}
    if not required_paths.issubset(PATHS):
        missing = ", ".join(sorted(required_paths - set(PATHS)))
        raise ValueError(f"config/local.py: PATHS is missing: {missing}")
    if contains_placeholder(PATHS):
        raise ValueError("config/local.py: PATHS still contains template values")
    if not isinstance(N_PROCS, int) or N_PROCS < 1:
        raise ValueError("config/local.py: N_PROCS must be a positive integer")


def load_case(label):
    """Return configuration for a private case label or public ``PHANTOM``."""
    if label == "PHANTOM":
        return _load_phantom()
    if PRIVATE_CASES is None:
        raise ImportError(_NO_LOCAL)

    _validate_local_settings()
    if label not in PRIVATE_CASES:
        expected = ", ".join(sorted(map(str, PRIVATE_CASES)))
        raise KeyError(f"unknown private case {label!r}; configured labels: {expected}")

    cfg = validate_case(label, PRIVATE_CASES[label])
    cfg["case"] = label
    cfg["data_root"] = os.fspath(DATA_ROOT)
    cfg["paths"] = dict(PATHS)
    cfg["n_procs"] = N_PROCS

    dataset_id, sim_folder = cfg["dataset_id"], cfg["sim_folder"]
    sim = os.path.join(cfg["data_root"], "simulation", sim_folder)
    mesh_complete = os.path.join(sim, f"{dataset_id}_FD-mesh-complete")

    cfg["segmentation_dir"] = os.path.join(cfg["data_root"], "segmentation", dataset_id)
    cfg["mesh_surfaces_dir"] = os.path.join(mesh_complete, "mesh-surfaces")
    cfg["mesh_exterior"] = os.path.join(mesh_complete, "mesh-complete.exterior.vtp")
    cfg["centerline"] = os.path.join(sim, "mesh", f"{dataset_id}_Centerlines.vtp")
    cfg["cuts_posit"] = os.path.join(cfg["segmentation_dir"], "cuts_posit.txt")
    cfg["deformed_mesh_dir"] = os.path.join(sim, "def_mesh")
    cfg["wall_motion_out_dir"] = os.path.join(sim, "mesh_def_files")
    cfg["inlet_velocity_out"] = os.path.join(sim, "inlet_velocity_vectors.txt")
    cfg["imaging_dir"] = os.path.join(cfg["data_root"], "imaging", dataset_id)
    cfg["sim_dir"] = sim
    cfg["mri_venc_reference"] = os.path.join(
        cfg["imaging_dir"], "mri", cfg["venc_series"], cfg["venc_instance"]
    )
    cfg["cycle_duration_s"] = 60.0 / cfg["heart_rate_bpm"]
    return cfg
