"""Private configuration template.

Copy this file to ``config/local.py`` and fill it only on an authorised research
machine. ``config/local.py`` is Git-ignored and must never be committed. It holds
all clinical paths, identifiers, geometry-derived choices and physiological inputs.

The public synthetic example does not use this file::

    from config import load_case
    cfg = load_case("PHANTOM")

Private data layout expected by the paper workflow::

    <DATA_ROOT>/
      segmentation/<dataset_id>/
        Segmentation_AI/
        cuts_posit.txt
        centerline.vtk
        transformation_details.xlsx
        arc_cut_posit.txt
        dispm/
      simulation/<sim_folder>/
        <dataset_id>_FD-mesh-complete/
          mesh-surfaces/
          mesh-complete.exterior.vtp
        <dataset_id>_SD-mesh-complete/mesh-surfaces/S_interface.vtp
        mesh/<dataset_id>_Centerlines.vtp
        def_mesh/
        <dataset_id>_ROM/solver_1d.in
        <dataset_id>_CSM_calib/
      imaging/<dataset_id>/
        images/
        mri/
"""


DATA_ROOT = r"<absolute path to authorised private data>"

# Duplicate and rename this neutral entry for each authorised local case. Every
# placeholder must be replaced before load_case() will accept the configuration.
# Numerical values shown as None are intentionally absent from the public template.
CASES = {
    "CASE_NAME": {
        "dataset_id": "<private dataset folder>",
        "sim_folder": "<private simulation folder>",
        "venc_series": "<private 4D-flow series folder>",
        "venc_instance": "<private DICOM instance filename>",
        "heart_rate_bpm": None,
        "systolic_mmHg": None,
        "diastolic_mmHg": None,
        "outlet_mean_flow": [None, None, None, None],
        "outlet_branch_order": ["out", "out1", "out2", "out3"],
        "rings_ignored": [],
        "mri_translation": [None, None, None],
        "mri_rotation_deg": [None, None, None],
        # Geometry-dependent selections made on the private centreline. Indices
        # depend on its sampling and cannot be universal public defaults.
        "mri_outlet_sections": {
            "out1": {"centerline_index": None, "radius_mm": None},
            "out2": {"centerline_index": None, "radius_mm": None},
            "out3": {"centerline_index": None, "radius_mm": None},
        },
        # The historical Stage-2 implementation treats the final supplied phase
        # as the periodic endpoint and replaces it with the reference phase.
        "wall_cycle_closure": "<replace_last_with_reference>",
        # Stage-4 file indices depend on the private temporal discretisation and
        # solver decks. They have no safe universal defaults.
        "stiffness_target_phase_index": None,
        "prestress_result_step": None,
        "csm_result_step": None,
        "trim_split_origin": [None, None, None],
        "trim_split_normal": [None, None, None],
        "trim_floor_origin": [None, None, None],
        "trim_floor_normal": [None, None, None],
    },
}

PATHS = {
    "svfsi_bin": "<WSL path to svFSI>",
    "onedsolver_bin": "<WSL path to svOneDSolver>",
}

N_PROCS = 1
