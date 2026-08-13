"""Regression tests for the public/private configuration boundary."""

import importlib
import os
import sys
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(COMPONENT_ROOT))


class ConfigurationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = importlib.import_module("config")
        cls.schema = importlib.import_module("config.cases")

    def test_phantom_loads_without_private_configuration(self):
        cfg = self.config.load_case("PHANTOM")

        self.assertEqual(cfg["case"], "PHANTOM")
        self.assertEqual(cfg["dataset_id"], "phantom")
        self.assertEqual(cfg["heart_rate_bpm"], 70)
        self.assertAlmostEqual(cfg["cycle_duration_s"], 60.0 / 70.0)
        self.assertIsNone(cfg["imaging_dir"])
        self.assertIsNone(cfg["mri_venc_reference"])

    def test_private_case_is_refused_without_local_configuration(self):
        if self.config.PRIVATE_CASES is not None:
            self.skipTest("authorised config/local.py exists on this machine")

        with self.assertRaisesRegex(ImportError, r"config/local\.py not found"):
            self.config.load_case("PRIVATE_CASE")

    def test_unfilled_case_template_is_rejected(self):
        template = {
            field: None for field in self.schema.REQUIRED_CASE_FIELDS
        }

        with self.assertRaisesRegex(ValueError, "template values"):
            self.schema.validate_case("CASE_NAME", template)

    def test_complete_neutral_case_shape_is_accepted(self):
        case = self._complete_neutral_case()

        self.assertEqual(self.schema.validate_case("CASE_NAME", case), case)

    def test_private_case_paths_are_built_from_ignored_settings(self):
        previous = (
            self.config.PRIVATE_CASES,
            self.config.DATA_ROOT,
            self.config.PATHS,
            self.config.N_PROCS,
        )
        self.config.PRIVATE_CASES = {"PRIVATE_CASE": self._complete_neutral_case()}
        self.config.DATA_ROOT = "/authorised/private-data"
        self.config.PATHS = {
            "svfsi_bin": "/opt/private/svFSI",
            "onedsolver_bin": "/opt/private/OneDSolver",
        }
        self.config.N_PROCS = 2
        try:
            cfg = self.config.load_case("PRIVATE_CASE")
        finally:
            (
                self.config.PRIVATE_CASES,
                self.config.DATA_ROOT,
                self.config.PATHS,
                self.config.N_PROCS,
            ) = previous

        self.assertEqual(cfg["case"], "PRIVATE_CASE")
        self.assertEqual(cfg["n_procs"], 2)
        self.assertEqual(
            cfg["segmentation_dir"],
            "/authorised/private-data/segmentation/private-folder",
        )
        self.assertEqual(
            cfg["mri_venc_reference"],
            "/authorised/private-data/imaging/private-folder/mri/"
            "private-series/private-instance.dcm",
        )
        self.assertEqual(
            cfg["inlet_velocity_out"],
            "/authorised/private-data/simulation/private-simulation/"
            "inlet_velocity_vectors.txt",
        )

    @staticmethod
    def _complete_neutral_case():
        return {
            "dataset_id": "private-folder",
            "sim_folder": "private-simulation",
            "venc_series": "private-series",
            "venc_instance": "private-instance.dcm",
            "heart_rate_bpm": 60,
            "systolic_mmHg": 120,
            "diastolic_mmHg": 80,
            "outlet_mean_flow": [1.0, 2.0, 3.0, 4.0],
            "outlet_branch_order": ["out", "out1", "out2", "out3"],
            "rings_ignored": [],
            "mri_translation": [0.0, 0.0, 0.0],
            "mri_rotation_deg": [0.0, 0.0, 0.0],
            "mri_outlet_sections": {
                "out1": {"centerline_index": 10, "radius_mm": 1.0},
                "out2": {"centerline_index": 20, "radius_mm": 1.0},
                "out3": {"centerline_index": 30, "radius_mm": 1.0},
            },
            "wall_cycle_closure": "replace_last_with_reference",
            "stiffness_target_phase_index": 3,
            "prestress_result_step": 100,
            "csm_result_step": 100,
            "trim_split_origin": [0.0, 0.0, 0.0],
            "trim_split_normal": [0.0, 1.0, 0.0],
            "trim_floor_origin": [0.0, 0.0, 0.0],
            "trim_floor_normal": [0.0, 0.0, 1.0],
        }


if __name__ == "__main__":
    unittest.main()
