"""Checks for the public-release boundary and solver-template contracts."""

import json
import re
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = COMPONENT_ROOT / "solver-inputs"


class PatientInputBoundaryTests(unittest.TestCase):
    def test_study_solver_decks_are_absent(self):
        retired = (
            "CSM.inp",
            "cfd_mw_svFSI.inp",
            "prest_CSM.inp",
            "solver_1d.in",
            "svFSI.inp",
        )
        for name in retired:
            with self.subTest(name=name):
                self.assertFalse((SOLVER_DIR / name).exists())

    def test_private_configuration_is_absent(self):
        self.assertFalse((COMPONENT_ROOT / "config" / "local.py").exists())

        schema = (COMPONENT_ROOT / "config" / "cases.py").read_text(encoding="utf-8")
        self.assertNotRegex(schema, r"(?m)^\s*CASES\s*=")

    def test_notebooks_have_no_saved_outputs(self):
        for path in COMPONENT_ROOT.rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                with self.subTest(notebook=path.name, cell=index):
                    self.assertFalse(cell.get("outputs"))
                    self.assertIsNone(cell.get("execution_count"))


class SolverTemplateTests(unittest.TestCase):
    templates = (
        "rigid_wall_svFSI.inp.template",
        "moving_wall_svFSI.inp.template",
        "fsi_svFSI.inp.template",
        "solver_1d.in.template",
        "prest_CSM.inp.template",
        "CSM.inp.template",
    )

    def test_all_solver_inputs_are_explicit_templates(self):
        self.assertEqual(
            sorted(path.name for path in SOLVER_DIR.glob("*.template")),
            sorted(self.templates),
        )
        for name in self.templates:
            with self.subTest(name=name):
                text = (SOLVER_DIR / name).read_text(encoding="utf-8")
                self.assertIn("Non-runnable", text)
                self.assertIn("{{", text)

    def test_rcr_template_has_expected_substitution_targets(self):
        text = (SOLVER_DIR / "solver_1d.in.template").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"(?m)^DATATABLE RCR_[0-3] LIST$", text)), 4)
        self.assertEqual(len(re.findall(r"(?m)^SOLVEROPTIONS ", text)), 1)

    def test_structural_templates_have_one_elasticity_target_each(self):
        pattern = re.compile(r"(?m)^\s+Elasticity modulus:")
        for name in ("prest_CSM.inp.template", "CSM.inp.template"):
            with self.subTest(name=name):
                text = (SOLVER_DIR / name).read_text(encoding="utf-8")
                self.assertEqual(len(pattern.findall(text)), 1)

    def test_active_drivers_do_not_restore_hard_coded_solver_locations(self):
        drivers = {
            "RCR_calibration.ipynb": (
                "/usr/local/sv/oneDSolver",
                "mpirun -np 10",
                "run_iter2001",
                "pkill -f",
            ),
            "elastic_cycle_calib.ipynb": (
                "CSM_150.vtu",
                "result_500.vtu",
                "\\\\10-procs\\\\",
            ),
        }
        for filename, forbidden in drivers.items():
            text = (COMPONENT_ROOT / filename).read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(filename=filename, value=value):
                    self.assertNotIn(value, text)

    def test_stage_zero_does_not_call_unavailable_vmtk_fallback(self):
        text = (COMPONENT_ROOT / "select_cut.py").read_text(encoding="utf-8")
        self.assertNotIn("cl_s.centerline", text)
        self.assertIn("centerline.vtk", text)


if __name__ == "__main__":
    unittest.main()
