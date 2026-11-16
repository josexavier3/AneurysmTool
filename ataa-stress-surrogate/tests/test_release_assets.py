"""Patient-free release integrity, documentation, and inference-interface checks."""
import ast
import hashlib
import importlib.util
import pathlib
import re
import runpy
import subprocess
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseAssetTests(unittest.TestCase):
    def test_synthetic_example(self):
        path = ROOT / "data" / "example_synthetic.npz"
        with np.load(path, allow_pickle=False) as example:
            self.assertEqual(set(example.files), {"inputs", "S_ref", "C_ref"})
            self.assertEqual(example["inputs"].shape, (50, 10))
            self.assertEqual(example["S_ref"].shape, (50, 8124))
            self.assertEqual(example["C_ref"].shape, (50, 8124))
            for key in example.files:
                self.assertEqual(example[key].dtype, np.float32)
                self.assertTrue(np.isfinite(example[key]).all())

    def test_scalers(self):
        with np.load(ROOT / "models" / "cardiac_grouped_free_input_scaler.npz") as archive:
            self.assertEqual(archive["mean"].shape, (10,))
            self.assertEqual(archive["scale"].shape, (10,))
            self.assertTrue(np.all(archive["scale"] > 0))
            self.assertTrue(np.isfinite(archive["mean"]).all())
            self.assertTrue(np.isfinite(archive["scale"]).all())
        with np.load(ROOT / "models" / "cardiac_grouped_free_output_scaler.npz") as archive:
            self.assertEqual(archive["mean"].shape, (16248,))
            self.assertEqual(archive["std"].shape, (16248,))
            self.assertTrue(np.all(archive["std"] > 0))
            self.assertTrue(np.isfinite(archive["mean"]).all())
            self.assertTrue(np.isfinite(archive["std"]).all())

    def test_model_architecture_contract(self):
        model_source = (ROOT / "src" / "ataa_surrogate" / "model.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(model_source)
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"DEFAULT_HIDDEN", "DEFAULT_DROPOUT"}
        }
        constants = runpy.run_path(ROOT / "src" / "ataa_surrogate" / "constants.py")
        self.assertEqual(assignments["DEFAULT_HIDDEN"], (64, 128, 256, 512, 1024))
        self.assertEqual(assignments["DEFAULT_DROPOUT"], 0.2)
        self.assertEqual(constants["IN_DIM"], 10)
        self.assertEqual(constants["OUT_DIM"], 16248)

        # A clean installation includes PyTorch and additionally exercises a real
        # forward pass. Source-only audits still verify the declared contract.
        if importlib.util.find_spec("torch") is not None:
            code = (
                "import sys, torch; "
                "sys.path.insert(0, 'src'); "
                "from ataa_surrogate.model import SurrogateMLP; "
                "output = SurrogateMLP()(torch.zeros((2, 10))); "
                "assert tuple(output.shape) == (2, 16248)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prediction_requires_explicit_user_checkpoint(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "predict.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--checkpoint", result.stderr)
        checkpoints = [
            path.relative_to(ROOT)
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
        ]
        self.assertEqual(checkpoints, [])

    def test_sha256sums_cover_and_match_all_binary_assets(self):
        records = {}
        for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            records[relative.strip()] = digest
        expected = {
            "models/cardiac_grouped_free_input_scaler.npz",
            "models/cardiac_grouped_free_output_scaler.npz",
            "data/example_synthetic.npz",
        }
        self.assertEqual(set(records), expected)
        for relative, expected_digest in records.items():
            self.assertEqual(sha256(ROOT / relative), expected_digest, relative)

    def test_citation_licence_and_documentation_contract(self):
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("cff-version: 1.2.0", citation)
        self.assertIn("license: Apache-2.0", citation)
        self.assertIn(
            "https://github.com/josexavier3/AneurysmTool/tree/main/ataa-stress-surrogate",
            citation,
        )
        self.assertNotIn("Manuscript submitted", citation)
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertTrue((ROOT / "data" / "LICENSE").is_file())

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        normalised_readme = re.sub(r"\s+", " ", readme)
        self.assertIn("50 bundled held-out synthetic loading-state rows", readme)
        self.assertIn("stored, incomplete deformation-tensor block", readme)
        self.assertIn(
            "trained checkpoint is deliberately not distributed", normalised_readme
        )
        self.assertNotIn("50 held-out synthetic simulations", readme)

    def test_public_tree_has_no_restricted_payload_or_absolute_checkout_path(self):
        forbidden_suffixes = {".dcm", ".nii", ".nii.gz", ".vtp", ".vtu"}
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            self.assertFalse(
                any(lower_name.endswith(suffix) for suffix in forbidden_suffixes),
                str(path.relative_to(ROOT)),
            )
            if path.suffix.lower() in {".py", ".md", ".cff", ".txt", ".json", ".yaml", ".yml"}:
                content = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"[A-Za-z]:\\(?:Users|GitHub)\\", content), path)

    def test_pipeline_help_and_dry_run_require_no_private_data(self):
        command = [sys.executable, str(ROOT / "pipeline" / "scripts" / "run_pipeline.py")]
        help_result = subprocess.run(
            [*command, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--dry-run", help_result.stdout)
        dry_result = subprocess.run(
            [*command, "--dry-run"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(dry_result.returncode, 0, dry_result.stderr)
        self.assertIn('"release-staging"', dry_result.stdout)
        self.assertIn('"finalise"', dry_result.stdout)


if __name__ == "__main__":
    unittest.main()
