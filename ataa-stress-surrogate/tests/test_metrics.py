"""Deterministic unit tests for the NumPy-only public metric implementation."""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import N_NODES, field_metrics, first_principal  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_first_principal_for_diagonal_tensor(self):
        tensors = np.zeros((1, N_NODES, 6), dtype=np.float32)
        tensors[..., 0] = 1.0
        tensors[..., 1] = 3.0
        tensors[..., 2] = 2.0
        result = first_principal(tensors.reshape(1, -1))
        np.testing.assert_allclose(result, 3.0)

    def test_identical_fields_have_perfect_agreement(self):
        ref = np.linspace(1.0, 2.0, N_NODES, dtype=np.float32)[None, :]
        result = field_metrics(ref.copy(), ref)
        self.assertAlmostEqual(result["R2"], 1.0)
        self.assertAlmostEqual(result["relL2"], 0.0)
        self.assertAlmostEqual(result["MAE"], 0.0)
        self.assertAlmostEqual(result["pearson"], 1.0)


if __name__ == "__main__":
    unittest.main()
