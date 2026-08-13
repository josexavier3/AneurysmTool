"""Regression tests for consequential numerical helper defects."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(COMPONENT_ROOT))

from SUPORT_def_deformation import calc_cut_posit, find_optimal_point_intrs  # noqa: E402
from inlet_boundary import rotate_vectors_to_source, write_inlet_velocity_file  # noqa: E402


class CentrelineArclengthTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
            ]
        )

    def test_distance_on_first_non_unit_segment(self):
        point, direction = calc_cut_posit(self.points, 1.0)
        np.testing.assert_allclose(point, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])

    def test_distance_accumulates_true_segment_lengths(self):
        point, direction = calc_cut_posit(self.points, 3.0)
        np.testing.assert_allclose(point, [2.0, 1.0, 0.0])
        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0])

    def test_duplicate_points_are_skipped(self):
        points = np.insert(self.points, 1, self.points[0], axis=0)
        point, _ = calc_cut_posit(points, 1.0)
        np.testing.assert_allclose(point, [1.0, 0.0, 0.0])

    def test_distance_beyond_polyline_returns_no_cut(self):
        self.assertEqual(calc_cut_posit(self.points, 6.0), (None, None))

    def test_negative_distance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calc_cut_posit(self.points, -1.0)


class InletMaskTests(unittest.TestCase):
    def test_mask_indexes_each_point_including_indices_above_one(self):
        node_ids = np.array([101, 102, 103])
        velocities = np.array(
            [
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
                [[20.0, 30.0, 40.0], [50.0, 60.0, 70.0], [80.0, 90.0, 100.0]],
            ]
        )
        mask = np.array([False, False, True])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inlet_velocity_vectors.txt"
            times = write_inlet_velocity_file(
                output,
                node_ids,
                velocities,
                mask,
                cycle_duration_s=1.0,
            )
            text = output.read_text(encoding="utf-8")

        np.testing.assert_allclose(times, [0.0, 1.0])
        self.assertIn("101\n1.000000 2.000000 3.000000", text)
        self.assertIn("102\n4.000000 5.000000 6.000000", text)
        self.assertIn("103\n0.000000 0.000000 0.000000\n0.000000 0.000000 0.000000", text)

    def test_non_boolean_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Boolean"):
                write_inlet_velocity_file(
                    Path(directory) / "out.txt",
                    [1, 2],
                    np.zeros((2, 2, 3)),
                    [0, 1],
                    cycle_duration_s=1.0,
                )


class VelocityFrameTests(unittest.TestCase):
    def test_target_vectors_are_rotated_back_to_source_frame(self):
        rotation_source_to_target = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        vectors_in_target = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        actual = rotate_vectors_to_source(vectors_in_target, rotation_source_to_target)

        np.testing.assert_allclose(actual, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_non_rotation_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            rotate_vectors_to_source([[1.0, 0.0, 0.0]], np.diag([2.0, 1.0, 1.0]))


class RayIntersectionTests(unittest.TestCase):
    def test_miss_returns_explicit_none_pair(self):
        class Mesh:
            points = np.array(
                [
                    [-1.0, 10.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [1.0, 10.0, 0.0],
                ]
            )

        point, distance = find_optimal_point_intrs(
            Mesh(),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertIsNone(point)
        self.assertIsNone(distance)


if __name__ == "__main__":
    unittest.main()
