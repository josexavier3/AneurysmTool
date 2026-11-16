"""Locate the iso-topological template mesh and extract geometry evidence."""

import glob
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def main() -> None:
    C.init_run()
    node_count = C.N_NODES
    search_dirs = [
        os.path.join(C.DATA_ROOT, "1.Meshes", "7.PCA1", "2.Morphed"),
        os.path.join(C.DATA_ROOT, "1.Meshes", "7.PCA1", "3.Sampled"),
        os.path.join(C.DATA_ROOT, "1.Meshes", "7.PCA1", "1.Registrated"),
        os.path.join(C.DATA_ROOT, "1.Meshes", "6.Morphed"),
    ]

    found = None
    for directory in search_dirs:
        for filename in sorted(glob.glob(os.path.join(directory, "*.vtp")))[:3]:
            try:
                mesh = pv.read(filename)
                print(f"  {os.path.relpath(filename, C.DATA_ROOT)}: {mesh.n_points} points")
                if mesh.n_points == node_count and found is None:
                    found = filename
            except Exception as error:
                print(f"  {filename}: read error {error}")
        if found:
            break

    if found is None:
        raise SystemExit(f"No .vtp with {node_count} points found in the searched dirs.")

    print(f"\nusing template: {os.path.relpath(found, C.DATA_ROOT)}")
    mesh = pv.read(found)
    points = np.asarray(mesh.points, dtype=np.float32)
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    boundary_points = np.asarray(edges.points)
    tree = cKDTree(points)
    _, boundary_indices = tree.query(boundary_points)
    boundary = np.zeros(node_count, dtype=bool)
    boundary[np.unique(boundary_indices)] = True
    print(f"boundary nodes (inlet+outlet rings): {boundary.sum()} of {node_count}")

    np.save(os.path.join(C.OUTPUTS_DIR, "mesh_points.npy"), points)
    np.save(os.path.join(C.OUTPUTS_DIR, "boundary_mask.npy"), boundary)
    print("saved mesh_points.npy, boundary_mask.npy")


if __name__ == "__main__":
    main()
