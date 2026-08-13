"""Small, testable helpers for the inlet velocity boundary condition."""

import numpy as np


def rotate_vectors_to_source(vectors_in_target, rotation_source_to_target):
    """Express target-frame vectors in the source frame of a rigid transform.

    Geometry is mapped as ``x_target = R @ x_source + T``. Vector components
    therefore map back as ``v_source = R.T @ v_target``; translation does not
    apply to vectors. The leading dimensions of the vector array are preserved.
    """
    vectors = np.asarray(vectors_in_target, dtype=float)
    rotation = np.asarray(rotation_source_to_target, dtype=float)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError("vectors_in_target must end in a three-component axis")
    if rotation.shape != (3, 3):
        raise ValueError("rotation_source_to_target must have shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("rotation_source_to_target must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
        raise ValueError("rotation_source_to_target must be a proper rotation")
    return vectors @ rotation


def write_inlet_velocity_file(
    output_file,
    global_node_ids,
    velocity_by_time,
    close_points_mask,
    cycle_duration_s,
):
    """Write an svFSI time-dependent vector boundary-condition file.

    ``velocity_by_time`` has shape ``(n_timesteps, n_points, 3)`` and is expressed
    in mm/s. Values are converted to cm/s. Points flagged by the Boolean mask are
    written as zero at every timestep.
    """
    node_ids = np.asarray(global_node_ids)
    velocities = np.asarray(velocity_by_time, dtype=float)
    mask = np.asarray(close_points_mask)

    if velocities.ndim != 3 or velocities.shape[2] != 3:
        raise ValueError("velocity_by_time must have shape (n_timesteps, n_points, 3)")
    n_timesteps, n_points, _ = velocities.shape
    if n_timesteps < 2:
        raise ValueError("at least two timesteps are required")
    if node_ids.shape != (n_points,):
        raise ValueError("global_node_ids must contain one value per point")
    if mask.dtype.kind != "b" or mask.shape != (n_points,):
        raise ValueError("close_points_mask must be a Boolean value per point")
    if cycle_duration_s <= 0:
        raise ValueError("cycle_duration_s must be positive")

    time_steps = np.linspace(0.0, cycle_duration_s, n_timesteps)
    with open(output_file, "w", encoding="utf-8") as stream:
        stream.write(f"3 {n_timesteps} {n_points}\n\n")
        for time_value in time_steps:
            stream.write(f"{time_value:.4f}\n")
        stream.write("\n")

        for point_index, node_id in enumerate(node_ids):
            stream.write(f"{node_id}\n")
            point_velocity = (
                np.zeros((n_timesteps, 3))
                if mask[point_index]
                else velocities[:, point_index, :] / 10.0
            )
            for vector in point_velocity:
                stream.write(f"{vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}\n")

    return time_steps
