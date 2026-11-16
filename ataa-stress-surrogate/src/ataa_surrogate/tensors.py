"""Tensor packing, principal values and physical-unit constants."""

from __future__ import annotations

import numpy as np

from .constants import COMPONENT_ORDER, DYNE_CM2_TO_KPA

def _validated_block(block: np.ndarray, n_nodes: int | None) -> tuple[np.ndarray, int]:
    values = np.asarray(block)
    if values.ndim != 2 or values.shape[1] % 6:
        raise ValueError("tensor block must have shape (samples, 6 * nodes)")
    inferred = values.shape[1] // 6
    if n_nodes is not None and inferred != n_nodes:
        raise ValueError(f"tensor block has {inferred} nodes, expected {n_nodes}")
    if not np.isfinite(values).all():
        raise ValueError("tensor block contains non-finite values")
    return values, inferred


def symmetric_tensors(block: np.ndarray, n_nodes: int | None = None) -> np.ndarray:
    """Unpack ``[11,22,33,12,23,13]`` components into symmetric 3x3 tensors.

    This is VTK Voigt order for Stress. Stored C from ``preProcessing.py`` uses
    ``ataa_surrogate.constants.C_SOURCE_FLAT9_INDICES`` instead, so slot 4 is C21
    rather than C23.
    """

    values, nodes = _validated_block(block, n_nodes)
    packed = values.reshape(values.shape[0], nodes, 6)
    matrices = np.empty((values.shape[0], nodes, 3, 3), dtype=values.dtype)
    matrices[..., 0, 0] = packed[..., 0]
    matrices[..., 1, 1] = packed[..., 1]
    matrices[..., 2, 2] = packed[..., 2]
    matrices[..., 0, 1] = matrices[..., 1, 0] = packed[..., 3]
    matrices[..., 1, 2] = matrices[..., 2, 1] = packed[..., 4]
    matrices[..., 0, 2] = matrices[..., 2, 0] = packed[..., 5]
    return matrices


def symmetric_eigenvalues(matrices: np.ndarray) -> np.ndarray:
    """Eigenvalues of real symmetric 3x3 tensors without a BLAS/LAPACK runtime.

    The closed-form algorithm follows the stable trace-shift formulation for Hermitian
    3x3 matrices. Avoiding ``numpy.linalg`` also prevents the duplicate Intel OpenMP
    runtime carried by some Windows Conda NumPy/PyTorch combinations. Inputs may differ
    from exact symmetry only by a dtype-scaled round-off tolerance; accepted inputs are
    symmetrised before evaluation.
    """

    source = np.asarray(matrices)
    values = np.asarray(source, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError("matrices must end with shape (3, 3)")
    if not np.isfinite(values).all():
        raise ValueError("matrices contain non-finite values")
    transpose = np.swapaxes(values, -1, -2)
    source_epsilon = (
        np.finfo(source.dtype).eps if np.issubdtype(source.dtype, np.inexact) else 0.0
    )
    epsilon = max(float(source_epsilon), float(np.finfo(np.float64).eps))
    scale = np.maximum(np.max(np.abs(values), axis=(-2, -1)), 1.0)
    asymmetry = np.max(np.abs(values - transpose), axis=(-2, -1))
    if np.any(asymmetry > 32.0 * epsilon * scale):
        raise ValueError("matrices must be symmetric within floating-point round-off")
    values = 0.5 * (values + transpose)

    a = values[..., 0, 0]
    d = values[..., 1, 1]
    f = values[..., 2, 2]
    b = values[..., 0, 1]
    c = values[..., 0, 2]
    e = values[..., 1, 2]
    centre = (a + d + f) / 3.0
    p_squared = (
        (a - centre) ** 2
        + (d - centre) ** 2
        + (f - centre) ** 2
        + 2.0 * (b * b + c * c + e * e)
    ) / 6.0
    p = np.sqrt(np.maximum(p_squared, 0.0))
    nonconstant = p > 0
    safe_p = np.where(nonconstant, p, 1.0)

    b00 = (a - centre) / safe_p
    b11 = (d - centre) / safe_p
    b22 = (f - centre) / safe_p
    b01 = b / safe_p
    b02 = c / safe_p
    b12 = e / safe_p
    determinant = (
        b00 * b11 * b22
        + 2.0 * b01 * b02 * b12
        - b00 * b12 * b12
        - b11 * b02 * b02
        - b22 * b01 * b01
    )
    angle = np.arccos(np.clip(determinant / 2.0, -1.0, 1.0)) / 3.0
    largest = centre + 2.0 * p * np.cos(angle)
    smallest = centre + 2.0 * p * np.cos(angle + 2.0 * np.pi / 3.0)
    middle = 3.0 * centre - largest - smallest
    eigenvalues = np.sort(np.stack((smallest, middle, largest), axis=-1), axis=-1)
    eigenvalues[~nonconstant] = np.stack((centre, centre, centre), axis=-1)[~nonconstant]
    return eigenvalues


def first_principal(
    block: np.ndarray,
    *,
    n_nodes: int | None = None,
    scale: float = 1.0,
    batch_size: int = 1500,
) -> np.ndarray:
    """Return the algebraically largest eigenvalue at every node."""

    values, nodes = _validated_block(block, n_nodes)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result = np.empty((values.shape[0], nodes), dtype=np.float32)
    for start in range(0, values.shape[0], batch_size):
        matrices = symmetric_tensors(values[start : start + batch_size], nodes)
        result[start : start + batch_size] = symmetric_eigenvalues(matrices)[..., -1]
    return result * np.float32(scale)


def spd_fraction(block: np.ndarray, *, n_nodes: int | None = None, batch_size: int = 1500) -> float:
    """Fraction of node tensors positive definite above floating-point roundoff."""

    values, nodes = _validated_block(block, n_nodes)
    positive = 0
    for start in range(0, values.shape[0], batch_size):
        eigenvalues = symmetric_eigenvalues(
            symmetric_tensors(values[start : start + batch_size], nodes)
        )
        spectral_scale = np.maximum(np.max(np.abs(eigenvalues), axis=-1), 1.0)
        tolerance = 16.0 * np.finfo(np.float64).eps * spectral_scale
        positive += int(np.all(eigenvalues > tolerance[..., None], axis=-1).sum())
    return positive / (values.shape[0] * nodes)


def split_tensor_fields(outputs: np.ndarray, *, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split concatenated S and C fields, each with six components per node."""

    values = np.asarray(outputs)
    field_width = n_nodes * 6
    if values.ndim != 2 or values.shape[1] != 2 * field_width:
        raise ValueError(f"outputs must have shape (samples, {2 * field_width})")
    return values[:, :field_width], values[:, field_width:]
