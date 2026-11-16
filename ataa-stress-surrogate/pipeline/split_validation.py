"""Leakage and accounting checks for immutable ATAA split archives."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from evidence_hashing import sha256_file


def save_or_verify_split(path: str | Path, **arrays: np.ndarray) -> str:
    """Create a split once; later runs must reproduce its exact archive bytes."""

    destination = Path(path)
    if not destination.exists():
        np.savez_compressed(destination, **arrays)
        return "created"
    temporary = destination.with_suffix(".verify.npz")
    temporary.unlink(missing_ok=True)
    np.savez_compressed(temporary, **arrays)
    try:
        if sha256_file(temporary) != sha256_file(destination):
            raise RuntimeError(f"Recomputed split differs from frozen archive: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    return "verified_existing"


def load_bad_rows(path: str | Path, n_samples: int) -> np.ndarray:
    """Load the mandatory diagnostic exclusion list and validate its row identities."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Required diagnostic output is missing: {source}. Run diagnose_data.py first."
        )
    bad = np.load(source, allow_pickle=False)
    if bad.ndim != 1 or not np.issubdtype(bad.dtype, np.integer):
        raise ValueError("bad_rows.npy must be a one-dimensional integer array")
    bad = np.asarray(bad, dtype=np.int64)
    if np.unique(bad).size != bad.size:
        raise ValueError("bad_rows.npy contains duplicate row indices")
    if bad.size and (bad.min() < 0 or bad.max() >= n_samples):
        raise ValueError(f"bad_rows.npy contains an index outside [0, {n_samples})")
    return bad


def validate_split_partition(
    split: Mapping[str, np.ndarray],
    valid_indices: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    require_group_disjoint: bool = False,
) -> dict[str, int]:
    """Validate row accounting, uniqueness and optional geometry disjointness."""

    names = ("train", "val", "test")
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(split[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"split {name!r} must be a one-dimensional integer array")
        values = values.astype(np.int64, copy=False)
        if np.unique(values).size != values.size:
            raise ValueError(f"split {name!r} contains duplicate row indices")
        arrays[name] = values

    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = np.intersect1d(arrays[left], arrays[right], assume_unique=True)
            if overlap.size:
                raise ValueError(f"split rows overlap between {left} and {right}: {overlap[:5]}")

    expected = np.unique(np.asarray(valid_indices, dtype=np.int64))
    combined = np.sort(np.concatenate([arrays[name] for name in names]))
    if not np.array_equal(combined, expected):
        missing = np.setdiff1d(expected, combined)
        unexpected = np.setdiff1d(combined, expected)
        raise ValueError(
            "split sample accounting does not equal the valid-row set: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    group_overlap = 0
    if groups is not None:
        group_values = np.asarray(groups)
        if group_values.ndim != 1:
            raise ValueError("groups must be one-dimensional")
        if expected.size and (expected.min() < 0 or expected.max() >= group_values.size):
            raise ValueError("split row index is outside the groups array")
        group_sets = {name: set(group_values[arrays[name]].tolist()) for name in names}
        shared = (
            (group_sets["train"] & group_sets["val"])
            | (group_sets["train"] & group_sets["test"])
            | (group_sets["val"] & group_sets["test"])
        )
        group_overlap = len(shared)
        if require_group_disjoint and shared:
            raise ValueError(
                f"grouped split leaks {len(shared)} geometries across train/val/test"
            )

    return {
        "n_valid": int(expected.size),
        "n_train": int(arrays["train"].size),
        "n_val": int(arrays["val"].size),
        "n_test": int(arrays["test"].size),
        "group_overlap": int(group_overlap),
    }
