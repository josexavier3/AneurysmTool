"""Validated scaler loading and transforms shared by training-time verification and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScalerBundle:
    input_mean: np.ndarray
    input_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray


def _load_npz_pair(path: Path, first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {first, second}:
            raise ValueError(f"Unexpected scaler arrays in {path}: {archive.files}")
        a = archive[first].copy()
        b = archive[second].copy()
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"Scaler arrays in {path} must be matching one-dimensional vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError(f"Scaler arrays in {path} must be finite with strictly positive scales")
    return a, b


def load_training_scalers(models_dir: str | Path, run_name: str) -> ScalerBundle:
    """Load the exact scalers persisted by a named training run."""

    root = Path(models_dir)
    input_mean, input_scale = _load_npz_pair(
        root / f"{run_name}_input_scaler.npz", "mean", "scale"
    )
    output_mean, output_scale = _load_npz_pair(
        root / f"{run_name}_output_scaler.npz", "mean", "std"
    )
    return ScalerBundle(input_mean, input_scale, output_mean, output_scale)


def standardise_inputs(raw_inputs: np.ndarray, scalers: ScalerBundle) -> np.ndarray:
    values = np.asarray(raw_inputs)
    if values.ndim != 2 or values.shape[1] != scalers.input_mean.size:
        raise ValueError("input matrix is incompatible with the saved input scaler")
    if not np.isfinite(values).all():
        raise ValueError("input matrix contains non-finite values")
    return ((values - scalers.input_mean) / scalers.input_scale).astype(np.float32)


def restore_outputs(standardised_outputs: np.ndarray, scalers: ScalerBundle) -> np.ndarray:
    values = np.asarray(standardised_outputs)
    if values.ndim != 2 or values.shape[1] != scalers.output_mean.size:
        raise ValueError("output matrix is incompatible with the saved output scaler")
    if not np.isfinite(values).all():
        raise ValueError("standardised output matrix contains non-finite values")
    return (values * scalers.output_scale + scalers.output_mean).astype(np.float32)
