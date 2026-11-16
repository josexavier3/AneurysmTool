"""Small, testable provenance helpers shared by ATAA training entry points."""

from __future__ import annotations

from pathlib import Path
import os
import random
from typing import Any, Mapping

import numpy as np
import torch

from evidence_hashing import (
    ARRAY_HASH_ALGORITHM,
    FILE_HASH_ALGORITHM,
    sha256_array,
    sha256_file,
)


ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create an immutable CPU snapshot of a model state."""

    return {name: value.detach().cpu().clone() for name, value in state.items()}


def set_determinism(
    seed: int,
    deterministic: bool = True,
    *,
    evidence_device: str = "cpu",
    torch_threads: int | None = None,
    torch_interop_threads: int | None = None,
) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch and return the effective settings."""

    if evidence_device != "cpu":
        raise RuntimeError(
            "The audited evidence device policy is CPU-only. CUDA requires a separate validated "
            "policy and must not be selected implicitly."
        )
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "KMP_DUPLICATE_LIB_OK must not be enabled for an evidence run; resolve the duplicate "
            "OpenMP runtime instead of masking it."
        )
    if torch_threads is not None:
        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
    if torch_interop_threads is not None:
        if torch_interop_threads <= 0:
            raise ValueError("torch_interop_threads must be positive")
        if torch.get_num_interop_threads() != torch_interop_threads:
            try:
                torch.set_num_interop_threads(torch_interop_threads)
            except RuntimeError as error:
                raise RuntimeError(
                    "PyTorch inter-op threads were initialised before the evidence policy was "
                    "applied. Launch training through the hardened pipeline."
                ) from error
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed_all": seed if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "cuda_matmul_allow_tf32": bool(
            getattr(getattr(torch.backends.cuda, "matmul", object()), "allow_tf32", False)
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "evidence_device_policy": evidence_device,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def save_input_scaler(path: str | Path, mean: np.ndarray, scale: np.ndarray) -> None:
    np.savez_compressed(
        path,
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
    )


def save_output_scaler(path: str | Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez_compressed(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )


def model_run_name(split: str, variant: str, seed: int, tag: str = "") -> str:
    """Return a seed-explicit artefact stem."""

    stem = f"cardiac_{split}_{variant}_seed{seed}"
    return f"{stem}_{tag}" if tag else stem
