"""Machine-readable result persistence for the ATAA evidence pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from evidence_hashing import (
    ARRAY_HASH_ALGORITHM,
    FILE_HASH_ALGORITHM,
    sha256_array,
    sha256_file,
)


RESULT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write strict JSON, representing non-finite metric degeneracies as null."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_compatible(payload),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, destination)
    return destination


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    """Atomically write deterministic CSV evidence with explicit degeneracy handling."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    serialisable = frame.copy()
    for column in serialisable.columns:
        series = serialisable[column]
        if pd.api.types.is_object_dtype(series.dtype):
            non_missing = series[series.notna()]
            if any(isinstance(value, (numbers.Real, np.number)) for value in non_missing):
                raise TypeError(
                    f"CSV evidence column {column!r} stores numeric values with object dtype; "
                    "coerce it to an explicit numeric dtype before writing."
                )
        elif pd.api.types.is_numeric_dtype(series.dtype):
            serialisable[column] = series.replace([np.inf, -np.inf], np.nan)
    serialisable.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    )
    os.replace(temporary, destination)
    return destination


def write_text(path: str | Path, text: str) -> Path:
    """Atomically write UTF-8 text with normalised LF line endings."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(normalised)
        if normalised and not normalised.endswith("\n"):
            stream.write("\n")
    os.replace(temporary, destination)
    return destination


def finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarise a one-dimensional result array and count metric degeneracies."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    array = array.reshape(-1)
    finite = array[np.isfinite(array)]
    base: dict[str, float | int | None] = {
        "n_total": int(array.size),
        "n_finite": int(finite.size),
        "n_degenerate": int(array.size - finite.size),
    }
    if finite.size == 0:
        return {
            **base,
            "mean": None,
            "sd_ddof1": None,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        **base,
        "mean": float(np.mean(finite)),
        "sd_ddof1": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "minimum": float(np.min(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
    }


def result_document(
    *,
    run_id: str,
    analysis: str,
    model_runs: Iterable[str] = (),
    definitions: Mapping[str, Any] | None = None,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common top-level schema used by every analysis summary."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "analysis": analysis,
        "run_id": run_id,
        "generated_utc": utc_now(),
        "model_runs": list(model_runs),
        "hash_algorithms": {
            "array": ARRAY_HASH_ALGORITHM,
            "file": FILE_HASH_ALGORITHM,
        },
        "definitions": dict(definitions or {}),
        "results": dict(results),
    }


def artifact_record(path: str | Path, run_root: str | Path) -> dict[str, Any]:
    """Return a publishable record for an artefact that must live within the run root."""

    artefact = Path(path).resolve()
    root = Path(run_root).resolve()
    try:
        relative = artefact.relative_to(root)
    except ValueError as error:
        raise ValueError(f"result artefact is outside the declared run root: {artefact}") from error
    return {
        "path": relative.as_posix(),
        "bytes": artefact.stat().st_size,
        "sha256": sha256_file(artefact),
    }


def load_run_manifest(path: str | Path, expected_run_id: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("run_id") != expected_run_id:
        raise RuntimeError(
            f"manifest run_id {manifest.get('run_id')!r} does not match {expected_run_id!r}"
        )
    return manifest


def discover_model_runs(
    manifest: Mapping[str, Any],
    *,
    split: str,
    seeds: Iterable[int],
    variant: str = "free",
    tag: str | None = None,
) -> list[str]:
    """Discover exactly one training run per seed, optionally requiring an exact tag."""

    training_runs = manifest.get("training_runs", {})
    requested = tuple(int(seed) for seed in seeds)
    discovered: list[str] = []
    for seed in requested:
        matches = [
            name
            for name, record in training_runs.items()
            if record.get("split") == split
            and record.get("variant") == variant
            and int(record.get("seed", -1)) == seed
            and (
                tag is None
                or name
                == (
                    f"cardiac_{split}_{variant}_seed{seed}"
                    + (f"_{tag}" if tag else "")
                )
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {split}/{variant} model for seed {seed}, found {matches}"
            )
        discovered.append(matches[0])
    return discovered
