"""Canonical hashing primitives for ATAA arrays and files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ARRAY_HASH_ALGORITHM = "sha256-v1:dtype.str|repr(tuple(shape))|C-order-bytes"
FILE_HASH_ALGORITHM = "sha256-v1:file-bytes"


def sha256_file(path: str | Path) -> str:
    """Hash file bytes in bounded-memory blocks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash dtype identity, shape and C-order bytes using the versioned ATAA contract."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
