"""Privacy and path-leakage checks for public ATAA release staging."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


RESTRICTED_EXTENSIONS = {
    ".dcm",
    ".dicom",
    ".nii",
    ".nrrd",
    ".mha",
    ".mhd",
    ".vtu",
    ".vtp",
    ".stl",
}
TEXT_EXTENSIONS = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
IDENTIFIER_PATTERNS = (
    re.compile(r"(?i)(?:patient|subject|individual)[-_ ]?\d{1,8}"),
)
MACHINE_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:\\+(?:Users|Mourato|GitHub|Data)\\+[^\s\"']+"),
    re.compile(r"(?i)(?:^|[\s\"'])/(?:home|Users|mnt)/[^\s\"']+"),
)


def _matches(text: str, patterns: Iterable[re.Pattern[str]]) -> list[str]:
    return sorted({match.group(0) for pattern in patterns for match in pattern.finditer(text)})


def scan_public_tree(root: str | Path) -> list[dict[str, str]]:
    """Return public-release privacy findings without reading large binary payloads."""

    release_root = Path(root).resolve()
    findings: list[dict[str, str]] = []
    for path in sorted(release_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(release_root).as_posix()
        if path.suffix.lower() in RESTRICTED_EXTENSIONS:
            findings.append(
                {"path": relative, "kind": "restricted_extension", "detail": path.suffix.lower()}
            )
        filename_matches = _matches(path.name, IDENTIFIER_PATTERNS)
        for match in filename_matches:
            findings.append({"path": relative, "kind": "identifier_in_filename", "detail": match})
        if path.suffix.lower() not in TEXT_EXTENSIONS or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _matches(text, (*IDENTIFIER_PATTERNS, *MACHINE_PATH_PATTERNS)):
            findings.append({"path": relative, "kind": "restricted_text", "detail": match})
    return findings


def scan_provenance_text(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    """Identify machine-specific paths that must not be promoted with public evidence."""

    findings: list[dict[str, str]] = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _matches(text, MACHINE_PATH_PATTERNS):
            findings.append({"path": str(path), "kind": "machine_path", "detail": match})
        for match in _matches(text, IDENTIFIER_PATTERNS):
            findings.append({"path": str(path), "kind": "possible_identifier", "detail": match})
    return findings
