"""Central configuration and run isolation for the ATAA evidence rebuild.

Every analysis command must declare ``ATAA_RUN_ID``. Generated arrays, splits,
models, outputs, figures and logs are then confined to one run directory. The
historical source archive and the provisional 19--20 August 2026 artefacts are
inputs only and are never selected as output directories.
"""

from __future__ import annotations

import atexit
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from evidence_hashing import FILE_HASH_ALGORITHM, sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{name} must be one of 1/0, true/false, yes/no or on/off; got {value!r}."
    )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    return _is_within(first, second) or _is_within(second, first)


def _git_snapshot() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "root": str(REPO_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "dirty_paths": None if status is None else status.splitlines(),
    }


def current_git_snapshot() -> dict[str, Any]:
    """Return the current repository identity for immutable orchestration records."""

    return _git_snapshot()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


# --- Source and repository roots -------------------------------------------------

ANALYSIS_DIR = Path(__file__).resolve().parent
PUBLIC_PIPELINE_MODE = ANALYSIS_DIR.name == "pipeline" and (ANALYSIS_DIR.parent / "src").is_dir()
if PUBLIC_PIPELINE_MODE:
    DEFAULT_WORK_ROOT = ANALYSIS_DIR / "work"
    REPO_ROOT = ANALYSIS_DIR.parent
    PUBLIC_RELEASE_ROOT = REPO_ROOT
else:
    DEFAULT_WORK_ROOT = ANALYSIS_DIR.parent
    REPO_ROOT = DEFAULT_WORK_ROOT.parent
    PUBLIC_RELEASE_ROOT = DEFAULT_WORK_ROOT / "release" / "ataa-stress-surrogate"
PUBLIC_SRC_DIR = PUBLIC_RELEASE_ROOT / "src"

DATA_ROOT = Path(
    os.environ.get(
        "ATAA_DATA_ROOT",
        str(ANALYSIS_DIR / "private-dataset-not-configured"),
    )
).expanduser().resolve()

# ``ATAA_WORK_ROOT`` relocates all untracked heavy work. It does not alter the
# repository source/configuration paths.
WORK_ROOT = Path(os.environ.get("ATAA_WORK_ROOT", str(DEFAULT_WORK_ROOT))).expanduser().resolve()
DEFAULT_RUNS_ROOT = (WORK_ROOT / "runs").resolve()
RUNS_ROOT = Path(os.environ.get("ATAA_RUNS_ROOT", str(DEFAULT_RUNS_ROOT))).expanduser().resolve()

# Historical/provisional paths are named explicitly for auditing only. Clean-run
# scripts must use the run-scoped paths declared below.
PROVISIONAL_OUTPUTS_DIR = ANALYSIS_DIR / "outputs"
PROVISIONAL_FIGURES_DIR = ANALYSIS_DIR / "figures"
PROVISIONAL_DATA_CACHE = DEFAULT_WORK_ROOT / "data"
PROVISIONAL_SPLITS_DIR = DEFAULT_WORK_ROOT / "splits"
PROVISIONAL_MODELS_DIR = DEFAULT_WORK_ROOT / "models"

# --- Full cardiac-cycle source data ---------------------------------------------

INPUTS_FULL = DATA_ROOT / "3.Inputs_Outputs" / "sim_inputs_coupled_extended.txt"
OUTPUTS_FULL = DATA_ROOT / "3.Inputs_Outputs" / "sim_outputs_coupled_extended.txt"
PRESTRESS_DIR = DATA_ROOT / "5.Final_trained_models" / "prestress_surrogate_ml"

# --- Problem dimensions ----------------------------------------------------------

N_NODES = 1354
N_COMP = 6
N_INPUTS = 10
N_PCA = 6
SEED = 42

# --- Mandatory run identity ------------------------------------------------------

RUN_ID = os.environ.get("ATAA_RUN_ID", "").strip()
if not RUN_ID:
    raise RuntimeError(
        "ATAA_RUN_ID is required for every analysis command. In PowerShell, set for example: "
        "$env:ATAA_RUN_ID='20260820_clean_evidence'."
    )
if not re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?", RUN_ID):
    raise RuntimeError(
        "ATAA_RUN_ID must be 1--64 lowercase characters, start and end with a letter or digit, "
        "and contain only lowercase letters, digits, '.', '_' or '-'."
    )
_windows_device_names = {"con", "prn", "aux", "nul"}
_windows_device_names.update(f"com{index}" for index in range(1, 10))
_windows_device_names.update(f"lpt{index}" for index in range(1, 10))
if RUN_ID.split(".", 1)[0] in _windows_device_names:
    raise RuntimeError(f"ATAA_RUN_ID uses a reserved Windows device name: {RUN_ID!r}.")

RESUME_COMPLETED = _environment_flag("ATAA_RESUME", default=False)
RESTRICTED_DATA = _environment_flag("ATAA_RESTRICTED_DATA", default=False)
APPROVE_EXTERNAL_RUNS_ROOT = _environment_flag(
    "ATAA_APPROVE_EXTERNAL_RUNS_ROOT", default=False
)
APPROVE_EXTERNAL_WORK_ROOT = _environment_flag(
    "ATAA_APPROVE_EXTERNAL_WORK_ROOT", default=False
)
AUDITED_EVIDENCE_MODE = _environment_flag("ATAA_AUDITED_EVIDENCE_MODE", default=False)
REQUIRE_CLEAN_GIT = _environment_flag("ATAA_REQUIRE_CLEAN_GIT", default=False)
ALLOW_DIRTY_GIT = _environment_flag("ATAA_ALLOW_DIRTY_GIT", default=False)
DIRTY_OVERRIDE_REASON = os.environ.get("ATAA_DIRTY_OVERRIDE_REASON", "").strip()
if ALLOW_DIRTY_GIT and not DIRTY_OVERRIDE_REASON:
    raise RuntimeError(
        "ATAA_ALLOW_DIRTY_GIT=1 requires a non-empty ATAA_DIRTY_OVERRIDE_REASON audit record."
    )

if WORK_ROOT != DEFAULT_WORK_ROOT.resolve():
    if _is_within(WORK_ROOT, REPO_ROOT):
        raise RuntimeError(
            "ATAA_WORK_ROOT may not be relocated inside the tracked repository. Use the "
            "canonical work root or an explicitly approved external scratch root."
        )
    if not APPROVE_EXTERNAL_WORK_ROOT:
        raise RuntimeError(
            "Non-canonical ATAA_WORK_ROOT requires ATAA_APPROVE_EXTERNAL_WORK_ROOT=1 and must "
            "resolve outside the repository."
        )

if RUNS_ROOT != DEFAULT_RUNS_ROOT:
    if _is_within(RUNS_ROOT, REPO_ROOT):
        raise RuntimeError(
            "ATAA_RUNS_ROOT may not be relocated inside the tracked repository. Use the "
            "canonical <work-root>/runs directory or an explicitly approved external scratch root."
        )
    if not APPROVE_EXTERNAL_RUNS_ROOT:
        raise RuntimeError(
            "Non-canonical ATAA_RUNS_ROOT requires ATAA_APPROVE_EXTERNAL_RUNS_ROOT=1 and must "
            "resolve outside the repository."
        )

RUN_ROOT = (RUNS_ROOT / RUN_ID).resolve()
if RUN_ROOT.parent != RUNS_ROOT.resolve():
    raise RuntimeError(f"Resolved run root escapes ATAA_RUNS_ROOT: {RUN_ROOT}")

for forbidden_root in (
    DATA_ROOT,
    REPO_ROOT / "IJNMBE",
    REPO_ROOT / "IJNMBE_R1",
    REPO_ROOT / "IJNMBE_R1_clean",
    PROVISIONAL_OUTPUTS_DIR,
    PROVISIONAL_FIGURES_DIR,
    PROVISIONAL_DATA_CACHE,
    PROVISIONAL_SPLITS_DIR,
    PROVISIONAL_MODELS_DIR,
):
    if _paths_overlap(RUN_ROOT, forbidden_root):
        raise RuntimeError(
            f"Run root overlaps a read-only source or provisional path: {forbidden_root}"
        )

DATA_CACHE = RUN_ROOT / "data"
SPLITS_DIR = RUN_ROOT / "splits"
MODELS_DIR = RUN_ROOT / "models"
OUTPUTS_DIR = RUN_ROOT / "outputs"
FIGURES_DIR = RUN_ROOT / "figures"
LOGS_DIR = RUN_ROOT / "logs"
AUDIT_DIR = RUN_ROOT / "audit"
RELEASE_STAGING_DIR = RUN_ROOT / "release" / "ataa-stress-surrogate"
MANIFEST_PATH = RUN_ROOT / "run_manifest.json"
ENVIRONMENT_PATH = RUN_ROOT / "environment.txt"
PIP_FREEZE_PATH = RUN_ROOT / "pip-freeze.txt"
COMMANDS_LOG_PATH = LOGS_DIR / "commands.log"
COMPLETE_MARKER = RUN_ROOT / "RUN_COMPLETE.json"
RUN_LOCK_PATH = RUNS_ROOT / ".locks" / f"{RUN_ID}.lock"

_INITIALISED = False
_LOCK_OWNER = False
_LOCK_TOKEN: str | None = None


def _current_command_code_hashes() -> dict[str, str]:
    """Hash the invoked script and loaded project modules for this command."""

    roots = (ANALYSIS_DIR.resolve(), PUBLIC_SRC_DIR.resolve())
    paths = {Path(__file__).resolve()}
    invoked = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if invoked is not None and invoked.is_file():
        paths.add(invoked)
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and path.is_file() and any(
            path == root or root in path.parents for root in roots
        ):
            paths.add(path)

    def label(path: Path) -> str:
        for root, prefix in ((ANALYSIS_DIR.resolve(), "analysis"), (PUBLIC_SRC_DIR.resolve(), "src")):
            try:
                return f"{prefix}/{path.relative_to(root).as_posix()}"
            except ValueError:
                continue
        return str(path)

    return {label(path): sha256_file(path) for path in sorted(paths)}


def _initial_manifest() -> dict[str, Any]:
    package_versions = {
        name: _version(distribution)
        for name, distribution in {
            "torch": "torch",
            "numpy": "numpy",
            "scipy": "scipy",
            "scikit_learn": "scikit-learn",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "vtk": "vtk",
            "joblib": "joblib",
        }.items()
    }
    code_hashes = _current_command_code_hashes()

    return {
        "schema_version": 1,
        "status": "initialised",
        "run_id": RUN_ID,
        "created_utc": _utc_now(),
        "resume_completed_requested": RESUME_COMPLETED,
        "repository": _git_snapshot(),
        "invocation": {
            "argv": sys.argv,
            "command": subprocess.list2cmdline(sys.argv) if os.name == "nt" else shlex.join(sys.argv),
            "working_directory": os.getcwd(),
        },
        "paths": {
            "data_root": str(DATA_ROOT),
            "work_root": str(WORK_ROOT),
            "runs_root": str(RUNS_ROOT),
            "run_root": str(RUN_ROOT),
            "data_cache": str(DATA_CACHE),
            "splits": str(SPLITS_DIR),
            "models": str(MODELS_DIR),
            "outputs": str(OUTPUTS_DIR),
            "figures": str(FIGURES_DIR),
            "logs": str(LOGS_DIR),
            "audit": str(AUDIT_DIR),
            "release_staging": str(RELEASE_STAGING_DIR),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "package_versions": package_versions,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "gpu_and_torch_runtime": "pending training initialisation (I02)",
        },
        "seeds_and_determinism": {
            "default_seed": SEED,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "training_runtime": "pending training initialisation (I02)",
        },
        "source_inputs": {
            "inputs_full": str(INPUTS_FULL),
            "outputs_full": str(OUTPUTS_FULL),
            "hashes": "pending source-hash iteration I06",
        },
        "code_hashes_at_initialisation": code_hashes,
        "hash_algorithms": {"file": FILE_HASH_ALGORITHM},
        "scientific_layout": {
            "n_inputs": N_INPUTS,
            "n_pca_geometry_columns": N_PCA,
            "n_nodes": N_NODES,
            "symmetric_components_per_tensor": N_COMP,
            "target_columns": N_NODES * 12,
            "tensor_packing": ["11", "22", "33", "12", "23", "13"],
            "stress_source_unit": "pending I06 source audit; dyne/cm^2 is provisional",
            "stress_report_unit": "kPa only if the I06 source-unit audit confirms conversion",
            "stress_to_kpa": "provisional 1e-4 pending I06 source audit",
            "masking_definitions": "pending boundary audit (I04/I09)",
        },
        "training_configuration": "pending training integration (I02/I03)",
        "artifact_policy": {
            "restricted_data": RESTRICTED_DATA,
            "canonical_runs_root": str(DEFAULT_RUNS_ROOT),
            "external_runs_root_approved": APPROVE_EXTERNAL_RUNS_ROOT,
            "external_work_root_approved": APPROVE_EXTERNAL_WORK_ROOT,
            "clean_git_required": REQUIRE_CLEAN_GIT,
            "audited_evidence_mode": AUDITED_EVIDENCE_MODE,
            "dirty_git_override_requested": ALLOW_DIRTY_GIT,
            "dirty_git_override_reason": DIRTY_OVERRIDE_REASON or None,
            "run_outputs_private": True,
            "public_release_staging": str(RELEASE_STAGING_DIR),
            "author_approved_promotion_target": str(
                PUBLIC_RELEASE_ROOT
            ),
            "provisional_evidence_is_input_only": True,
        },
    }


def _read_lock() -> dict[str, Any] | None:
    try:
        with RUN_LOCK_PATH.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _release_run_lock() -> None:
    global _LOCK_OWNER, _LOCK_TOKEN
    if not _LOCK_OWNER or _LOCK_TOKEN is None:
        return
    record = _read_lock()
    if record is not None and record.get("token") == _LOCK_TOKEN:
        try:
            RUN_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
    if os.environ.get("ATAA_RUN_LOCK_TOKEN") == _LOCK_TOKEN:
        os.environ.pop("ATAA_RUN_LOCK_TOKEN", None)
    _LOCK_OWNER = False
    _LOCK_TOKEN = None


def _acquire_run_lock() -> None:
    """Acquire the cross-platform per-run lock or join an authorised child process."""

    global _LOCK_OWNER, _LOCK_TOKEN
    inherited = os.environ.get("ATAA_RUN_LOCK_TOKEN")
    existing = _read_lock()
    if inherited and existing is not None and existing.get("token") == inherited:
        _LOCK_TOKEN = inherited
        _LOCK_OWNER = False
        return

    token = secrets.token_hex(24)
    record = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "created_utc": _utc_now(),
        "token": token,
    }
    RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(RUN_LOCK_PATH, flags, 0o600)
    except FileExistsError as error:
        owner = _read_lock()
        raise RuntimeError(
            f"Run {RUN_ID!r} is locked by another command: {owner or str(RUN_LOCK_PATH)}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _LOCK_TOKEN = token
    _LOCK_OWNER = True
    os.environ["ATAA_RUN_LOCK_TOKEN"] = token
    atexit.register(_release_run_lock)


def _write_environment_files(manifest: dict[str, Any]) -> None:
    environment = manifest["environment"]
    with ENVIRONMENT_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"captured_utc={manifest['created_utc']}\n")
        stream.write(f"run_id={RUN_ID}\n")
        stream.write(f"python_executable={environment['python_executable']}\n")
        stream.write(f"python_version={environment['python_version']}\n")
        stream.write(f"platform={environment['platform']}\n")
        stream.write(f"processor={environment['processor']}\n")
        stream.write(f"logical_cpu_count={environment['logical_cpu_count']}\n")
        for name, version in environment["package_versions"].items():
            stream.write(f"{name}={version}\n")
        for name, value in environment["thread_environment"].items():
            stream.write(f"{name}={value}\n")

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        content = completed.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        content = f"ERROR: unable to capture pip freeze: {error}\n"
    with PIP_FREEZE_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def init_run() -> None:
    """Explicitly initialise one command and hold its run lock until process exit."""

    global _INITIALISED
    if _INITIALISED:
        return

    if COMPLETE_MARKER.exists():
        raise RuntimeError(
            f"Run {RUN_ID!r} is marked complete at {COMPLETE_MARKER}. Refusing to modify it. "
            "Completed evidence is immutable; ATAA_RESUME cannot reopen it."
        )
    if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()) and not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Run directory exists and is non-empty but has no manifest: {RUN_ROOT}. "
            "ATAA_RESUME never adopts an unmanifested directory; choose a fresh ATAA_RUN_ID or "
            "move the unknown directory aside after manual inspection."
        )

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _acquire_run_lock()
    try:
        for directory in (
            DATA_CACHE,
            SPLITS_DIR,
            MODELS_DIR,
            OUTPUTS_DIR,
            FIGURES_DIR,
            LOGS_DIR,
            AUDIT_DIR,
            RELEASE_STAGING_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        repository = _git_snapshot()
        if repository["git_commit"] is None or repository["dirty"] is None:
            raise RuntimeError(
                "Git commit/dirty provenance is unavailable; refusing to initialise evidence work."
            )
        if REQUIRE_CLEAN_GIT and repository["dirty"] and not ALLOW_DIRTY_GIT:
            raise RuntimeError(
                "The final evidence pipeline requires a clean Git worktree. Commit/stash changes "
                "or set ATAA_ALLOW_DIRTY_GIT=1 only for an explicit audited dry-run override."
            )

        if not MANIFEST_PATH.exists():
            manifest = _initial_manifest()
            manifest["repository"] = repository
            _atomic_json(MANIFEST_PATH, manifest)
            _write_environment_files(manifest)
        else:
            with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("run_id") != RUN_ID:
                raise RuntimeError(
                    f"Manifest run_id {manifest.get('run_id')!r} does not match "
                    f"ATAA_RUN_ID {RUN_ID!r}."
                )
        command_record = {
            "timestamp_utc": _utc_now(),
            "argv": list(sys.argv),
            "working_directory": os.getcwd(),
            "repository": repository,
            "code_hashes": _current_command_code_hashes(),
        }
        manifest.setdefault("command_provenance", []).append(command_record)
        manifest["repository"] = repository
        manifest.setdefault("artifact_policy", {}).update(
            {
                "clean_git_required": REQUIRE_CLEAN_GIT,
                "audited_evidence_mode": AUDITED_EVIDENCE_MODE,
                "dirty_git_override_requested": ALLOW_DIRTY_GIT,
                "dirty_git_override_reason": DIRTY_OVERRIDE_REASON or None,
            }
        )
        manifest["updated_utc"] = command_record["timestamp_utc"]
        _atomic_json(MANIFEST_PATH, manifest)

        with COMMANDS_LOG_PATH.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp_utc": _utc_now(),
                        "run_id": RUN_ID,
                        "argv": sys.argv,
                        "working_directory": os.getcwd(),
                        "python_executable": sys.executable,
                        "resume_completed_requested": RESUME_COMPLETED,
                        "clean_git_required": REQUIRE_CLEAN_GIT,
                        "audited_evidence_mode": AUDITED_EVIDENCE_MODE,
                        "dirty_git_override_requested": ALLOW_DIRTY_GIT,
                        "dirty_git_override_reason": DIRTY_OVERRIDE_REASON or None,
                        "repository": repository,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _INITIALISED = True
    except Exception:
        _release_run_lock()
        raise


def lock_token_environment() -> dict[str, str]:
    """Return the authorised-child lock token for a subprocess spawned by this command."""

    if not _INITIALISED or _LOCK_TOKEN is None:
        raise RuntimeError("init_run() must be called before spawning an authorised child.")
    return {"ATAA_RUN_LOCK_TOKEN": _LOCK_TOKEN}


def as_strings() -> dict[str, str]:
    """Return run paths as strings for diagnostics and JSON serialisation."""

    return {
        "RUN_ID": RUN_ID,
        "RUN_ROOT": str(RUN_ROOT),
        "DATA_CACHE": str(DATA_CACHE),
        "SPLITS_DIR": str(SPLITS_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "OUTPUTS_DIR": str(OUTPUTS_DIR),
        "FIGURES_DIR": str(FIGURES_DIR),
        "LOGS_DIR": str(LOGS_DIR),
        "AUDIT_DIR": str(AUDIT_DIR),
        "RELEASE_STAGING_DIR": str(RELEASE_STAGING_DIR),
        "MANIFEST_PATH": str(MANIFEST_PATH),
        "COMMANDS_LOG_PATH": str(COMMANDS_LOG_PATH),
        "COMPLETE_MARKER": str(COMPLETE_MARKER),
        "RUN_LOCK_PATH": str(RUN_LOCK_PATH),
        "INITIALISED": str(_INITIALISED),
    }


def update_manifest(patch: dict[str, Any]) -> None:
    """Deep-merge a bounded update into the current run manifest atomically.

    The I05 per-run process lock must be in place before concurrent commands are
    permitted to call this function.
    """

    if not _INITIALISED:
        raise RuntimeError("init_run() must be called before updating the run manifest.")
    if COMPLETE_MARKER.exists():
        raise RuntimeError("Completed evidence is immutable; refusing to update its manifest.")
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("run_id") != RUN_ID:
        raise RuntimeError("Refusing to update a manifest belonging to another run.")
    _deep_merge(manifest, patch)
    manifest["updated_utc"] = _utc_now()
    _atomic_json(MANIFEST_PATH, manifest)
