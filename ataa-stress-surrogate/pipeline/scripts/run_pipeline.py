"""Run the complete ATAA evidence pipeline in one fail-fast, locked process graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ANALYSIS_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ANALYSIS_DIR / "scripts"
sys.path.insert(0, str(ANALYSIS_DIR))


def _parser() -> argparse.ArgumentParser:
    from pipeline_orchestration import PIPELINE_STEPS

    names = tuple(step.name for step in PIPELINE_STEPS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", choices=names)
    parser.add_argument("--through-step", choices=names)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-reason")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    return parser


def _load_loss_policy(path: Path) -> dict:
    from evidence_gates import load_loss_policy

    return load_loss_policy(path)


def _load_data_audit(path: Path) -> dict:
    from evidence_gates import load_data_audit

    return load_data_audit(path)


def _log_refusal(runs_root: Path, payload: dict) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with (runs_root / "refusals.log").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _step_record_path(audit_dir: Path, step_name: str, state: str) -> Path:
    return audit_dir / "orchestration_steps" / f"{step_name}.{state}.json"


def _write_new_step_record(path: Path, payload: dict) -> None:
    """Create an immutable step record; never replace evidence from an earlier attempt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {
                "schema_version": 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _build_evidence_environment(
    base: dict[str, str],
    *,
    torch_threads: int,
    torch_interop_threads: int,
    dirty_override_reason: str | None,
) -> dict[str, str]:
    """Return the exact environment propagated to every evidence subprocess."""

    environment = dict(base)
    environment["ATAA_REQUIRE_CLEAN_GIT"] = "1"
    environment["ATAA_AUDITED_EVIDENCE_MODE"] = "1"
    environment["ATAA_EVIDENCE_DEVICE"] = "cpu"
    environment["ATAA_TORCH_THREADS"] = str(torch_threads)
    environment["ATAA_TORCH_INTEROP_THREADS"] = str(torch_interop_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(torch_threads)
    if dirty_override_reason:
        environment["ATAA_ALLOW_DIRTY_GIT"] = "1"
        environment["ATAA_DIRTY_OVERRIDE_REASON"] = dirty_override_reason
    else:
        environment.pop("ATAA_ALLOW_DIRTY_GIT", None)
        environment.pop("ATAA_DIRTY_OVERRIDE_REASON", None)
    return environment


def main() -> None:
    args = _parser().parse_args()
    if args.torch_threads <= 0 or args.torch_interop_threads <= 0:
        raise SystemExit("thread counts must be positive")

    from pipeline_orchestration import output_collisions, selected_steps

    steps = selected_steps(args.from_step, args.through_step)
    plan = [
        {
            "name": step.name,
            "command": step.command(SCRIPTS_DIR),
            "requires_locked_loss": step.requires_locked_loss,
            "outputs": list(step.outputs),
        }
        for step in steps
    ]
    if args.dry_run:
        print(json.dumps({"schema_version": 1, "steps": plan}, indent=2))
        return

    evidence_environment = _build_evidence_environment(
        os.environ,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        dirty_override_reason=args.allow_dirty_reason,
    )
    for name, value in evidence_environment.items():
        os.environ[name] = value
    for name in ("ATAA_ALLOW_DIRTY_GIT", "ATAA_DIRTY_OVERRIDE_REASON"):
        if name not in evidence_environment:
            os.environ.pop(name, None)

    import config as C

    C.init_run()
    if args.allow_dirty_reason:
        C.update_manifest(
            {
                "artifact_policy": {
                    "dirty_git_override_reason": args.allow_dirty_reason,
                }
            }
        )

    if any(step.name not in {"convert", "diagnose", "splits"} for step in steps):
        data_audit_path = Path(C.AUDIT_DIR) / "data_audit.json"
        data_audit = _load_data_audit(data_audit_path)
        C.update_manifest({"data_audit_gate": data_audit})
    if any(step.requires_locked_loss for step in steps):
        policy_path = Path(C.AUDIT_DIR) / "loss_policy.json"
        policy = _load_loss_policy(policy_path)
        C.update_manifest({"loss_policy": policy})

    attempt_collisions = [
        str(_step_record_path(Path(C.AUDIT_DIR), step.name, "started"))
        for step in steps
        if _step_record_path(Path(C.AUDIT_DIR), step.name, "started").exists()
    ]
    collisions = [*output_collisions(C.RUN_ROOT, steps), *attempt_collisions]
    if collisions:
        _log_refusal(
            Path(C.RUNS_ROOT),
            {"run_id": C.RUN_ID, "reason": "output_collision", "paths": collisions},
        )
        raise FileExistsError(
            "Refusing to overwrite existing pipeline evidence; choose a fresh run or a later "
            f"--from-step after inspection: {collisions}"
        )

    started = time.perf_counter()
    completed_steps: list[dict] = []
    child_environment = evidence_environment.copy()
    child_environment.update(C.lock_token_environment())
    for step in steps:
        command = step.command(SCRIPTS_DIR)
        started_record = _step_record_path(Path(C.AUDIT_DIR), step.name, "started")
        _write_new_step_record(
            started_record,
            {
                "run_id": C.RUN_ID,
                "step": step.name,
                "state": "started",
                "command": command,
                "declared_outputs": list(step.outputs),
                "repository": C.current_git_snapshot(),
            },
        )
        print(f"\n=== {step.name} ===", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        step_started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=C.REPO_ROOT,
            env=child_environment,
            check=False,
        )
        elapsed = time.perf_counter() - step_started
        if result.returncode != 0:
            _log_refusal(
                Path(C.RUNS_ROOT),
                {
                    "run_id": C.RUN_ID,
                    "reason": "step_failed",
                    "step": step.name,
                    "returncode": result.returncode,
                },
            )
            raise RuntimeError(
                f"Pipeline stopped at {step.name!r} with exit code {result.returncode}. "
                "No later command was started."
            )
        missing = [
            relative for relative in step.outputs if not (Path(C.RUN_ROOT) / relative).exists()
        ]
        if missing:
            _log_refusal(
                Path(C.RUNS_ROOT),
                {"run_id": C.RUN_ID, "reason": "missing_outputs", "step": step.name, "paths": missing},
            )
            raise RuntimeError(f"Step {step.name!r} returned success but omitted outputs: {missing}")
        if step.name != "finalise":
            _write_new_step_record(
                _step_record_path(Path(C.AUDIT_DIR), step.name, "completed"),
                {
                    "run_id": C.RUN_ID,
                    "step": step.name,
                    "state": "completed",
                    "seconds": elapsed,
                    "declared_outputs": list(step.outputs),
                },
            )
        completed_steps.append({"name": step.name, "seconds": elapsed})
        if step.name != "finalise":
            C.update_manifest(
                {
                    "orchestration": {
                        "last_completed_step": step.name,
                        "completed_steps": completed_steps,
                        "torch_threads": args.torch_threads,
                        "torch_interop_threads": args.torch_interop_threads,
                        "evidence_device": "cpu",
                    }
                }
            )

    print(
        f"\nCompleted {len(steps)} pipeline steps in {time.perf_counter() - started:.1f} s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
