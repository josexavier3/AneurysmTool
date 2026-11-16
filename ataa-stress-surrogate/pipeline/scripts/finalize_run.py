"""Validate final evidence gates and immutably mark one run complete."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from evidence_gates import (
    load_data_audit,
    load_loss_policy,
    validate_data_audit,
    validate_expected_training_runs,
    validate_loss_policy,
)
from evidence_hashing import sha256_file
from training_provenance import ENSEMBLE_SEEDS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    C.init_run()
    with Path(C.MANIFEST_PATH).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("artifact_policy", {}).get("clean_git_required"):
        raise RuntimeError("Finalisation is allowed only through clean-Git evidence orchestration.")
    privacy = manifest.get("privacy_audit", {})
    if privacy.get("passed") is not True:
        raise RuntimeError("Finalisation requires a passing public-release privacy audit.")
    if not manifest.get("result_bundle"):
        raise RuntimeError("Finalisation requires the machine-readable result bundle.")
    data_audit_path = Path(C.AUDIT_DIR) / "data_audit.json"
    loss_policy_path = Path(C.AUDIT_DIR) / "loss_policy.json"
    data_audit = load_data_audit(data_audit_path)
    loss_policy = load_loss_policy(loss_policy_path)
    if data_audit != validate_data_audit(manifest.get("data_audit_gate", {})):
        raise RuntimeError("Final Gate-D file differs from the data audit recorded in the manifest.")
    if loss_policy != validate_loss_policy(manifest.get("loss_policy", {})):
        raise RuntimeError("Final Gate-L file differs from the loss policy recorded in the manifest.")

    training_runs = manifest.get("training_runs", {})
    expected = {
        *(f"cardiac_grouped_free_seed{seed}" for seed in ENSEMBLE_SEEDS),
        "cardiac_random_free_seed42",
    }
    validate_expected_training_runs(
        training_runs,
        expected,
        data_audit_sha256=sha256_file(data_audit_path),
        loss_policy_sha256=sha256_file(loss_policy_path),
    )

    completed_utc = _utc_now()
    C.update_manifest(
        {
            "status": "complete",
            "completed_utc": completed_utc,
            "orchestration": {"last_completed_step": "finalise"},
            "completion_marker": {
                "path": Path(C.COMPLETE_MARKER).relative_to(C.RUN_ROOT).as_posix(),
                "contract": "marker hashes the final immutable run_manifest.json",
            },
        }
    )
    marker = {
        "schema_version": 1,
        "status": "complete",
        "run_id": C.RUN_ID,
        "completed_utc": completed_utc,
        "manifest_sha256": sha256_file(C.MANIFEST_PATH),
    }
    with Path(C.COMPLETE_MARKER).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"run {C.RUN_ID} marked complete at {C.COMPLETE_MARKER}")


if __name__ == "__main__":
    main()
