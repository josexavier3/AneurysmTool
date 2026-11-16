"""Shared validation for data, loss and training-evidence gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DATA_AUDIT_REQUIRED_TRUE = (
    "raw_to_matrix_mapping_verified",
    "tensor_layout_verified",
    "node_order_verified",
    "stress_source_unit_verified",
    "pressure_input_units_verified",
)

LOSS_POLICY_REQUIRED = {
    "author_approved": True,
    "grouped_test_set_used_for_selection": False,
    "validation_only_selection": True,
}
LOSS_AB_PROTOCOL_REQUIRED = {
    "author_approved": True,
    "grouped_test_set_sealed": True,
    "selection_partition": "validation",
}


def validate_data_audit(audit: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in DATA_AUDIT_REQUIRED_TRUE if audit.get(key) is not True]
    if missing:
        raise RuntimeError(f"Gate-D data audit is incomplete; required true fields: {missing}")
    return audit


def load_data_audit(
    path: str | Path,
    *,
    expected_run_id: str | None = None,
    import_record_path: str | Path | None = None,
) -> dict[str, Any]:
    audit_path = Path(path)
    if not audit_path.is_file():
        raise RuntimeError(f"Post-split evidence work is blocked until Gate D writes {audit_path}")
    with audit_path.open("r", encoding="utf-8") as stream:
        audit = validate_data_audit(json.load(stream))
    if expected_run_id is None or audit.get("run_id") == expected_run_id:
        return audit
    if import_record_path is None:
        raise RuntimeError(
            f"Data audit run_id {audit.get('run_id')!r} does not match {expected_run_id!r} "
            "and no import record was supplied"
        )
    record_path = Path(import_record_path)
    with record_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    imported = record.get("files", {}).get(audit_path.name, {})
    from evidence_hashing import sha256_file

    if (
        record.get("destination_run_id") != expected_run_id
        or record.get("source_run_id") != audit.get("run_id")
        or imported.get("sha256") != sha256_file(audit_path)
    ):
        raise RuntimeError("Data audit import record does not authenticate the imported audit")
    return audit


def validate_loss_policy(
    policy: dict[str, Any], *, expected_loss_id: str | None = None
) -> dict[str, Any]:
    for key, expected in LOSS_POLICY_REQUIRED.items():
        if policy.get(key) != expected:
            raise RuntimeError(
                f"Loss policy {key!r} must be {expected!r}; got {policy.get(key)!r}"
            )
    selected = policy.get("selected_loss_id")
    if not selected:
        raise RuntimeError("Loss policy does not name selected_loss_id")
    if expected_loss_id is not None and selected != expected_loss_id:
        raise RuntimeError(
            "The training implementation does not match the author-approved Gate-L loss "
            f"policy: selected={selected!r}, implemented={expected_loss_id!r}."
        )
    return policy


def load_loss_policy(
    path: str | Path, *, expected_loss_id: str | None = None
) -> dict[str, Any]:
    policy_path = Path(path)
    if not policy_path.is_file():
        raise RuntimeError(
            f"Training is blocked until Gate L writes an author-approved loss policy: {policy_path}"
        )
    with policy_path.open("r", encoding="utf-8") as stream:
        return validate_loss_policy(json.load(stream), expected_loss_id=expected_loss_id)


def validate_loss_ab_protocol(
    protocol: dict[str, Any], *, expected_loss_id: str | None = None
) -> dict[str, Any]:
    for key, expected in LOSS_AB_PROTOCOL_REQUIRED.items():
        if protocol.get(key) != expected:
            raise RuntimeError(
                f"Loss A/B protocol {key!r} must be {expected!r}; got {protocol.get(key)!r}"
            )
    candidates = protocol.get("candidate_loss_ids")
    if not isinstance(candidates, list) or len(candidates) != 2 or len(set(candidates)) != 2:
        raise RuntimeError("Loss A/B protocol must name exactly two distinct candidate_loss_ids")
    if expected_loss_id is not None and expected_loss_id not in candidates:
        raise RuntimeError(f"Loss A/B protocol does not authorise {expected_loss_id!r}")
    rule = protocol.get("decision_rule")
    if not isinstance(rule, dict) or not rule:
        raise RuntimeError("Loss A/B protocol must contain a prospective decision_rule")
    return protocol


def load_loss_ab_protocol(
    path: str | Path, *, expected_loss_id: str | None = None
) -> dict[str, Any]:
    protocol_path = Path(path)
    if not protocol_path.is_file():
        raise RuntimeError(
            "Loss A/B training is blocked until the author-approved prospective protocol exists: "
            f"{protocol_path}"
        )
    with protocol_path.open("r", encoding="utf-8") as stream:
        return validate_loss_ab_protocol(json.load(stream), expected_loss_id=expected_loss_id)


def validate_expected_training_runs(
    training_runs: dict[str, Any],
    expected: Iterable[str],
    *,
    data_audit_sha256: str | None = None,
    loss_policy_sha256: str | None = None,
) -> None:
    required = set(expected)
    available = set(training_runs)
    missing = sorted(required - available)
    failed_reload = sorted(
        name
        for name in required & available
        if training_runs[name].get("reload_verification_passed") is not True
    )
    non_evidence = sorted(
        name
        for name in required & available
        if training_runs[name].get("audited_evidence_mode") is not True
    )
    gate_mismatch = sorted(
        name
        for name in required & available
        if (
            data_audit_sha256 is not None
            and training_runs[name].get("evidence_gates", {}).get("data_audit_sha256")
            != data_audit_sha256
        )
        or (
            loss_policy_sha256 is not None
            and training_runs[name].get("evidence_gates", {}).get("loss_policy_sha256")
            != loss_policy_sha256
        )
    )
    if missing or failed_reload or non_evidence or gate_mismatch:
        raise RuntimeError(
            "Finalisation requires all audited model runs and reload checks: "
            f"missing={missing}, failed_reload={failed_reload}, non_evidence={non_evidence}, "
            f"gate_mismatch={gate_mismatch}"
        )
