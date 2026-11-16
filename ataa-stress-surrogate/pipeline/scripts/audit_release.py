"""Fail the evidence pipeline if public release staging contains restricted material."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from release_safety import scan_provenance_text, scan_public_tree
from result_provenance import artifact_record, utc_now, write_json


def main() -> None:
    C.init_run()
    public_findings = scan_public_tree(C.RELEASE_STAGING_DIR)
    provenance_findings = scan_provenance_text((C.COMMANDS_LOG_PATH, C.MANIFEST_PATH))
    destination = Path(C.AUDIT_DIR) / "privacy_audit.json"
    write_json(
        destination,
        {
            "schema_version": 1,
            "run_id": C.RUN_ID,
            "generated_utc": utc_now(),
            "public_staging_passed": not public_findings,
            "public_findings": public_findings,
            "private_provenance_publishable_without_redaction": not provenance_findings,
            "private_provenance_findings": provenance_findings,
            "policy": {
                "public_findings_are_blocking": True,
                "private_provenance_findings_require_redaction_before_publication": True,
                "private_evidence_may_retain_audited_machine_paths": True,
            },
        },
    )
    record = artifact_record(destination, C.RUN_ROOT)
    C.update_manifest(
        {
            "privacy_audit": {
                "passed": not public_findings,
                "artefact": record,
                "private_provenance_requires_redaction": bool(provenance_findings),
            }
        }
    )
    if public_findings:
        raise RuntimeError(f"Public release privacy audit failed: {public_findings}")
    print(f"public release privacy audit passed; report: {destination}")


if __name__ == "__main__":
    main()
