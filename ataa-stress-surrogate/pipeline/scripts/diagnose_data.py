"""Diagnose NaN/inf and scaling pathologies in the converted data."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from evidence_hashing import sha256_file
from result_provenance import artifact_record, utc_now, write_json

EXPECTED_SHAPE = (79626, 16248)
EXPECTED_BAD_ROWS = 1400


def _provenance_csv() -> Path | None:
    named = os.environ.get("ATAA_ROW_PROVENANCE", "").strip()
    candidates = [Path(C.AUDIT_DIR) / "row_provenance.csv"]
    if named:
        candidates.insert(0, Path(named))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _provenance_for_bad_rows(bad: list[int]) -> dict[str, object]:
    source = _provenance_csv()
    if source is None or not bad:
        return {"provenance_csv_present": source is not None, "n_bad_cases": None}
    frame = pd.read_csv(source)
    matched = frame[frame["source_row"].isin(bad)]
    cases = (
        matched.groupby(["campaign", "case_id"], dropna=False)
        .size()
        .reset_index(name="n_bad_rows")
    )
    return {
        "provenance_csv_present": True,
        "n_mapped_bad_rows": int(len(matched)),
        "n_bad_cases": int(len(cases)),
        "n_full_20_state_bad_cases": int((cases["n_bad_rows"] == 20).sum()),
        "partial_bad_cases": cases.loc[cases["n_bad_rows"] != 20].to_dict("records"),
    }


def main() -> None:
    C.init_run()
    inputs = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype("float64")
    print("INPUTS", inputs.shape)
    print("  NaN:", int(np.isnan(inputs).sum()), " inf:", int(np.isinf(inputs).sum()))
    print("  col min:", np.round(inputs.min(0), 3))
    print("  col max:", np.round(inputs.max(0), 3))

    output_path = Path(C.DATA_CACHE) / "sim_outputs_full.npy"
    outputs = np.load(output_path, mmap_mode="r")
    print("OUTPUTS", outputs.shape)
    nan_rows, inf_rows, all_nan_rows, n_nan, n_inf = set(), set(), set(), 0, 0
    global_minimum, global_maximum = np.inf, -np.inf
    batch_size = 2000
    for start in range(0, outputs.shape[0], batch_size):
        batch = np.asarray(outputs[start : start + batch_size], dtype="float64")
        mask_nan = np.isnan(batch)
        mask_inf = np.isinf(batch)
        if mask_nan.any():
            n_nan += int(mask_nan.sum())
            nan_rows.update((start + np.where(mask_nan.any(1))[0]).tolist())
            all_nan_rows.update((start + np.where(mask_nan.all(1))[0]).tolist())
        if mask_inf.any():
            n_inf += int(mask_inf.sum())
            inf_rows.update((start + np.where(mask_inf.any(1))[0]).tolist())
        finite = batch[np.isfinite(batch)]
        if finite.size:
            global_minimum = min(global_minimum, finite.min())
            global_maximum = max(global_maximum, finite.max())
    print(f"  NaN entries: {n_nan} in {len(nan_rows)} rows")
    print(f"  inf entries: {n_inf} in {len(inf_rows)} rows")
    print(f"  finite value range: [{global_minimum:.4g}, {global_maximum:.4g}]")

    stress = np.asarray(outputs[:200, : C.N_NODES * 6], dtype="float64")
    deformation = np.asarray(outputs[:200, C.N_NODES * 6 :], dtype="float64")
    print(
        "  S block |mean| ~",
        np.nanmean(np.abs(stress)),
        " C block |mean| ~",
        np.nanmean(np.abs(deformation)),
    )

    bad = sorted(nan_rows | inf_rows)
    shape_ok = tuple(outputs.shape) == EXPECTED_SHAPE
    dtype_ok = outputs.dtype == np.dtype("float32")
    expected_bad = len(bad) == EXPECTED_BAD_ROWS
    all_bad_fully_nan = set(bad) == all_nan_rows and not inf_rows
    if not shape_ok or not dtype_ok or not expected_bad or not all_bad_fully_nan:
        raise RuntimeError(
            "Data diagnosis failed: "
            f"shape_ok={shape_ok}, dtype_ok={dtype_ok}, "
            f"expected_bad_rows={expected_bad}, all_bad_fully_nan={all_bad_fully_nan}"
        )
    destination = Path(C.OUTPUTS_DIR) / "bad_rows.npy"
    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    bad_array = np.array(bad, dtype=np.int64)
    if destination.exists():
        existing_bad = np.load(destination, allow_pickle=False)
        if not np.array_equal(existing_bad, bad_array):
            raise RuntimeError("Recomputed bad-row IDs differ from the frozen exclusion file")
    else:
        np.save(destination, bad_array)
    print(f"  saved {len(bad)} bad row indices -> {destination}")

    provenance = _provenance_for_bad_rows(bad)
    source_output_sha256 = sha256_file(C.OUTPUTS_FULL)
    converted_sha256 = sha256_file(output_path)
    report = {
        "schema_version": 1,
        "analysis": "data_diagnosis",
        "run_id": C.RUN_ID,
        "generated_utc": utc_now(),
        "inputs": {
            "path": str(C.INPUTS_FULL),
            "shape": [int(inputs.shape[0]), int(inputs.shape[1])],
            "n_nan": int(np.isnan(inputs).sum()),
            "n_inf": int(np.isinf(inputs).sum()),
            "sha256": sha256_file(C.INPUTS_FULL),
        },
        "outputs": {
            "path": str(output_path),
            "shape": [int(outputs.shape[0]), int(outputs.shape[1])],
            "dtype": str(outputs.dtype),
            "shape_matches_expected": shape_ok,
            "dtype_is_float32": dtype_ok,
            "n_nan_entries": int(n_nan),
            "n_inf_entries": int(n_inf),
            "n_bad_rows": len(bad),
            "n_finite_rows": int(outputs.shape[0]) - len(bad),
            "bad_row_count_matches_stage0": len(bad) == EXPECTED_BAD_ROWS,
            "all_bad_rows_are_fully_nan": all_bad_fully_nan,
            "finite_min": None if not np.isfinite(global_minimum) else float(global_minimum),
            "finite_max": None if not np.isfinite(global_maximum) else float(global_maximum),
            "sha256": converted_sha256,
            "source_txt": str(C.OUTPUTS_FULL),
            "source_txt_sha256": source_output_sha256,
            "global_range_is_descriptive_only": True,
        },
        "bad_rows": artifact_record(destination, C.RUN_ROOT),
        "failed_simulation_join": provenance,
    }
    report_path = write_json(Path(C.OUTPUTS_DIR) / "data_diagnosis.json", report)

    imported_files = {}
    for name in (
        "data_audit.json",
        "source_hashes.sha256",
        "row_provenance.csv",
        "row_mapping_summary.json",
        "raw_fe_inventory.csv",
    ):
        path = Path(C.AUDIT_DIR) / name
        if path.is_file():
            imported_files[name] = artifact_record(path, C.RUN_ROOT)
    data_audit_path = Path(C.AUDIT_DIR) / "data_audit.json"
    source_audit_run_id = None
    if data_audit_path.is_file():
        import json

        with data_audit_path.open("r", encoding="utf-8") as stream:
            source_audit_run_id = json.load(stream).get("run_id")
    import_record = {
        "schema_version": 1,
        "destination_run_id": C.RUN_ID,
        "source_run_id": source_audit_run_id,
        "files": imported_files,
    }
    import_record_path = write_json(
        Path(C.AUDIT_DIR) / "imported_evidence.json", import_record
    )
    C.update_manifest(
        {
            "source_inputs": {
                "inputs_full": str(C.INPUTS_FULL),
                "outputs_full": str(C.OUTPUTS_FULL),
                "hashes": {
                    "inputs_full_sha256": report["inputs"]["sha256"],
                    "outputs_full_sha256": source_output_sha256,
                    "converted_outputs_sha256": converted_sha256,
                },
            },
            "scientific_layout": {
                "stress_source_unit": "dyne/cm^2 (audited I06)",
                "stress_report_unit": "kPa",
                "stress_to_kpa": 1e-4,
                "tensor_packing": {
                    "stress": ["11", "22", "33", "12", "23", "13"],
                    "deformation": ["11", "22", "33", "12", "21", "13"],
                    "limitation": "C23 is absent from the preserved prepared matrix",
                },
            },
            "imported_evidence": {
                "record": artifact_record(import_record_path, C.RUN_ROOT),
                "source_run_id": source_audit_run_id,
                "files": imported_files,
            },
            "i07_diagnosis": {
                "report": artifact_record(report_path, C.RUN_ROOT),
                "n_bad_rows": len(bad),
                "n_finite_rows": report["outputs"]["n_finite_rows"],
                "passed": True,
            },
        }
    )


if __name__ == "__main__":
    main()
