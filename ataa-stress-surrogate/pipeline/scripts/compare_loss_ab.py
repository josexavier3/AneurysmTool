"""Compare the two prospective I07A losses on grouped validation rows only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from evidence_gates import load_loss_ab_protocol
from evidence_hashing import sha256_file
from result_provenance import artifact_record, finite_summary, utc_now, write_csv, write_json

sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.metrics import (
    euclidean_peak_location_error,
    per_sample_field_metrics,
    summarise_metric_arrays,
)
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal

FIELD_WIDTH = C.N_NODES * C.N_COMP
LEGACY_ID = "legacy_standardised_sigma_v0"
GHAZI_ID = "ataa_group_balanced_ghazi_v1"
RUN_NAMES = {
    LEGACY_ID: f"cardiac_grouped_free_seed42_{LEGACY_ID}",
    GHAZI_ID: f"cardiac_grouped_free_seed42_{GHAZI_ID}",
}


def _select_loss(
    legacy_r2: float,
    legacy_rel_l2: float,
    ghazi_r2: float,
    ghazi_rel_l2: float,
    *,
    max_r2_decrease: float,
    max_rel_l2_ratio: float,
) -> dict[str, object]:
    r2_decrease = legacy_r2 - ghazi_r2
    rel_l2_ratio = ghazi_rel_l2 / legacy_rel_l2
    within_limits = bool(
        r2_decrease <= max_r2_decrease and rel_l2_ratio <= max_rel_l2_ratio
    )
    return {
        "selected_loss_id": GHAZI_ID if within_limits else LEGACY_ID,
        "ghazi_minus_legacy_r2": ghazi_r2 - legacy_r2,
        "ghazi_to_legacy_relative_l2_ratio": rel_l2_ratio,
        "within_both_noninferiority_limits": within_limits,
    }


def _load_log(run_name: str) -> tuple[Path, dict]:
    path = Path(C.MODELS_DIR) / f"{run_name}_train_log.json"
    with path.open("r", encoding="utf-8") as stream:
        return path, json.load(stream)


def _summary(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    return summarise_metric_arrays(metrics)


def _clustered_sensitivity(
    per_case: pd.DataFrame, geometry_ids: np.ndarray
) -> dict[str, object]:
    frame = per_case.copy()
    frame["geometry_id"] = geometry_ids
    geometry = frame.groupby("geometry_id", sort=True).median(numeric_only=True)
    legacy_r2 = geometry[f"{LEGACY_ID}_principal_R2"].to_numpy()
    legacy_rel = geometry[f"{LEGACY_ID}_principal_relL2"].to_numpy()
    ghazi_r2 = geometry[f"{GHAZI_ID}_principal_R2"].to_numpy()
    ghazi_rel = geometry[f"{GHAZI_ID}_principal_relL2"].to_numpy()

    rng = np.random.default_rng(4246)
    draws = 4000
    r2_decrease = np.empty(draws, dtype=np.float64)
    rel_ratio = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.integers(0, len(geometry), size=len(geometry))
        r2_decrease[index] = np.median(legacy_r2[sample]) - np.median(
            ghazi_r2[sample]
        )
        rel_ratio[index] = np.median(ghazi_rel[sample]) / np.median(
            legacy_rel[sample]
        )
    return {
        "unit": "geometry; median of per-case metrics within each geometry",
        "n_geometries": int(len(geometry)),
        "geometry_median_metrics": {
            LEGACY_ID: {
                "R2": finite_summary(legacy_r2),
                "relL2": finite_summary(legacy_rel),
            },
            GHAZI_ID: {
                "R2": finite_summary(ghazi_r2),
                "relL2": finite_summary(ghazi_rel),
            },
        },
        "cluster_bootstrap": {
            "seed": 4246,
            "draws": draws,
            "r2_decrease_legacy_minus_ghazi": finite_summary(r2_decrease),
            "relative_l2_ratio_ghazi_over_legacy": finite_summary(rel_ratio),
            "probability_r2_decrease_exceeds_0_02": float(
                np.mean(r2_decrease > 0.02)
            ),
            "probability_relative_l2_ratio_exceeds_1_10": float(
                np.mean(rel_ratio > 1.10)
            ),
        },
    }


def main() -> None:
    C.init_run()
    protocol_path = Path(C.AUDIT_DIR) / "loss_ab_protocol.json"
    protocol = load_loss_ab_protocol(protocol_path)

    split_path = Path(C.SPLITS_DIR) / "grouped_split.npz"
    with np.load(split_path) as split:
        train_indices = split["train"].astype(np.int64, copy=True)
        validation_indices = split["val"].astype(np.int64, copy=True)
        # Deliberately do not access split["test"]; it remains sealed for Gate L.

    targets = np.load(Path(C.DATA_CACHE) / "sim_outputs_full.npy", mmap_mode="r")
    validation_reference = np.asarray(targets[validation_indices], dtype=np.float32)
    validation_stress = validation_reference[:, :FIELD_WIDTH]
    reference_principal = first_principal(
        validation_stress, n_nodes=C.N_NODES, scale=DYNE_CM2_TO_KPA
    )

    template_path = C.DATA_ROOT / "1.Meshes" / "7.PCA1" / "3.Sampled" / "0.vtp"
    points = np.asarray(pv.read(template_path).points, dtype=np.float64)
    if points.shape != (C.N_NODES, 3):
        raise RuntimeError(f"Unexpected template point shape: {points.shape}")

    model_results: dict[str, dict] = {}
    provenance = pd.read_csv(Path(C.AUDIT_DIR) / "row_provenance.csv")
    if not np.array_equal(provenance["source_row"].to_numpy(), np.arange(len(targets))):
        raise RuntimeError("row provenance is not aligned with the prepared matrix")
    validation_geometry_ids = provenance.loc[
        validation_indices, "sample_id"
    ].to_numpy()
    per_case = pd.DataFrame(
        {
            "source_row": validation_indices,
            "geometry_id": validation_geometry_ids,
        }
    )
    shared_architecture = None
    shared_split_hash = None
    shared_arguments = None
    shared_determinism = None
    shared_scaler_hashes = None
    for loss_id, run_name in RUN_NAMES.items():
        log_path, log = _load_log(run_name)
        arguments = log["arguments"]
        if (
            arguments.get("evaluation_partition") != "validation"
            or arguments.get("grouped_test_set_sealed") is not True
            or arguments.get("split") != "grouped"
            or arguments.get("seed") != 42
            or arguments.get("loss_policy") != loss_id
        ):
            raise RuntimeError(f"Run {run_name} is not an authorised sealed I07A run")
        architecture = log["architecture"]
        split_hash = log["split_hashes"]["archive_sha256"]
        comparable_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in {"loss_policy", "tag"}
        }
        scaler_hashes = {
            key: log["hashes"][key]
            for key in ("input_scaler_sha256", "output_scaler_sha256")
        }
        if shared_architecture is None:
            shared_architecture = architecture
            shared_split_hash = split_hash
            shared_arguments = comparable_arguments
            shared_determinism = log["determinism"]
            shared_scaler_hashes = scaler_hashes
        elif (
            architecture != shared_architecture
            or split_hash != shared_split_hash
            or comparable_arguments != shared_arguments
            or log["determinism"] != shared_determinism
            or scaler_hashes != shared_scaler_hashes
        ):
            raise RuntimeError(
                "I07A runs differ in architecture, split, arguments, determinism or scalers"
            )

        prediction_path = Path(C.MODELS_DIR) / f"{run_name}_pred_validation.npy"
        prediction = np.load(prediction_path, mmap_mode="r")
        if prediction.shape != validation_reference.shape:
            raise RuntimeError(
                f"Prediction/reference shape mismatch for {run_name}: "
                f"{prediction.shape} != {validation_reference.shape}"
            )
        predicted_principal = first_principal(
            np.asarray(prediction[:, :FIELD_WIDTH]),
            n_nodes=C.N_NODES,
            scale=DYNE_CM2_TO_KPA,
        )
        principal_metrics = per_sample_field_metrics(
            predicted_principal, reference_principal
        )
        pws_predicted = predicted_principal.max(axis=1)
        pws_reference = reference_principal.max(axis=1)
        pws_magnitude_error = np.abs(pws_predicted - pws_reference)
        pws_location_error = euclidean_peak_location_error(
            predicted_principal, reference_principal, points
        )
        c_metrics = per_sample_field_metrics(
            np.asarray(prediction[:, FIELD_WIDTH:]),
            validation_reference[:, FIELD_WIDTH:],
        )
        for name, values in principal_metrics.items():
            per_case[f"{loss_id}_principal_{name}"] = values
        per_case[f"{loss_id}_pws_magnitude_error_kpa"] = pws_magnitude_error
        per_case[f"{loss_id}_pws_location_error_mesh_units"] = pws_location_error
        model_results[loss_id] = {
            "run_name": run_name,
            "training_log": artifact_record(log_path, C.RUN_ROOT),
            "predictions": artifact_record(prediction_path, C.RUN_ROOT),
            "best_epoch": log["best_epoch"],
            "stopping_epoch": log["stopping_epoch"],
            "wall_time_seconds": log["wall_time_seconds"],
            "first_principal_stress_kpa": _summary(principal_metrics),
            "peak_wall_stress_magnitude_error_kpa": finite_summary(
                pws_magnitude_error
            ),
            "peak_wall_stress_location_error_mesh_units": finite_summary(
                pws_location_error
            ),
            "C_component_diagnostic": {
                "limitation": "C23 is absent; component metrics are diagnostic only",
                "metrics": _summary(c_metrics),
            },
            "best_checkpoint_loss_contributions": log["validation_history"][
                log["best_epoch"]
            ],
            "final_epoch_loss_contributions": log["validation_history"][-1],
        }

    stress_sum = np.zeros(FIELD_WIDTH, dtype=np.float64)
    deformation_sum = np.zeros(FIELD_WIDTH, dtype=np.float64)
    for start in range(0, len(train_indices), 2000):
        rows = train_indices[start : start + 2000]
        stress_sum += np.asarray(
            targets[rows, :FIELD_WIDTH], dtype=np.float64
        ).sum(axis=0)
        deformation_sum += np.asarray(
            targets[rows, FIELD_WIDTH:], dtype=np.float64
        ).sum(axis=0)
    mean_stress = (stress_sum / len(train_indices)).astype(np.float32)[None, :]
    mean_principal = first_principal(
        mean_stress, n_nodes=C.N_NODES, scale=DYNE_CM2_TO_KPA
    )
    meanfield_metrics = per_sample_field_metrics(
        np.repeat(mean_principal, len(validation_indices), axis=0),
        reference_principal,
    )
    mean_deformation = (
        deformation_sum / len(train_indices)
    ).astype(np.float32)[None, :]
    meanfield_c_metrics = per_sample_field_metrics(
        np.repeat(mean_deformation, len(validation_indices), axis=0),
        validation_reference[:, FIELD_WIDTH:],
    )

    primary = protocol["decision_rule"]["primary_metrics"]
    legacy_summary = model_results[LEGACY_ID]["first_principal_stress_kpa"]
    ghazi_summary = model_results[GHAZI_ID]["first_principal_stress_kpa"]
    decision = _select_loss(
        float(legacy_summary["R2"]["median"]),
        float(legacy_summary["relL2"]["median"]),
        float(ghazi_summary["R2"]["median"]),
        float(ghazi_summary["relL2"]["median"]),
        max_r2_decrease=float(
            primary["first_principal_stress_r2_max_absolute_decrease"]
        ),
        max_rel_l2_ratio=float(
            primary["first_principal_stress_relative_l2_max_ratio"]
        ),
    )
    decision["summary_statistic"] = "median of equally weighted per-case metrics"
    decision["test_partition_accessed"] = False
    geometry_sensitivity = _clustered_sensitivity(
        per_case.drop(columns=["geometry_id"]), validation_geometry_ids
    )

    per_case_path = write_csv(
        Path(C.OUTPUTS_DIR) / "loss_ab_validation_per_case.csv", per_case
    )
    code_hashes = {
        "compare_loss_ab.py": sha256_file(Path(__file__).resolve()),
        "ataa_surrogate/metrics.py": sha256_file(
            C.PUBLIC_SRC_DIR / "ataa_surrogate" / "metrics.py"
        ),
        "ataa_surrogate/tensors.py": sha256_file(
            C.PUBLIC_SRC_DIR / "ataa_surrogate" / "tensors.py"
        ),
        "result_provenance.py": sha256_file(
            Path(__file__).resolve().parents[1] / "result_provenance.py"
        ),
    }
    disclosures = {
        "prior_test_exposure": (
            "The byte-identical grouped test partition was evaluated in provisional "
            "19-20 August analyses. It was not accessed by I07A, but I08 is not a first "
            "project-level look at those rows."
        ),
        "selection_set_role": (
            "The 6,280 validation rows from 40 geometries drive scheduling, early stopping "
            "and loss selection; their absolute metrics are selection diagnostics, not "
            "generalisation estimates."
        ),
        "protocol_estimator_clarification": (
            "The committed comparison implementation operationalised the approved endpoints "
            "as medians of equally weighted per-case metrics; no degeneracies occurred. "
            "Geometry-level and clustered-bootstrap sensitivity are reported separately."
        ),
        "manifest_ledger_gap": (
            "The initial dirty-worktree conversion and split commands pre-date accumulated "
            "command_provenance. Converted and source hashes authenticate the matrix, and the "
            "clean-tree split rerun verified both frozen archives byte-for-byte."
        ),
        "C_source_limitation": (
            "Stored C duplicates C12 as C21 and omits C23; both objectives therefore weight "
            "the duplicated C12 entry, and all C results are diagnostic."
        ),
        "gradient_norm_history": "not recorded; per-field loss contributions were recorded",
    }
    clarification_path = write_json(
        Path(C.AUDIT_DIR) / "loss_ab_protocol_clarification.json",
        {
            "schema_version": 1,
            "status": "post-execution clarification; original approved protocol is immutable",
            "original_protocol": artifact_record(protocol_path, C.RUN_ROOT),
            "operational_estimator": "median of equally weighted per-case metrics",
            "selection_set": {
                "rows": int(len(validation_indices)),
                "geometries": int(np.unique(validation_geometry_ids).size),
            },
            "degenerate_case_policy": "canonical metric policy; no degeneracies occurred",
            "geometry_clustered_sensitivity_reported": True,
        },
    )
    report = {
        "schema_version": 1,
        "analysis": "prospective_loss_ab_grouped_validation",
        "run_id": C.RUN_ID,
        "generated_utc": utc_now(),
        "selection_partition": "grouped validation",
        "test_partition_accessed": False,
        "n_validation_cases": int(len(validation_indices)),
        "protocol": artifact_record(protocol_path, C.RUN_ROOT),
        "protocol_clarification": artifact_record(
            clarification_path, C.RUN_ROOT
        ),
        "split_archive": artifact_record(split_path, C.RUN_ROOT),
        "models": model_results,
        "meanfield_validation_diagnostic": _summary(meanfield_metrics),
        "C_meanfield_validation_diagnostic": {
            "limitation": "C23 is absent; component metrics are diagnostic only",
            "metrics": _summary(meanfield_c_metrics),
        },
        "geometry_clustered_sensitivity": geometry_sensitivity,
        "decision": decision,
        "disclosures": disclosures,
        "code_hashes": code_hashes,
        "artefacts": {
            "per_case_metrics": artifact_record(per_case_path, C.RUN_ROOT)
        },
    }
    report_path = write_json(
        Path(C.OUTPUTS_DIR) / "loss_ab_validation.json", report
    )
    C.update_manifest(
        {
            "loss_ab_validation": {
                "report": artifact_record(report_path, C.RUN_ROOT),
                "selected_loss_id": decision["selected_loss_id"],
                "test_partition_accessed": False,
            }
        }
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
