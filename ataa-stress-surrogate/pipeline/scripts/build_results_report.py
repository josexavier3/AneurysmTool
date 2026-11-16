"""Build the manuscript-value bundle, claim register and human-readable result report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from result_provenance import (
    ARRAY_HASH_ALGORITHM,
    FILE_HASH_ALGORITHM,
    artifact_record,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


REQUIRED_SUMMARIES = (
    "metrics_grouped_summary.json",
    "metrics_random_summary.json",
    "metrics_patient_summary.json",
    "pws_spd_summary.json",
    "boundary_pws_summary.json",
    "pressure_dependence_summary.json",
    "percentiles_summary.json",
    "mc_dropout_summary.json",
    "ensemble_summary.json",
    "baselines_summary.json",
    "meanfield_summary.json",
    "meanfield_componentwise_summary.json",
)


CLAIM_SPECS = (
    ("grouped_n", "pws_spd_summary.json", ("results", "grouped", "n_cases"), "cases"),
    ("patient_n", "pws_spd_summary.json", ("results", "patient", "n_cases"), "cases"),
    ("grouped_fp_R2_median", "pws_spd_summary.json", ("results", "grouped", "first_principal_stress", "per_case_metrics", "R2", "median"), "dimensionless"),
    ("grouped_fp_relL2_median", "pws_spd_summary.json", ("results", "grouped", "first_principal_stress", "per_case_metrics", "relL2", "median"), "dimensionless"),
    ("grouped_fp_ccc_median", "pws_spd_summary.json", ("results", "grouped", "first_principal_stress", "per_case_metrics", "ccc", "median"), "dimensionless"),
    ("grouped_fp_mae_median", "pws_spd_summary.json", ("results", "grouped", "first_principal_stress", "per_case_metrics", "mae", "median"), "kPa"),
    ("grouped_fp_rmse_median", "pws_spd_summary.json", ("results", "grouped", "first_principal_stress", "per_case_metrics", "rmse", "median"), "kPa"),
    ("patient_fp_R2_median", "pws_spd_summary.json", ("results", "patient", "first_principal_stress", "per_case_metrics", "R2", "median"), "dimensionless"),
    ("patient_fp_relL2_median", "pws_spd_summary.json", ("results", "patient", "first_principal_stress", "per_case_metrics", "relL2", "median"), "dimensionless"),
    ("patient_fp_ccc_median", "pws_spd_summary.json", ("results", "patient", "first_principal_stress", "per_case_metrics", "ccc", "median"), "dimensionless"),
    ("patient_fp_mae_median", "pws_spd_summary.json", ("results", "patient", "first_principal_stress", "per_case_metrics", "mae", "median"), "kPa"),
    ("patient_fp_rmse_median", "pws_spd_summary.json", ("results", "patient", "first_principal_stress", "per_case_metrics", "rmse", "median"), "kPa"),
    ("grouped_pws_bias", "pws_spd_summary.json", ("results", "grouped", "peak_wall_stress", "bland_altman", "bias"), "kPa"),
    ("grouped_pws_lower_loa", "pws_spd_summary.json", ("results", "grouped", "peak_wall_stress", "bland_altman", "lower_loa"), "kPa"),
    ("grouped_pws_upper_loa", "pws_spd_summary.json", ("results", "grouped", "peak_wall_stress", "bland_altman", "upper_loa"), "kPa"),
    ("patient_pws_bias", "pws_spd_summary.json", ("results", "patient", "peak_wall_stress", "bland_altman", "bias"), "kPa"),
    ("patient_pws_lower_loa", "pws_spd_summary.json", ("results", "patient", "peak_wall_stress", "bland_altman", "lower_loa"), "kPa"),
    ("patient_pws_upper_loa", "pws_spd_summary.json", ("results", "patient", "peak_wall_stress", "bland_altman", "upper_loa"), "kPa"),
    ("grouped_pws_location_unmasked_median", "boundary_pws_summary.json", ("results", "grouped", "pws_location_unmasked_mm", "median"), "mm"),
    ("patient_pws_location_unmasked_median", "boundary_pws_summary.json", ("results", "patient", "pws_location_unmasked_mm", "median"), "mm"),
    ("grouped_pws_location_masked_median", "boundary_pws_summary.json", ("results", "grouped", "pws_location_boundary_masked_mm", "median"), "mm"),
    ("patient_pws_location_masked_median", "boundary_pws_summary.json", ("results", "patient", "pws_location_boundary_masked_mm", "median"), "mm"),
    ("grouped_pressure_spearman", "pressure_dependence_summary.json", ("results", "grouped", "spearman_rho"), "dimensionless"),
    ("patient_pressure_spearman", "pressure_dependence_summary.json", ("results", "patient", "spearman_rho"), "dimensionless"),
    ("grouped_pressure_sextile_1_mae", "pressure_dependence_summary.json", ("results", "grouped", "sextiles", 0, "mae_median_kpa"), "kPa"),
    ("grouped_pressure_sextile_6_mae", "pressure_dependence_summary.json", ("results", "grouped", "sextiles", 5, "mae_median_kpa"), "kPa"),
    ("patient_pressure_sextile_1_mae", "pressure_dependence_summary.json", ("results", "patient", "sextiles", 0, "mae_median_kpa"), "kPa"),
    ("patient_pressure_sextile_6_mae", "pressure_dependence_summary.json", ("results", "patient", "sextiles", 5, "mae_median_kpa"), "kPa"),
    ("grouped_pressure_sextile_2_mae", "pressure_dependence_summary.json", ("results", "grouped", "sextiles", 1, "mae_median_kpa"), "kPa"),
    ("grouped_pressure_sextile_3_mae", "pressure_dependence_summary.json", ("results", "grouped", "sextiles", 2, "mae_median_kpa"), "kPa"),
    ("grouped_pressure_sextile_4_mae", "pressure_dependence_summary.json", ("results", "grouped", "sextiles", 3, "mae_median_kpa"), "kPa"),
    ("grouped_pressure_sextile_5_mae", "pressure_dependence_summary.json", ("results", "grouped", "sextiles", 4, "mae_median_kpa"), "kPa"),
    ("patient_pressure_sextile_2_mae", "pressure_dependence_summary.json", ("results", "patient", "sextiles", 1, "mae_median_kpa"), "kPa"),
    ("patient_pressure_sextile_3_mae", "pressure_dependence_summary.json", ("results", "patient", "sextiles", 2, "mae_median_kpa"), "kPa"),
    ("patient_pressure_sextile_4_mae", "pressure_dependence_summary.json", ("results", "patient", "sextiles", 3, "mae_median_kpa"), "kPa"),
    ("patient_pressure_sextile_5_mae", "pressure_dependence_summary.json", ("results", "patient", "sextiles", 4, "mae_median_kpa"), "kPa"),
    ("grouped_mc_dropout_coverage", "mc_dropout_summary.json", ("results", "grouped", "pointwise_95_coverage_fraction"), "fraction"),
    ("patient_mc_dropout_coverage", "mc_dropout_summary.json", ("results", "patient", "pointwise_95_coverage_fraction"), "fraction"),
    ("grouped_ensemble_coverage", "ensemble_summary.json", ("results", "grouped", "uncertainty", "pointwise_95_coverage_fraction"), "fraction"),
    ("patient_ensemble_coverage", "ensemble_summary.json", ("results", "patient", "uncertainty", "pointwise_95_coverage_fraction"), "fraction"),
    ("grouped_predicted_C_spd_fraction", "pws_spd_summary.json", ("results", "grouped", "right_cauchy_green_admissibility", "predicted_spd_fraction"), "fraction"),
    ("patient_predicted_C_spd_fraction", "pws_spd_summary.json", ("results", "patient", "right_cauchy_green_admissibility", "predicted_spd_fraction"), "fraction"),
    ("random_n", "pws_spd_summary.json", ("results", "random", "n_cases"), "cases"),
    ("random_fp_R2_median", "pws_spd_summary.json", ("results", "random", "first_principal_stress", "per_case_metrics", "R2", "median"), "dimensionless"),
    ("random_fp_relL2_median", "pws_spd_summary.json", ("results", "random", "first_principal_stress", "per_case_metrics", "relL2", "median"), "dimensionless"),
    ("random_fp_mae_median", "pws_spd_summary.json", ("results", "random", "first_principal_stress", "per_case_metrics", "mae", "median"), "kPa"),
    ("grouped_fp_pearson_median", "pws_spd_summary.json", ("results", "grouped", "first_principal_stress", "per_case_metrics", "pearson", "median"), "dimensionless"),
    ("patient_fp_pearson_median", "pws_spd_summary.json", ("results", "patient", "first_principal_stress", "per_case_metrics", "pearson", "median"), "dimensionless"),
    ("grouped_relL2_p90", "percentiles_summary.json", ("results", "grouped", "metrics", "relL2", "p90"), "dimensionless"),
    ("grouped_relL2_p95", "percentiles_summary.json", ("results", "grouped", "metrics", "relL2", "p95"), "dimensionless"),
    ("grouped_relL2_p99", "percentiles_summary.json", ("results", "grouped", "metrics", "relL2", "p99"), "dimensionless"),
    ("patient_relL2_p90", "percentiles_summary.json", ("results", "patient", "metrics", "relL2", "p90"), "dimensionless"),
    ("patient_relL2_p95", "percentiles_summary.json", ("results", "patient", "metrics", "relL2", "p95"), "dimensionless"),
    ("patient_relL2_p99", "percentiles_summary.json", ("results", "patient", "metrics", "relL2", "p99"), "dimensionless"),
    ("grouped_mae_p90", "percentiles_summary.json", ("results", "grouped", "metrics", "mae", "p90"), "kPa"),
    ("grouped_mae_p95", "percentiles_summary.json", ("results", "grouped", "metrics", "mae", "p95"), "kPa"),
    ("grouped_mae_p99", "percentiles_summary.json", ("results", "grouped", "metrics", "mae", "p99"), "kPa"),
    ("patient_mae_p90", "percentiles_summary.json", ("results", "patient", "metrics", "mae", "p90"), "kPa"),
    ("patient_mae_p95", "percentiles_summary.json", ("results", "patient", "metrics", "mae", "p95"), "kPa"),
    ("patient_mae_p99", "percentiles_summary.json", ("results", "patient", "metrics", "mae", "p99"), "kPa"),
    ("grouped_reference_pws_median", "pws_spd_summary.json", ("results", "grouped", "peak_wall_stress", "reference", "median"), "kPa"),
    ("grouped_predicted_pws_median", "pws_spd_summary.json", ("results", "grouped", "peak_wall_stress", "predicted", "median"), "kPa"),
    ("patient_reference_pws_median", "pws_spd_summary.json", ("results", "patient", "peak_wall_stress", "reference", "median"), "kPa"),
    ("patient_predicted_pws_median", "pws_spd_summary.json", ("results", "patient", "peak_wall_stress", "predicted", "median"), "kPa"),
    ("grouped_pws_location_unmasked_within_5mm_fraction", "boundary_pws_summary.json", ("results", "grouped", "pws_location_unmasked_mm", "fraction_within_5_mm"), "fraction"),
    ("patient_pws_location_unmasked_within_5mm_fraction", "boundary_pws_summary.json", ("results", "patient", "pws_location_unmasked_mm", "fraction_within_5_mm"), "fraction"),
    ("grouped_boundary_masked_R2_median", "boundary_pws_summary.json", ("results", "grouped", "boundary_masked_first_principal_stress", "R2", "median"), "dimensionless"),
    ("patient_boundary_masked_R2_median", "boundary_pws_summary.json", ("results", "patient", "boundary_masked_first_principal_stress", "R2", "median"), "dimensionless"),
    ("grouped_boundary_masked_mae_median", "boundary_pws_summary.json", ("results", "grouped", "boundary_masked_first_principal_stress", "mae", "median"), "kPa"),
    ("patient_boundary_masked_mae_median", "boundary_pws_summary.json", ("results", "patient", "boundary_masked_first_principal_stress", "mae", "median"), "kPa"),
    ("grouped_mc_dropout_spearman", "mc_dropout_summary.json", ("results", "grouped", "spearman_sigma_vs_mae"), "dimensionless"),
    ("patient_mc_dropout_spearman", "mc_dropout_summary.json", ("results", "patient", "spearman_sigma_vs_mae"), "dimensionless"),
    ("grouped_mc_dropout_mean_sigma", "mc_dropout_summary.json", ("results", "grouped", "mean_component_sigma_kpa", "mean"), "kPa"),
    ("patient_mc_dropout_mean_sigma", "mc_dropout_summary.json", ("results", "patient", "mean_component_sigma_kpa", "mean"), "kPa"),
    ("grouped_ensemble_spearman", "ensemble_summary.json", ("results", "grouped", "uncertainty", "spearman_sigma_vs_mae"), "dimensionless"),
    ("patient_ensemble_spearman", "ensemble_summary.json", ("results", "patient", "uncertainty", "spearman_sigma_vs_mae"), "dimensionless"),
    ("grouped_ensemble_mean_sigma", "ensemble_summary.json", ("results", "grouped", "uncertainty", "mean_component_sigma_kpa", "mean"), "kPa"),
    ("patient_ensemble_mean_sigma", "ensemble_summary.json", ("results", "patient", "uncertainty", "mean_component_sigma_kpa", "mean"), "kPa"),
    ("ensemble_member_R2_mean", "ensemble_summary.json", ("results", "member_variation", "median_R2_across_members_mean"), "dimensionless"),
    ("ensemble_member_R2_sd", "ensemble_summary.json", ("results", "member_variation", "median_R2_across_members_sd_ddof1"), "dimensionless"),
    ("ensemble_member_mae_mean", "ensemble_summary.json", ("results", "member_variation", "median_mae_kpa_across_members_mean"), "kPa"),
    ("ensemble_member_mae_sd", "ensemble_summary.json", ("results", "member_variation", "median_mae_kpa_across_members_sd_ddof1"), "kPa"),
    ("pod_explained_variance_fraction", "baselines_summary.json", ("results", "pod", "explained_variance_fraction"), "fraction"),
    ("meanfield_C_moderate_fraction", "meanfield_componentwise_summary.json", ("results", "deformation_meanfield", "moderate_fraction"), "fraction"),
    ("meanfield_C_high_fraction", "meanfield_componentwise_summary.json", ("results", "deformation_meanfield", "high_fraction"), "fraction"),
    ("meanfield_C_pearson_median", "meanfield_componentwise_summary.json", ("results", "deformation_meanfield", "pearson", "median"), "dimensionless"),
    ("meanfield_C_nmae_median", "meanfield_componentwise_summary.json", ("results", "deformation_meanfield", "nmae_range", "median"), "dimensionless"),
    ("meanfield_S_moderate_fraction", "meanfield_componentwise_summary.json", ("results", "stress_meanfield", "moderate_fraction"), "fraction"),
    ("meanfield_S_high_fraction", "meanfield_componentwise_summary.json", ("results", "stress_meanfield", "high_fraction"), "fraction"),
    ("identity_C_nmae_median", "meanfield_componentwise_summary.json", ("results", "deformation_identity", "nmae_range", "median"), "dimensionless"),
    ("network_S_legacy_moderate_fraction", "metrics_grouped_summary.json", ("results", "legacy_componentwise_agreement", "S", "moderate_fraction"), "fraction"),
    ("network_S_legacy_high_fraction", "metrics_grouped_summary.json", ("results", "legacy_componentwise_agreement", "S", "high_fraction"), "fraction"),
    ("network_C_legacy_moderate_fraction", "metrics_grouped_summary.json", ("results", "legacy_componentwise_agreement", "C", "moderate_fraction"), "fraction"),
    ("network_C_legacy_high_fraction", "metrics_grouped_summary.json", ("results", "legacy_componentwise_agreement", "C", "high_fraction"), "fraction"),
)


def _load_summaries(outputs_dir: Path) -> dict[str, dict]:
    summaries = {}
    for filename in REQUIRED_SUMMARIES:
        path = outputs_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required result summary is missing: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema_version") != 1:
            raise RuntimeError(f"{filename}: unsupported schema_version")
        if payload.get("run_id") != C.RUN_ID:
            raise RuntimeError(
                f"{filename}: run_id {payload.get('run_id')!r} does not match {C.RUN_ID!r}"
            )
        summaries[filename] = payload
    return summaries


def _get(payload, path):
    value = payload
    for element in path:
        value = value[element]
    if value is None or isinstance(value, (dict, list)):
        raise RuntimeError(f"required scalar at {path!r} is missing or non-scalar")
    return value


def _json_pointer(path) -> str:
    return "/" + "/".join(str(element).replace("~", "~0").replace("/", "~1") for element in path)


def _baseline_claims(summaries):
    payload = summaries["baselines_summary.json"]
    rows = []
    for index, result in enumerate(payload["results"]["comparison_rows"]):
        safe_name = re.sub(r"[^a-z0-9]+", "_", result["model"].lower()).strip("_")
        for source_key, suffix, unit in (
            ("g_R2", "grouped_R2_median", "dimensionless"),
            ("g_relL2", "grouped_relL2_median", "dimensionless"),
            ("g_MAE", "grouped_mae_median", "kPa"),
            ("p_R2", "patient_R2_median", "dimensionless"),
            ("p_relL2", "patient_relL2_median", "dimensionless"),
            ("p_MAE", "patient_mae_median", "kPa"),
        ):
            rows.append(
                (
                    f"baseline_{safe_name}_{suffix}",
                    "baselines_summary.json",
                    ("results", "comparison_rows", index, source_key),
                    unit,
                )
            )
    return rows


def _claim_records(summaries, source_records):
    claims = {}
    for key, filename, path, unit in (*CLAIM_SPECS, *_baseline_claims(summaries)):
        value = _get(summaries[filename], path)
        claims[key] = {
            "value": value,
            "unit": unit,
            "source_file": filename,
            "json_path": _json_pointer(path),
            "source_sha256": source_records[filename]["sha256"],
            "manuscript_status": "pending I12 claim audit",
            "manuscript_location": None,
        }
    return claims


def _format(value, digits=4):
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}g}"


def _render_report(claims, summaries, source_records):
    value = lambda key: claims[key]["value"]
    lines = [
        "# ATAA clean evidence results report",
        "",
        f"- Run ID: `{C.RUN_ID}`",
        f"- Generated UTC: `{utc_now()}`",
        "- Status: machine-generated evidence; manuscript propagation remains pending Gate R/I12.",
        f"- File hash contract: `{FILE_HASH_ALGORITHM}`",
        f"- Array hash contract: `{ARRAY_HASH_ALGORITHM}`",
        "",
        "## First-principal-stress performance",
        "",
        "| Dataset | n | Median R2 | Median relative L2 | Median CCC | Median MAE (kPa) | Median RMSE (kPa) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("grouped", "patient"):
        lines.append(
            f"| {dataset} | {_format(value(dataset + '_n'))} | "
            f"{_format(value(dataset + '_fp_R2_median'))} | "
            f"{_format(value(dataset + '_fp_relL2_median'))} | "
            f"{_format(value(dataset + '_fp_ccc_median'))} | "
            f"{_format(value(dataset + '_fp_mae_median'))} | "
            f"{_format(value(dataset + '_fp_rmse_median'))} |"
        )
    lines.extend(
        [
            "",
            "## Grouped-versus-random split sensitivity",
            "",
            "| Split | n | Median R2 | Median relative L2 | Median MAE (kPa) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("grouped", "random"):
        lines.append(
            f"| {dataset} | {_format(value(dataset + '_n'))} | "
            f"{_format(value(dataset + '_fp_R2_median'))} | "
            f"{_format(value(dataset + '_fp_relL2_median'))} | "
            f"{_format(value(dataset + '_fp_mae_median'))} |"
        )
    lines.extend(
        [
            "",
            "## Error-distribution tails",
            "",
            "| Dataset | relL2 P90 | relL2 P95 | relL2 P99 | MAE P90 (kPa) | MAE P95 (kPa) | MAE P99 (kPa) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("grouped", "patient"):
        lines.append(
            f"| {dataset} | {_format(value(dataset + '_relL2_p90'))} | "
            f"{_format(value(dataset + '_relL2_p95'))} | "
            f"{_format(value(dataset + '_relL2_p99'))} | "
            f"{_format(value(dataset + '_mae_p90'))} | "
            f"{_format(value(dataset + '_mae_p95'))} | "
            f"{_format(value(dataset + '_mae_p99'))} |"
        )
    lines.extend(
        [
            "",
            "## Peak first-principal Second Piola--Kirchhoff stress and template coordinates",
            "",
            "| Dataset | Peak S1 bias (kPa) | Lower LoA | Upper LoA | Template-coordinate disagreement unmasked (mm) | Masked (mm) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("grouped", "patient"):
        lines.append(
            f"| {dataset} | {_format(value(dataset + '_pws_bias'))} | "
            f"{_format(value(dataset + '_pws_lower_loa'))} | "
            f"{_format(value(dataset + '_pws_upper_loa'))} | "
            f"{_format(value(dataset + '_pws_location_unmasked_median'))} | "
            f"{_format(value(dataset + '_pws_location_masked_median'))} |"
        )
    lines.extend(
        [
            "",
            "## Pressure and uncertainty diagnostics",
            "",
            "| Dataset | Pressure rho | Sextile 1 MAE | Sextile 6 MAE | MC rho | MC sigma (kPa) | MC coverage | Ensemble rho | Ensemble sigma (kPa) | Ensemble coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("grouped", "patient"):
        lines.append(
            f"| {dataset} | {_format(value(dataset + '_pressure_spearman'))} | "
            f"{_format(value(dataset + '_pressure_sextile_1_mae'))} | "
            f"{_format(value(dataset + '_pressure_sextile_6_mae'))} | "
            f"{_format(value(dataset + '_mc_dropout_spearman'))} | "
            f"{_format(value(dataset + '_mc_dropout_mean_sigma'))} | "
            f"{_format(value(dataset + '_mc_dropout_coverage'))} | "
            f"{_format(value(dataset + '_ensemble_spearman'))} | "
            f"{_format(value(dataset + '_ensemble_mean_sigma'))} | "
            f"{_format(value(dataset + '_ensemble_coverage'))} |"
        )
    lines.extend(
        [
            "",
            "## Classical and neural baselines",
            "",
            "| Model | Grouped R2 | Grouped MAE (kPa) | Patient R2 | Patient MAE (kPa) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries["baselines_summary.json"]["results"]["comparison_rows"]:
        lines.append(
            f"| {row['model']} | {_format(row['g_R2'])} | {_format(row['g_MAE'])} | "
            f"{_format(row['p_R2'])} | {_format(row['p_MAE'])} |"
        )
    lines.extend(
        [
            "",
            "## Boundary and legacy component-wise diagnostics",
            "",
            "| Quantity | Grouped | Imaging-derived | Unit |",
            "|---|---:|---:|---|",
            f"| Boundary-masked median R2 | {_format(value('grouped_boundary_masked_R2_median'))} | {_format(value('patient_boundary_masked_R2_median'))} | dimensionless |",
            f"| Boundary-masked median MAE | {_format(value('grouped_boundary_masked_mae_median'))} | {_format(value('patient_boundary_masked_mae_median'))} | kPa |",
            f"| Predicted C SPD fraction | {_format(value('grouped_predicted_C_spd_fraction'))} | {_format(value('patient_predicted_C_spd_fraction'))} | fraction |",
            "",
            "| Legacy diagnostic | Value |",
            "|---|---:|",
            f"| Mean-field C moderate fraction | {_format(value('meanfield_C_moderate_fraction'))} |",
            f"| Mean-field C high fraction | {_format(value('meanfield_C_high_fraction'))} |",
            f"| Mean-field S moderate fraction | {_format(value('meanfield_S_moderate_fraction'))} |",
            f"| Mean-field S high fraction | {_format(value('meanfield_S_high_fraction'))} |",
            f"| Network S moderate fraction | {_format(value('network_S_legacy_moderate_fraction'))} |",
            f"| Network S high fraction | {_format(value('network_S_legacy_high_fraction'))} |",
        ]
    )
    lines.extend(["", "## Source summaries", "", "| File | SHA-256 |", "|---|---|"])
    for filename in REQUIRED_SUMMARIES:
        lines.append(f"| `{filename}` | `{source_records[filename]['sha256']}` |")
    lines.extend(
        [
            "",
            "All values above are copied from the JSON paths recorded in `claim_traceability.csv`; "
            "they are not independent hand-entered results.",
        ]
    )
    return "\n".join(lines)


def main():
    C.init_run()
    outputs = Path(C.OUTPUTS_DIR)
    summaries = _load_summaries(outputs)
    source_records = {
        filename: artifact_record(outputs / filename, C.RUN_ROOT)
        for filename in REQUIRED_SUMMARIES
    }
    claims = _claim_records(summaries, source_records)
    values_path = outputs / "manuscript_values.json"
    write_json(
        values_path,
        {
            "schema_version": 1,
            "run_id": C.RUN_ID,
            "generated_utc": utc_now(),
            "status": "pending Gate R and I12 manuscript propagation",
            "hash_algorithms": {
                "array": ARRAY_HASH_ALGORITHM,
                "file": FILE_HASH_ALGORITHM,
            },
            "source_summaries": source_records,
            "values": claims,
        },
    )
    trace_rows = [
        {
            "claim_key": key,
            "run_id": C.RUN_ID,
            "array_hash_algorithm": ARRAY_HASH_ALGORITHM,
            "file_hash_algorithm": FILE_HASH_ALGORITHM,
            **record,
        }
        for key, record in claims.items()
    ]
    trace_path = write_csv(outputs / "claim_traceability.csv", pd.DataFrame(trace_rows))
    report_path = write_text(
        outputs / "RESULTS_REPORT.md", _render_report(claims, summaries, source_records)
    )
    bundle = {
        "hash_algorithms": {
            "array": ARRAY_HASH_ALGORITHM,
            "file": FILE_HASH_ALGORITHM,
        },
        "manuscript_values": artifact_record(values_path, C.RUN_ROOT),
        "claim_traceability": artifact_record(trace_path, C.RUN_ROOT),
        "results_report": artifact_record(report_path, C.RUN_ROOT),
    }
    C.update_manifest({"result_bundle": bundle})
    print(f"saved {values_path}, {trace_path} and {report_path}")


if __name__ == "__main__":
    main()
