"""Persist the association between instantaneous pressure and physical stress error."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from result_provenance import (
    artifact_record,
    finite_summary,
    result_document,
    write_csv,
    write_json,
)
from isolated_statistics import spearmanr_isolated
from ps_transfer import PRIMARY_RUN_NAME


P_COL = 9
N_SEXTILES = 6


def analyse(pressure, mae, label, source_rows):
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    mae = np.asarray(mae, dtype=np.float64).reshape(-1)
    source_rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    if pressure.shape != mae.shape or pressure.shape != source_rows.shape:
        raise ValueError(f"{label}: pressure, MAE and source-row arrays must have equal length")
    if not np.isfinite(pressure).all() or not np.isfinite(mae).all():
        raise ValueError(f"{label}: pressure and MAE must be finite")

    rho, _ = spearmanr_isolated(pressure, mae)
    if not np.isfinite(rho):
        raise ValueError(f"{label}: Spearman correlation is undefined")
    edges = np.quantile(pressure, np.linspace(0.0, 1.0, N_SEXTILES + 1))
    bin_index = np.digitize(pressure, edges[1:-1], right=True)
    rows = []
    sextiles = []
    for index in range(N_SEXTILES):
        selected = bin_index == index
        if not np.any(selected):
            raise RuntimeError(f"{label}: pressure sextile {index + 1} is empty")
        values = mae[selected]
        pressure_values = pressure[selected]
        row = {
            "dataset": label,
            "sextile": index + 1,
            "n_cases": int(selected.sum()),
            "pressure_min_raw": float(pressure_values.min()),
            "pressure_max_raw": float(pressure_values.max()),
            "pressure_median_raw": float(np.median(pressure_values)),
            "mae_median_kpa": float(np.median(values)),
            "mae_mean_kpa": float(np.mean(values)),
            "mae_p25_kpa": float(np.quantile(values, 0.25)),
            "mae_p75_kpa": float(np.quantile(values, 0.75)),
        }
        rows.append(row)
        sextiles.append(dict(row))
    print(
        f"[{label}] Spearman(instantaneous pressure, first-principal-stress MAE) "
        f"rho={rho:+.3f}, n={len(mae)} (descriptive; no row-wise p-value)"
    )
    return (
        {
            "n_cases": int(len(mae)),
            "spearman_rho": float(rho),
            "mae_kpa": finite_summary(mae),
            "pressure_raw": finite_summary(pressure),
            "sextiles": sextiles,
        },
        rows,
        pd.DataFrame(
            {
                "dataset": label,
                "source_row": source_rows,
                "instantaneous_pressure_raw": pressure,
                "first_principal_stress_mae_kpa": mae,
                "pressure_sextile": bin_index + 1,
            }
        ),
    )


def main():
    C.init_run()
    X = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype("float32")
    te = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))["test"]
    grouped_metrics = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_grouped.csv"))
    grouped, grouped_rows, grouped_cases = analyse(
        X[te, P_COL], grouped_metrics["mae"].to_numpy(), "grouped", te
    )

    patient_source = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    Xp = np.loadtxt(os.path.join(patient_source, "ps_sim_inputs_coupled.txt"), delimiter=",").astype(
        "float32"
    )
    Yp = pd.read_csv(
        os.path.join(patient_source, "ps_sim_outputs_coupled.txt"),
        header=None,
        sep=",",
        dtype=np.float32,
    ).to_numpy()
    good = ~np.isnan(Yp).any(1)
    patient_metrics = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_patient.csv"))
    patient, patient_rows, patient_cases = analyse(
        Xp[good, P_COL],
        patient_metrics["mae"].to_numpy(),
        "patient",
        np.flatnonzero(good),
    )

    sextile_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "pressure_sextiles.csv"),
        pd.DataFrame(grouped_rows + patient_rows),
    )
    cases_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "pressure_case_metrics.csv"),
        pd.concat([grouped_cases, patient_cases], ignore_index=True),
    )
    summary_path = os.path.join(C.OUTPUTS_DIR, "pressure_dependence_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="instantaneous_pressure_vs_first_principal_stress_error",
        model_runs=[PRIMARY_RUN_NAME],
        definitions={
            "pressure_column_zero_based": P_COL,
            "pressure_source_unit": "pending I06 source audit; no unit inferred by this script",
            "error_metric": "per-case mean absolute first-principal-stress error",
            "error_unit": "kPa",
            "binning": "six empirical quantile bins computed separately per dataset",
            "association": (
                "Spearman rank correlation; descriptive, not causal; no row-wise p-value because "
                "repeated loading/material states are correlated within geometry"
            ),
        },
        results={"grouped": grouped, "patient": patient},
    )
    document["artefacts"] = {
        "sextile_table": artifact_record(sextile_path, C.RUN_ROOT),
        "per_case_values": artifact_record(cases_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"saved {sextile_path}, {cases_path} and {summary_path}")


if __name__ == "__main__":
    main()
