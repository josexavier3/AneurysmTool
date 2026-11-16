"""Persist deep-ensemble accuracy variation and component-field uncertainty."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from isolated_statistics import spearmanr_isolated
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.inference import load_run_bundle, predict_raw
from ataa_surrogate.metrics import per_sample_field_metrics, summarise_metric_arrays
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal
from result_provenance import (
    artifact_record,
    discover_model_runs,
    finite_summary,
    load_run_manifest,
    result_document,
    write_csv,
    write_json,
)


NN = C.N_NODES
S_LEN = C.N_NODES * 6
TO_KPA = DYNE_CM2_TO_KPA
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)


def summarise_first_principal(predicted_s, reference_fp):
    predicted_fp = first_principal(predicted_s, n_nodes=NN, scale=TO_KPA)
    arrays = per_sample_field_metrics(predicted_fp, reference_fp)
    return summarise_metric_arrays(arrays)


def uncertainty_summary(stack_s_kpa, reference_s_kpa, label, source_rows):
    ensemble_mean = stack_s_kpa.mean(0)
    ensemble_sd = stack_s_kpa.std(0, ddof=0)
    sigma = ensemble_sd.mean(1)
    mae = np.abs(ensemble_mean - reference_s_kpa).mean(1)
    rho, p_value = spearmanr_isolated(sigma, mae)
    covered = np.abs(reference_s_kpa - ensemble_mean) <= 1.96 * ensemble_sd
    per_case_coverage = covered.mean(1)
    frame = pd.DataFrame(
        {
            "dataset": label,
            "source_row": np.asarray(source_rows, dtype=np.int64),
            "mean_component_sigma_kpa": sigma,
            "mean_component_mae_kpa": mae,
            "pointwise_95_coverage_fraction": per_case_coverage,
        }
    )
    summary = {
        "n_cases": int(len(mae)),
        "mean_component_sigma_kpa": finite_summary(sigma),
        "mean_component_mae_kpa": finite_summary(mae),
        "spearman_sigma_vs_mae": None if not np.isfinite(rho) else float(rho),
        "spearman_p_value": None if not np.isfinite(p_value) else float(p_value),
        "pointwise_95_coverage_fraction": float(covered.mean()),
        "per_case_coverage_fraction": finite_summary(per_case_coverage),
    }
    print(
        f"[{label}] mean sigma={sigma.mean():.3f} kPa, mean MAE={mae.mean():.3f} kPa, "
        f"rho={rho:+.3f}, coverage={covered.mean()*100:.1f}%"
    )
    return summary, ensemble_mean, frame


def main():
    C.init_run()
    manifest = load_run_manifest(C.MANIFEST_PATH, C.RUN_ID)
    members = discover_model_runs(
        manifest, split="grouped", variant="free", seeds=ENSEMBLE_SEEDS, tag=""
    )
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    te = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))["test"]
    grouped_reference = np.asarray(Y[te], np.float32)
    grouped_reference_fp = first_principal(
        grouped_reference[:, :S_LEN], n_nodes=NN, scale=TO_KPA
    )

    member_rows = []
    grouped_stack = []
    member_summaries = {}
    for member in members:
        prediction = np.load(os.path.join(C.MODELS_DIR, f"{member}_pred_test.npy"))
        summary = summarise_first_principal(prediction[:, :S_LEN], grouped_reference_fp)
        record = manifest["training_runs"][member]
        seed = int(record["seed"])
        member_summaries[member] = {"seed": seed, "first_principal_stress": summary}
        member_rows.append(
            {
                "model_run": member,
                "seed": seed,
                "median_R2": summary["R2"]["median"],
                "median_relL2": summary["relL2"]["median"],
                "median_mae_kpa": summary["mae"]["median"],
            }
        )
        grouped_stack.append(prediction[:, :S_LEN] * TO_KPA)

    grouped_stack = np.stack(grouped_stack)
    member_r2 = np.asarray([row["median_R2"] for row in member_rows], dtype=np.float64)
    member_mae = np.asarray([row["median_mae_kpa"] for row in member_rows], dtype=np.float64)
    # Seed-to-seed variability is a sample spread over the five seeds actually drawn, so the
    # sample SD (ddof=1) is the correct estimator and matches result_provenance.finite_summary.
    member_variation = {
        "n_members": len(member_rows),
        "median_R2_across_members_mean": float(member_r2.mean()),
        "median_R2_across_members_sd_ddof1": float(member_r2.std(ddof=1)),
        "median_mae_kpa_across_members_mean": float(member_mae.mean()),
        "median_mae_kpa_across_members_sd_ddof1": float(member_mae.std(ddof=1)),
    }
    grouped_uq, ensemble_mean_grouped, grouped_cases = uncertainty_summary(
        grouped_stack,
        grouped_reference[:, :S_LEN] * TO_KPA,
        "grouped",
        te,
    )
    ensemble_accuracy = summarise_first_principal(
        ensemble_mean_grouped / TO_KPA, grouped_reference_fp
    )
    del grouped_stack, ensemble_mean_grouped

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
    patient_stack = []
    for member in members:
        prediction = predict_raw(load_run_bundle(C.MODELS_DIR, member), Xp[good])
        patient_stack.append(prediction[:, :S_LEN] * TO_KPA)
    patient_stack = np.stack(patient_stack)
    patient_uq, _, patient_cases = uncertainty_summary(
        patient_stack,
        Yp[good][:, :S_LEN] * TO_KPA,
        "patient",
        np.flatnonzero(good),
    )

    member_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "ensemble_member_metrics.csv"),
        pd.DataFrame(member_rows),
    )
    cases_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "ensemble_case_metrics.csv"),
        pd.concat([grouped_cases, patient_cases], ignore_index=True),
    )
    summary_path = os.path.join(C.OUTPUTS_DIR, "ensemble_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="deep_ensemble_accuracy_and_stress_component_uncertainty",
        model_runs=members,
        definitions={
            "member_discovery": (
                "exact untagged grouped/free production stems for seeds 42--46"
            ),
            "ensemble_standard_deviation": "population SD across five independently seeded models",
            "coverage_rule": "abs(reference - ensemble_mean) <= 1.96 * ensemble_population_SD",
            "coverage_weighting": "all case-component-node values equally weighted",
            "stress_unit": "kPa",
            "spearman_execution": (
                "SciPy evaluated in a clean child process because this Windows environment "
                "links incompatible SciPy and PyTorch OpenMP runtimes"
            ),
        },
        results={
            "members": member_summaries,
            "member_variation": member_variation,
            "grouped": {
                "uncertainty": grouped_uq,
                "ensemble_mean_first_principal_stress": ensemble_accuracy,
            },
            "patient": {"uncertainty": patient_uq},
        },
    )
    document["artefacts"] = {
        "per_member_metrics": artifact_record(member_path, C.RUN_ROOT),
        "per_case_uncertainty": artifact_record(cases_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"saved {member_path}, {cases_path} and {summary_path}")


if __name__ == "__main__":
    main()
