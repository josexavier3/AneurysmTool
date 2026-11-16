"""Persist Monte Carlo dropout uncertainty for the stress-component field."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C
from isolated_statistics import spearmanr_isolated
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.inference import load_run_bundle
from ataa_surrogate.scaling import restore_outputs, standardise_inputs
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA
from ps_transfer import PRIMARY_RUN_NAME
from result_provenance import (
    artifact_record,
    finite_summary,
    result_document,
    write_csv,
    write_json,
)


S_LEN = C.N_NODES * 6


def enable_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_stats(bundle, raw_inputs, passes, seed):
    """Return component-wise mean and population SD in kPa over dropout passes."""

    torch.manual_seed(seed)
    x = torch.tensor(standardise_inputs(raw_inputs, bundle.scalers), dtype=torch.float32)
    n = x.shape[0]
    first = np.zeros((n, S_LEN), dtype=np.float64)
    second = np.zeros((n, S_LEN), dtype=np.float64)
    with torch.no_grad():
        for _ in range(passes):
            prediction = restore_outputs(bundle.model(x).numpy(), bundle.scalers)[:, :S_LEN]
            prediction = prediction * DYNE_CM2_TO_KPA
            first += prediction
            second += prediction * prediction
    mean = first / passes
    standard_deviation = np.sqrt(np.maximum(second / passes - mean * mean, 0.0))
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def evaluate(bundle, raw_inputs, reference_kpa, label, source_rows, passes, seed):
    mean, standard_deviation = mc_stats(bundle, raw_inputs, passes, seed)
    sigma_bar = standard_deviation.mean(1)
    mae = np.abs(mean - reference_kpa).mean(1)
    rho, p_value = spearmanr_isolated(sigma_bar, mae)
    covered = np.abs(reference_kpa - mean) <= 1.96 * standard_deviation
    per_case_coverage = covered.mean(1)
    frame = pd.DataFrame(
        {
            "dataset": label,
            "source_row": np.asarray(source_rows, dtype=np.int64),
            "mean_component_sigma_kpa": sigma_bar,
            "mean_component_mae_kpa": mae,
            "pointwise_95_coverage_fraction": per_case_coverage,
        }
    )
    print(
        f"[{label}] n={len(mae)}, rho={rho:+.3f}, mean sigma={sigma_bar.mean():.3f} kPa, "
        f"mean MAE={mae.mean():.3f} kPa, coverage={covered.mean()*100:.1f}%"
    )
    return {
        "n_cases": int(len(mae)),
        "dropout_seed": int(seed),
        "spearman_sigma_vs_mae": None if not np.isfinite(rho) else float(rho),
        "spearman_p_value": None if not np.isfinite(p_value) else float(p_value),
        "mean_component_sigma_kpa": finite_summary(sigma_bar),
        "mean_component_mae_kpa": finite_summary(mae),
        "pointwise_95_coverage_fraction": float(covered.mean()),
        "per_case_coverage_fraction": finite_summary(per_case_coverage),
    }, frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--synthetic-subsample", type=int, default=2500)
    arguments = parser.parse_args()
    if arguments.passes < 2:
        parser.error("--passes must be at least 2")
    if arguments.synthetic_subsample < 1:
        parser.error("--synthetic-subsample must be positive")
    return arguments


def main():
    C.init_run()
    args = parse_args()
    X = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype("float32")
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    te = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))["test"]
    bundle = load_run_bundle(C.MODELS_DIR, PRIMARY_RUN_NAME)
    enable_dropout(bundle.model)

    rng = np.random.default_rng(args.seed)
    split_positions = rng.choice(
        len(te), size=min(args.synthetic_subsample, len(te)), replace=False
    )
    selected_rows = te[split_positions]
    selection_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "mc_dropout_selected_rows.csv"),
        pd.DataFrame(
            {
                "grouped_test_position": split_positions,
                "source_row": selected_rows,
                "selection_seed": args.seed,
            }
        ),
    )
    grouped_reference = (
        np.asarray(Y[selected_rows], dtype=np.float32)[:, :S_LEN] * DYNE_CM2_TO_KPA
    )
    grouped, grouped_frame = evaluate(
        bundle,
        X[selected_rows],
        grouped_reference,
        "grouped",
        selected_rows,
        args.passes,
        args.seed,
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
    patient_reference = Yp[good][:, :S_LEN] * DYNE_CM2_TO_KPA
    patient, patient_frame = evaluate(
        bundle,
        Xp[good],
        patient_reference,
        "patient",
        np.flatnonzero(good),
        args.passes,
        args.seed + 1,
    )
    cases_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "mc_dropout_case_metrics.csv"),
        pd.concat([grouped_frame, patient_frame], ignore_index=True),
    )

    summary_path = os.path.join(C.OUTPUTS_DIR, "mc_dropout_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="monte_carlo_dropout_stress_component_uncertainty",
        model_runs=[PRIMARY_RUN_NAME],
        definitions={
            "passes": args.passes,
            "selection_seed": args.seed,
            "synthetic_subsample_requested": args.synthetic_subsample,
            "stress_unit": "kPa",
            "standard_deviation": "population SD across stochastic forward passes",
            "coverage_rule": "abs(reference - predictive_mean) <= 1.96 * predictive_population_SD",
            "coverage_weighting": "all case-component-node values equally weighted",
            "uncertainty_discrimination": "Spearman correlation of per-case mean component SD and MAE",
            "spearman_execution": (
                "SciPy evaluated in a clean child process because this Windows environment "
                "links incompatible SciPy and PyTorch OpenMP runtimes"
            ),
        },
        results={"grouped": grouped, "patient": patient},
    )
    document["artefacts"] = {
        "selected_synthetic_rows": artifact_record(selection_path, C.RUN_ROOT),
        "per_case_metrics": artifact_record(cases_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"saved {selection_path}, {cases_path} and {summary_path}")


if __name__ == "__main__":
    main()
