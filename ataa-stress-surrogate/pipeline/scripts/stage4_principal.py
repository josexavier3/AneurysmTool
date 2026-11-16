"""Stage 4 -- first-principal PK2 stress, peak S1, and C-block admissibility.

For the grouped model on (i) the synthetic grouped test set and (ii) the patient set:
  - first principal value of S per node (kPa) -> per-sample metric battery
  - peak first-principal Second Piola--Kirchhoff stress (peak S1) -> Bland--Altman
  - fraction of predicted C tensors that are NOT symmetric positive-definite
    (motivates the Cholesky C->S variant)

Component packing assumed [11,22,33,12,23,13] (as in the MSc). Verified via the C block:
a valid deformation tensor is SPD with eigenvalues ~1; the script prints the reference-C
SPD fraction as a packing sanity check (should be ~100%).
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.inference import load_run_bundle, predict_raw
from ataa_surrogate.metrics import bland_altman
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal, spd_fraction
from result_provenance import (
    artifact_record,
    finite_summary,
    result_document,
    write_csv,
    write_json,
)
from stage4_metrics import block_metrics, summarise, S_LEN
from ps_transfer import PRIMARY_RUN_NAME

NN = C.N_NODES
RANDOM_RUN_NAME = "cardiac_random_free_seed42"


def evaluate(pred, ref, label, source_rows):
    print(f"\n########## {label} ##########")
    # first principal stress (kPa)
    fp_pred = first_principal(pred[:, :S_LEN], n_nodes=NN, scale=DYNE_CM2_TO_KPA)
    fp_ref = first_principal(ref[:, :S_LEN], n_nodes=NN, scale=DYNE_CM2_TO_KPA)
    d = block_metrics(fp_pred, fp_ref)
    principal_summary = summarise(d, f"{label}: first principal stress (kPa)")
    principal_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"principal_{label}.csv"),
        pd.concat(
            [pd.Series(np.asarray(source_rows, dtype=np.int64), name="source_row"), d],
            axis=1,
        ),
    )
    # Peak first-principal Second Piola--Kirchhoff stress (legacy keys retain ``pws``).
    pws_pred, pws_ref = fp_pred.max(1), fp_ref.max(1)
    ba = bland_altman(pws_pred, pws_ref)
    print(f"  peak S1 [kPa]: ref median={np.median(pws_ref):.1f}, pred median={np.median(pws_pred):.1f}")
    print(
        f"  peak S1 Bland-Altman: bias={ba['bias']:+.2f} kPa, "
        f"95% LoA=[{ba['lower_loa']:+.2f}, {ba['upper_loa']:+.2f}]"
    )
    pws_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"pws_{label}.csv"),
        pd.DataFrame(
            {
                "source_row": np.asarray(source_rows, dtype=np.int64),
                "reference_pws_kpa": pws_ref,
                "predicted_pws_kpa": pws_pred,
                "difference_predicted_minus_reference_kpa": pws_pred - pws_ref,
            }
        ),
    )
    # C admissibility (predicted)
    spd = spd_fraction(pred[:, S_LEN:], n_nodes=NN)
    spd_ref = spd_fraction(ref[:, S_LEN:], n_nodes=NN)
    print(f"  predicted C SPD fraction: {spd*100:.2f}%   (reference C: {spd_ref*100:.2f}%)")
    return {
        "n_cases": int(len(ref)),
        "first_principal_stress": {
            "unit": "kPa",
            "per_case_metrics": principal_summary,
        },
        "peak_wall_stress": {
            "quantity": "maximum nodal first-principal Second Piola--Kirchhoff stress (peak S1)",
            "unit": "kPa",
            "reference": finite_summary(pws_ref),
            "predicted": finite_summary(pws_pred),
            "difference_predicted_minus_reference": finite_summary(pws_pred - pws_ref),
            "bland_altman": ba,
        },
        "right_cauchy_green_admissibility": {
            "criterion": "all three eigenvalues exceed scale-aware floating-point tolerance",
            "predicted_spd_fraction": float(spd),
            "reference_spd_fraction": float(spd_ref),
            "n_case_node_tensors": int(len(ref) * NN),
        },
        "artefacts": {
            "principal_metrics": artifact_record(principal_path, C.RUN_ROOT),
            "peak_wall_stress": artifact_record(pws_path, C.RUN_ROOT),
        },
    }


def main():
    C.init_run()
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    te = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))["test"]
    bundle = load_run_bundle(C.MODELS_DIR, PRIMARY_RUN_NAME)

    # grouped synthetic test
    pred_g = np.load(os.path.join(C.MODELS_DIR, f"{PRIMARY_RUN_NAME}_pred_test.npy"))
    results = {
        "grouped": evaluate(
            pred_g,
            np.asarray(Y[te], dtype=np.float32),
            "grouped",
            te,
        )
    }
    random_test = np.load(os.path.join(C.SPLITS_DIR, "random_split.npz"))["test"]
    random_prediction = np.load(
        os.path.join(C.MODELS_DIR, f"{RANDOM_RUN_NAME}_pred_test.npy")
    )
    results["random"] = evaluate(
        random_prediction,
        np.asarray(Y[random_test], dtype=np.float32),
        "random",
        random_test,
    )

    # patient transfer
    PS = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    Xp = np.loadtxt(os.path.join(PS, "ps_sim_inputs_coupled.txt"), delimiter=",").astype("float32")
    Yp = pd.read_csv(os.path.join(PS, "ps_sim_outputs_coupled.txt"), header=None, sep=",",
                     dtype=np.float32).to_numpy()
    good = ~np.isnan(Yp).any(1); Xp, Yp = Xp[good], Yp[good]
    patient_rows = np.flatnonzero(good)
    pred_p = predict_raw(bundle, Xp)
    results["patient"] = evaluate(pred_p, Yp, "patient", patient_rows)

    summary_path = os.path.join(C.OUTPUTS_DIR, "pws_spd_summary.json")
    write_json(
        summary_path,
        result_document(
            run_id=C.RUN_ID,
            analysis="first_principal_peak_wall_stress_and_c_admissibility",
            model_runs=[PRIMARY_RUN_NAME, RANDOM_RUN_NAME],
            definitions={
                "tensor_packing": ["11", "22", "33", "12", "23", "13"],
                "first_principal": "algebraically largest eigenvalue",
                "peak_wall_stress": "maximum first-principal stress over all common-mesh nodes",
                "stress_source_unit": "dyne/cm^2",
                "stress_report_unit": "kPa",
                "stress_to_kpa": DYNE_CM2_TO_KPA,
            },
            results=results,
        ),
    )
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
