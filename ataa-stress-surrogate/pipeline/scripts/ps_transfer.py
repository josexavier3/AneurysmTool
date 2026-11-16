"""Stage 4 (transfer) -- apply the trained grouped model to imaging-derived (patient) data.

The crucial out-of-distribution test: the grouped (leakage-free) model was trained only on
synthetic PCA-sampled geometries; here it is applied WITHOUT retraining to the 94 patient
geometries (x20 loading states). The exact model architecture and scalers persisted by the
training run are loaded; evaluation never refits a scaler.

Usage: python ps_transfer.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # ataa_work/analysis
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.inference import load_run_bundle, predict_raw
from result_provenance import artifact_record, result_document, write_csv, write_json
from stage4_metrics import block_metrics, summarise, S_LEN, DYNE_CM2_TO_KPA

PRIMARY_RUN_NAME = "cardiac_grouped_free_seed42"


def main():
    C.init_run()
    PS = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    bundle = load_run_bundle(C.MODELS_DIR, PRIMARY_RUN_NAME)

    # --- Patient inputs/outputs ---
    Xp = np.loadtxt(os.path.join(PS, "ps_sim_inputs_coupled.txt"), delimiter=",").astype("float32")
    Yp = pd.read_csv(os.path.join(PS, "ps_sim_outputs_coupled.txt"), header=None,
                     sep=",", dtype=np.float32).to_numpy()
    good = ~np.isnan(Yp).any(1)
    patient_rows = np.flatnonzero(good)
    Xp, Yp = Xp[good], Yp[good]
    print(f"patient samples: {good.sum()} (excluded {int((~good).sum())} NaN rows) of {len(good)}")

    pred = predict_raw(bundle, Xp)

    Sp, Sr = pred[:, :S_LEN] * DYNE_CM2_TO_KPA, Yp[:, :S_LEN] * DYNE_CM2_TO_KPA
    Cp, Cr = pred[:, S_LEN:], Yp[:, S_LEN:]
    dS = block_metrics(Sp, Sr); dC = block_metrics(Cp, Cr)
    stress_summary = summarise(dS, "PATIENT  S (kPa)")
    deformation_summary = summarise(dC, "PATIENT  C")
    stress_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "metrics_patient_S.csv"),
        pd.concat([pd.Series(patient_rows, name="source_row"), dS], axis=1),
    )
    deformation_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "metrics_patient_C.csv"),
        pd.concat([pd.Series(patient_rows, name="source_row"), dC], axis=1),
    )
    summary_path = os.path.join(C.OUTPUTS_DIR, "metrics_patient_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="imaging_derived_transfer_field_metrics",
        model_runs=[PRIMARY_RUN_NAME],
        definitions={
            "training_domain": "synthetic only",
            "evaluation_domain": "imaging-derived; no retraining",
            "case_weighting": "equal",
            "stress_unit": "kPa",
            "right_cauchy_green_unit": "dimensionless",
        },
        results={
            "n_source_rows": int(len(good)),
            "n_excluded_nonfinite_rows": int((~good).sum()),
            "n_evaluated_rows": int(good.sum()),
            "S_kPa": stress_summary,
            "C": deformation_summary,
        },
    )
    document["artefacts"] = {
        "stress_per_case": artifact_record(stress_path, C.RUN_ROOT),
        "deformation_per_case": artifact_record(deformation_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"\nsaved metrics_patient_[S|C].csv and {summary_path}")


if __name__ == "__main__":
    main()
