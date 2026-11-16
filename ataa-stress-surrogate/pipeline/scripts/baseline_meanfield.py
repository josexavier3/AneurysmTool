"""Persist the train-mean baseline for stress and deformation outputs."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.metrics import per_sample_field_metrics, summarise_metric_arrays
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal
from ps_transfer import PRIMARY_RUN_NAME
from result_provenance import (
    artifact_record,
    result_document,
    sha256_array,
    write_csv,
    write_json,
)


NN = C.N_NODES
S_LEN = C.N_NODES * 6
TO_KPA = DYNE_CM2_TO_KPA


def main():
    C.init_run()
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    split = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))
    train_rows, test_rows = split["train"], split["test"]

    training_mean = np.asarray(Y[train_rows], dtype=np.float32).mean(0)
    reference = np.asarray(Y[test_rows], dtype=np.float32)
    mean_prediction = np.broadcast_to(training_mean, reference.shape)

    mean_stress_fp = first_principal(
        mean_prediction[:, :S_LEN], n_nodes=NN, scale=TO_KPA
    )
    reference_stress_fp = first_principal(
        reference[:, :S_LEN], n_nodes=NN, scale=TO_KPA
    )
    stress_arrays = per_sample_field_metrics(mean_stress_fp, reference_stress_fp)
    deformation_arrays = per_sample_field_metrics(
        mean_prediction[:, S_LEN:], reference[:, S_LEN:]
    )
    columns = {"source_row": test_rows}
    for name, values in stress_arrays.items():
        columns[f"stress_first_principal_{name}"] = values
    for name, values in deformation_arrays.items():
        columns[f"deformation_component_{name}"] = values
    cases_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "meanfield_case_metrics.csv"),
        pd.DataFrame(columns),
    )

    model_path = os.path.join(C.OUTPUTS_DIR, "principal_grouped.csv")
    if not os.path.exists(model_path):
        raise RuntimeError(f"run stage4_principal.py before the mean-field comparison: {model_path}")
    model = pd.read_csv(model_path)
    model_summary = summarise_metric_arrays(
        {name: model[name].to_numpy() for name in stress_arrays}
    )
    stress_summary = summarise_metric_arrays(stress_arrays)
    deformation_summary = summarise_metric_arrays(deformation_arrays)
    print(
        "mean-field first-principal-stress median: "
        f"R2={stress_summary['R2']['median']:.4g}, "
        f"relL2={stress_summary['relL2']['median']:.4g}, "
        f"MAE={stress_summary['mae']['median']:.4g} kPa"
    )

    summary_path = os.path.join(C.OUTPUTS_DIR, "meanfield_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="training_mean_field_baseline",
        model_runs=[PRIMARY_RUN_NAME],
        definitions={
            "baseline": "repeat the grouped-training-set mean output field for every test case",
            "fit_scope": "grouped training rows only",
            "stress_quantity": "first-principal stress in kPa",
            "deformation_quantity": "full six-component Right Cauchy--Green field, dimensionless",
            "case_weighting": "equal",
        },
        results={
            "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows)),
            "training_mean_sha256": sha256_array(training_mean),
            "meanfield_first_principal_stress": stress_summary,
            "meanfield_deformation_components": deformation_summary,
            "dnn_first_principal_stress": model_summary,
        },
    )
    document["artefacts"] = {
        "per_case_metrics": artifact_record(cases_path, C.RUN_ROOT),
        "dnn_per_case_metrics": artifact_record(model_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"saved {cases_path} and {summary_path}")


if __name__ == "__main__":
    main()
