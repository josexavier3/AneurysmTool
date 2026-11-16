"""Persist error percentiles and empirical relative-L2 CDFs."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from ps_transfer import PRIMARY_RUN_NAME
from result_provenance import (
    artifact_record,
    finite_summary,
    result_document,
    write_csv,
    write_json,
)


def analyse(label):
    frame = pd.read_csv(os.path.join(C.OUTPUTS_DIR, f"principal_{label}.csv"))
    result = {"n_cases": int(len(frame)), "metrics": {}}
    print(f"\n=== {label} (first principal S, kPa) — n={len(frame)} ===")
    for column in ("relL2", "mae", "rmse", "R2", "ccc"):
        if column not in frame:
            raise RuntimeError(f"principal_{label}.csv is missing required column {column!r}")
        result["metrics"][column] = finite_summary(frame[column].to_numpy())
        stats = result["metrics"][column]
        print(
            f"  {column:6s} P50={stats['median']:.4g} P90={stats['p90']:.4g} "
            f"P95={stats['p95']:.4g} P99={stats['p99']:.4g}"
        )

    rel_l2 = frame["relL2"].to_numpy(dtype=np.float64)
    finite = rel_l2[np.isfinite(rel_l2)]
    if finite.size != rel_l2.size:
        raise RuntimeError(f"{label}: relative-L2 CDF cannot omit degenerate/non-finite cases")
    sorted_values = np.sort(finite)
    cdf_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"cdf_relL2_{label}.csv"),
        pd.DataFrame(
            {
                "relL2": sorted_values,
                "cdf": np.arange(1, len(sorted_values) + 1) / len(sorted_values),
            }
        ),
    )
    result["artefact"] = artifact_record(cdf_path, C.RUN_ROOT)
    return result


def main():
    C.init_run()
    results = {label: analyse(label) for label in ("grouped", "patient")}
    summary_path = os.path.join(C.OUTPUTS_DIR, "percentiles_summary.json")
    write_json(
        summary_path,
        result_document(
            run_id=C.RUN_ID,
            analysis="first_principal_stress_error_percentiles",
            model_runs=[PRIMARY_RUN_NAME],
            definitions={
                "case_weighting": "equal",
                "quantile_method": "NumPy linear interpolation",
                "mae_and_rmse_unit": "kPa",
                "relL2_R2_ccc_unit": "dimensionless",
                "cdf": "empirical cumulative fraction i/n after ascending sort",
            },
            results=results,
        ),
    )
    print(f"saved CDF CSVs and {summary_path}")


if __name__ == "__main__":
    main()
