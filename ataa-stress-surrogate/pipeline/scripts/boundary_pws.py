"""Boundary-masked metrics and template-coordinate peak-node disagreement.

- Recomputes first-principal-stress metrics EXCLUDING the inlet/outlet boundary nodes
  (which the Discussion calls boundary-condition artefacts) and compares masked vs unmasked.
- Peak-node disagreement: Euclidean chord distance (mm), not geodesic distance,
  between predicted and reference argmax coordinates on one fixed ordered template.
  It is not a patient-specific anatomical distance.
"""
import argparse
import os, sys


def _parser():
    return argparse.ArgumentParser(
        description="Evaluate boundary sensitivity and peak-node disagreement on a fixed template."
    )


PARSER = _parser()
if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
    PARSER.parse_args()

import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.inference import load_run_bundle, predict_raw
from ataa_surrogate.metrics import (
    euclidean_peak_location_error,
    per_sample_field_metrics,
    summarise_metric_arrays,
)
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal
from ps_transfer import PRIMARY_RUN_NAME
from result_provenance import (
    artifact_record,
    finite_summary,
    result_document,
    sha256_array,
    write_csv,
    write_json,
)

NN, S_LEN, TO_KPA = C.N_NODES, C.N_NODES * 6, DYNE_CM2_TO_KPA


def metric_arrays(fp_pred, fp_ref, mask):
    return per_sample_field_metrics(fp_pred[:, mask], fp_ref[:, mask])


def report(fp_pred, fp_ref, label, source_rows, *, points, boundary_mask):
    interior = ~boundary_mask
    unmasked = metric_arrays(fp_pred, fp_ref, np.ones(NN, bool))
    masked = metric_arrays(fp_pred, fp_ref, interior)
    um = tuple(float(np.nanmedian(unmasked[name])) for name in ("R2", "relL2", "mae"))
    ma = tuple(float(np.nanmedian(masked[name])) for name in ("R2", "relL2", "mae"))
    print(f"\n[{label}] first principal S (kPa), median")
    print(f"   unmasked: R2={um[0]:.3f} relL2={um[1]:.3f} MAE={um[2]:.2f}")
    print(f"   masked  : R2={ma[0]:.3f} relL2={ma[1]:.3f} MAE={ma[2]:.2f}  (interior {interior.sum()} nodes)")
    # Euclidean chord distance on the common mesh; this is not a geodesic distance.
    dist_unmasked = euclidean_peak_location_error(fp_pred, fp_ref, points)
    dist_masked = euclidean_peak_location_error(
        fp_pred[:, interior], fp_ref[:, interior], points[interior]
    )
    print(f"   peak-node template-coordinate disagreement: median={np.median(dist_unmasked):.2f} mm, "
          f"P90={np.percentile(dist_unmasked,90):.2f} mm, "
          f"within-5mm={np.mean(dist_unmasked<5)*100:.0f}%")

    metric_columns = {"source_row": np.asarray(source_rows, dtype=np.int64)}
    for name, values in unmasked.items():
        metric_columns[f"unmasked_{name}"] = values
    for name, values in masked.items():
        metric_columns[f"masked_{name}"] = values
    metrics_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"boundary_metrics_{label}.csv"),
        pd.DataFrame(metric_columns),
    )

    unmasked_predicted_node = np.argmax(fp_pred, axis=1)
    unmasked_reference_node = np.argmax(fp_ref, axis=1)
    interior_nodes = np.flatnonzero(interior)
    masked_predicted_node = interior_nodes[np.argmax(fp_pred[:, interior], axis=1)]
    masked_reference_node = interior_nodes[np.argmax(fp_ref[:, interior], axis=1)]
    location_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"pws_location_{label}.csv"),
        pd.DataFrame(
            {
                "source_row": np.asarray(source_rows, dtype=np.int64),
                "unmasked_predicted_node": unmasked_predicted_node,
                "unmasked_reference_node": unmasked_reference_node,
                "unmasked_euclidean_distance_mm": dist_unmasked,
                "masked_predicted_node": masked_predicted_node,
                "masked_reference_node": masked_reference_node,
                "masked_euclidean_distance_mm": dist_masked,
            }
        ),
    )
    return {
        "n_cases": int(len(fp_ref)),
        "unmasked_first_principal_stress": summarise_metric_arrays(unmasked),
        "boundary_masked_first_principal_stress": summarise_metric_arrays(masked),
        "pws_location_unmasked_mm": {
            **finite_summary(dist_unmasked),
            "fraction_within_5_mm": float(np.mean(dist_unmasked < 5.0)),
        },
        "pws_location_boundary_masked_mm": {
            **finite_summary(dist_masked),
            "fraction_within_5_mm": float(np.mean(dist_masked < 5.0)),
        },
        "artefacts": {
            "per_case_metrics": artifact_record(metrics_path, C.RUN_ROOT),
            "pws_locations": artifact_record(location_path, C.RUN_ROOT),
        },
    }


def main():
    PARSER.parse_args()
    C.init_run()
    points = np.load(os.path.join(C.OUTPUTS_DIR, "mesh_points.npy"))
    boundary_mask = np.load(os.path.join(C.OUTPUTS_DIR, "boundary_mask.npy"))
    interior = ~boundary_mask
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    sp = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz")); te = sp["test"]
    bundle = load_run_bundle(C.MODELS_DIR, PRIMARY_RUN_NAME)

    pred_g = np.load(os.path.join(C.MODELS_DIR, f"{PRIMARY_RUN_NAME}_pred_test.npy"))
    ref_g = np.asarray(Y[te], np.float32)
    results = {"grouped": report(
        first_principal(pred_g[:, :S_LEN], n_nodes=NN, scale=TO_KPA),
        first_principal(ref_g[:, :S_LEN], n_nodes=NN, scale=TO_KPA),
        "grouped",
        te,
        points=points,
        boundary_mask=boundary_mask,
    )}

    PS = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    Xp = np.loadtxt(os.path.join(PS, "ps_sim_inputs_coupled.txt"), delimiter=",").astype("float32")
    Yp = pd.read_csv(os.path.join(PS, "ps_sim_outputs_coupled.txt"), header=None, sep=",",
                     dtype=np.float32).to_numpy()
    good = ~np.isnan(Yp).any(1)
    pred_p = predict_raw(bundle, Xp[good])
    patient_rows = np.flatnonzero(good)
    results["patient"] = report(
        first_principal(pred_p[:, :S_LEN], n_nodes=NN, scale=TO_KPA),
        first_principal(Yp[good][:, :S_LEN], n_nodes=NN, scale=TO_KPA),
        "patient",
        patient_rows,
        points=points,
        boundary_mask=boundary_mask,
    )

    summary_path = os.path.join(C.OUTPUTS_DIR, "boundary_pws_summary.json")
    write_json(
        summary_path,
        result_document(
            run_id=C.RUN_ID,
            analysis="boundary_masked_metrics_and_peak_wall_stress_location",
            model_runs=[PRIMARY_RUN_NAME],
            definitions={
                "stress_unit": "kPa",
                "location_unit": "mm",
                "location_metric": (
                    "Euclidean chord distance between argmax-node coordinates on one fixed "
                    "ordered template; not patient-specific anatomy and not geodesic"
                ),
                "template_scope": "one common coordinate array reused for every evaluated row",
                "argmax_tie_policy": "first argmax node",
                "boundary_mask_true_nodes": int(boundary_mask.sum()),
                "interior_nodes": int(interior.sum()),
                "boundary_mask_sha256": sha256_array(boundary_mask),
                "mesh_points_sha256": sha256_array(points),
            },
            results=results,
        ),
    )
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
