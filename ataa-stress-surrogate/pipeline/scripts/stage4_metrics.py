"""Stage 4 -- metric battery for a trained cardiac-cycle model (S and C), per split.

Reads the saved seed-42 test predictions and the
reference rows from the outputs memmap, and computes packing-agnostic per-sample metrics
for the S block (kPa) and the C block (dimensionless):
  relative L2 error, R^2 vs identity, Pearson R, Lin's CCC, RMSE, MAE, NMAE (range).

Running it for both splits gives the leakage contrast (grouped vs random).

Usage:
    python stage4_metrics.py --split grouped
    python stage4_metrics.py --split random
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.metrics import per_sample_field_metrics, summarise_metric_arrays
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA
from result_provenance import artifact_record, result_document, write_csv, write_json

S_LEN = C.N_NODES * 6          # 8124 (first half = S)


def block_metrics(pred, ref):
    """DataFrame view of the canonical per-sample metric implementation."""
    return pd.DataFrame(per_sample_field_metrics(pred, ref))


def summarise(df, label):
    print(f"\n  [{label}]  (n={len(df)})")
    summary = summarise_metric_arrays({column: df[column].to_numpy() for column in df.columns})
    for column, stats in summary.items():
        print(
            f"    {column:12s} median={stats['median']:.4g}  "
            f"IQR=[{stats['q1']:.4g}, {stats['q3']:.4g}]  mean={stats['mean']:.4g}  "
            f"finite={stats['n_finite']}/{stats['n_total']}"
        )
    return summary


def legacy_agreement(df):
    """Legacy Pearson/range-NMAE thresholds, retained only as a labelled diagnostic."""

    pearson = df["pearson"].to_numpy()
    nmae = df["nmae_range"].to_numpy()
    if not np.isfinite(pearson).all() or not np.isfinite(nmae).all():
        raise RuntimeError("legacy agreement thresholds require finite Pearson and range-NMAE")
    return {
        "moderate_fraction": float(np.mean((pearson > 0.8) & (nmae < 0.2))),
        "high_fraction": float(np.mean((pearson > 0.95) & (nmae < 0.05))),
        "moderate_definition": "Pearson > 0.8 and range-NMAE < 0.2",
        "high_definition": "Pearson > 0.95 and range-NMAE < 0.05",
    }


def main():
    C.init_run()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["grouped", "random"], default="grouped")
    args = ap.parse_args()

    run_name = f"cardiac_{args.split}_free_seed42"
    pred = np.load(os.path.join(C.MODELS_DIR, f"{run_name}_pred_test.npy"))
    te = np.load(os.path.join(C.SPLITS_DIR, f"{args.split}_split.npz"))["test"]
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    ref = np.asarray(Y[te], dtype=np.float32)
    if pred.shape != ref.shape:
        raise RuntimeError(f"Prediction/reference shape mismatch: {pred.shape} vs {ref.shape}")
    print(f"[{args.split}] test samples: {len(te)}")

    # S block in kPa, C block dimensionless
    S_pred, S_ref = pred[:, :S_LEN] * DYNE_CM2_TO_KPA, ref[:, :S_LEN] * DYNE_CM2_TO_KPA
    C_pred, C_ref = pred[:, S_LEN:], ref[:, S_LEN:]

    dS = block_metrics(S_pred, S_ref)
    dC = block_metrics(C_pred, C_ref)
    summary = {
        "split": args.split,
        "n_test": int(len(te)),
        "S_kPa": summarise(dS, "S (kPa)"),
        "C": summarise(dC, "C (dimensionless)"),
        "legacy_componentwise_agreement": {
            "role": "legacy diagnostic only; not a primary endpoint",
            "S": legacy_agreement(dS),
            "C": legacy_agreement(dC),
        },
    }

    stress_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"metrics_{args.split}_S.csv"),
        pd.concat([pd.Series(te, name="source_row"), dS], axis=1),
    )
    deformation_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, f"metrics_{args.split}_C.csv"),
        pd.concat([pd.Series(te, name="source_row"), dC], axis=1),
    )
    summary_path = os.path.join(C.OUTPUTS_DIR, f"metrics_{args.split}_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="six_component_field_metrics",
        model_runs=[run_name],
        definitions={
            "split": args.split,
            "case_weighting": "equal",
            "stress_unit": "kPa",
            "right_cauchy_green_unit": "dimensionless",
            "tensor_component_order": ["11", "22", "33", "12", "23", "13"],
        },
        results=summary,
    )
    document["artefacts"] = {
        "stress_per_case": artifact_record(stress_path, C.RUN_ROOT),
        "deformation_per_case": artifact_record(deformation_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"\nsaved metrics_{args.split}_[S|C].csv + _summary.json")


if __name__ == "__main__":
    main()
