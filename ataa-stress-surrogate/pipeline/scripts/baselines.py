"""REVIEW s6 -- classical ROM baselines vs the DNN, first principal S (kPa).

Reduces the S field with POD (PCA), regresses the POD coefficients on the 10 inputs with
several classical methods, reconstructs, and reports first-principal-stress metrics on the
grouped test and patient sets. Answers "is a deep network needed?".

These are fixed, untuned contextual baselines, not a matched architecture or
hyperparameter comparison. Kernel ridge uses a documented 6,000-row training
subsample for computational tractability.
"""
import argparse
import os, sys, time


def _parser():
    return argparse.ArgumentParser(
        description="Evaluate fixed POD-regression baselines on an existing evidence run."
    )


PARSER = _parser()
if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
    PARSER.parse_args()

import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.metrics import per_sample_field_metrics
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal
from result_provenance import (
    artifact_record,
    result_document,
    sha256_array,
    write_csv,
    write_json,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

NN, S_LEN, TO_KPA, K = C.N_NODES, C.N_NODES * 6, DYNE_CM2_TO_KPA, 64


def fp_metrics(pred_block, ref_fp):
    fp = first_principal(pred_block, n_nodes=NN, scale=TO_KPA)
    values = per_sample_field_metrics(fp, ref_fp)
    return tuple(float(np.nanmedian(values[name])) for name in ("R2", "relL2", "mae"))


def main():
    PARSER.parse_args()
    C.init_run()
    X = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype("float32")
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    sp = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz")); tr, te = sp["train"], sp["test"]

    Str = np.asarray(Y[tr], dtype=np.float32)[:, :S_LEN]
    Ste = np.asarray(Y[te], dtype=np.float32)[:, :S_LEN]
    PS = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    Xp = np.loadtxt(os.path.join(PS, "ps_sim_inputs_coupled.txt"), delimiter=",").astype("float32")
    Yp = pd.read_csv(os.path.join(PS, "ps_sim_outputs_coupled.txt"), header=None, sep=",",
                     dtype=np.float32).to_numpy()
    good = ~np.isnan(Yp).any(1); Xp, Sp = Xp[good], Yp[good][:, :S_LEN]

    # reference first principal
    fp_te = first_principal(Ste, n_nodes=NN, scale=TO_KPA)
    fp_p = first_principal(Sp, n_nodes=NN, scale=TO_KPA)

    # inputs scaled; POD on training S
    xs = StandardScaler().fit(X[tr])
    Xtr, Xte, Xpp = xs.transform(X[tr]), xs.transform(X[te]), xs.transform(Xp)
    pod = PCA(n_components=K, random_state=C.SEED).fit(Str)
    Ctr = pod.transform(Str)
    print(f"POD {K} modes explain {pod.explained_variance_ratio_.sum()*100:.2f}% of S variance")

    configurations = {}

    def evaluate(name, model, configuration, poly=None):
        t0 = time.time()
        Xt = poly.fit_transform(Xtr) if poly else Xtr
        Xv = poly.transform(Xte) if poly else Xte
        Xq = poly.transform(Xpp) if poly else Xpp
        model.fit(Xt, Ctr)
        rec_te = pod.inverse_transform(model.predict(Xv)).astype(np.float32)
        rec_p = pod.inverse_transform(model.predict(Xq)).astype(np.float32)
        g = fp_metrics(rec_te, fp_te); p = fp_metrics(rec_p, fp_p)
        elapsed = time.time() - t0
        print(f"  {name:22s} grouped R2={g[0]:.3f} relL2={g[1]:.3f} MAE={g[2]:.2f} | "
              f"patient R2={p[0]:.3f} relL2={p[1]:.3f} MAE={p[2]:.2f}  ({elapsed:.0f}s)")
        configurations[name] = configuration
        return {"model": name, "g_R2": g[0], "g_relL2": g[1], "g_MAE": g[2],
                "p_R2": p[0], "p_relL2": p[1], "p_MAE": p[2],
                "wall_time_seconds": elapsed}

    rows = []
    print("\n=== BASELINES (first principal S, kPa; median) ===")
    rows.append(evaluate("POD+Linear", LinearRegression(), {"regressor": "LinearRegression"}))
    rows.append(evaluate("POD+Ridge", Ridge(alpha=1.0), {"regressor": "Ridge", "alpha": 1.0}))
    rows.append(evaluate(
        "POD+Poly2",
        LinearRegression(),
        {"regressor": "LinearRegression", "polynomial_degree": 2, "include_bias": False},
        PolynomialFeatures(2, include_bias=False),
    ))
    rows.append(evaluate(
        "POD+RandomForest",
        RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=C.SEED),
        {"regressor": "RandomForestRegressor", "n_estimators": 100, "random_state": C.SEED},
    ))
    # kernel ridge on a training subsample (exact kernel is O(n^3))
    rng = np.random.default_rng(C.SEED); ss = rng.choice(len(tr), size=6000, replace=False)
    kr = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.05)
    t0 = time.time(); kr.fit(Xtr[ss], Ctr[ss])
    g = fp_metrics(pod.inverse_transform(kr.predict(Xte)).astype(np.float32), fp_te)
    p = fp_metrics(pod.inverse_transform(kr.predict(Xpp)).astype(np.float32), fp_p)
    print(f"  {'POD+KernelRidge(6k)':22s} grouped R2={g[0]:.3f} relL2={g[1]:.3f} MAE={g[2]:.2f} | "
          f"patient R2={p[0]:.3f} relL2={p[1]:.3f} MAE={p[2]:.2f}  ({time.time()-t0:.0f}s)")
    kr_elapsed = time.time() - t0
    rows.append({"model": "POD+KernelRidge", "g_R2": g[0], "g_relL2": g[1], "g_MAE": g[2],
                 "p_R2": p[0], "p_relL2": p[1], "p_MAE": p[2],
                 "wall_time_seconds": kr_elapsed})
    configurations["POD+KernelRidge"] = {
        "regressor": "KernelRidge",
        "kernel": "rbf",
        "alpha": 1.0,
        "gamma": 0.05,
        "training_subsample_size": int(len(ss)),
        "selection_seed": C.SEED,
    }
    rows.append(evaluate(
        "POD+ShallowMLP",
        MLPRegressor(hidden_layer_sizes=(256,), max_iter=300, random_state=C.SEED),
        {"regressor": "MLPRegressor", "hidden_layer_sizes": [256], "max_iter": 300,
         "random_state": C.SEED},
    ))

    # DNN medians from saved per-sample CSVs
    dg = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_grouped.csv"))
    dp = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_patient.csv"))
    print(f"  {'DNN (deep, ours)':22s} grouped R2={dg['R2'].median():.3f} "
          f"relL2={dg['relL2'].median():.3f} MAE={dg['mae'].median():.2f} | "
          f"patient R2={dp['R2'].median():.3f} relL2={dp['relL2'].median():.3f} MAE={dp['mae'].median():.2f}")
    rows.append({"model": "DNN(ours)", "g_R2": dg["R2"].median(), "g_relL2": dg["relL2"].median(),
                 "g_MAE": dg["mae"].median(), "p_R2": dp["R2"].median(),
                 "p_relL2": dp["relL2"].median(), "p_MAE": dp["mae"].median(),
                 "wall_time_seconds": np.nan})
    configurations["DNN(ours)"] = {
        "source": "principal_grouped.csv and principal_patient.csv",
        "role": "comparison model; trained outside this baseline script",
    }

    comparison_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "baselines_comparison.csv"), pd.DataFrame(rows)
    )
    subsample_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "baseline_kernel_ridge_subsample.csv"),
        pd.DataFrame(
            {
                "training_split_position": ss,
                "source_row": tr[ss],
                "selection_seed": C.SEED,
            }
        ),
    )
    summary_path = os.path.join(C.OUTPUTS_DIR, "baselines_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="classical_pod_regression_baselines",
        definitions={
            "target": "first-principal Second Piola--Kirchhoff stress",
            "stress_unit": "kPa",
            "case_aggregation": "median of per-case metrics",
            "comparison_scope": (
                "fixed untuned reference configurations; not a matched architecture, capacity, "
                "optimisation-budget or hyperparameter study"
            ),
            "warning_policy": "estimator warnings are not suppressed",
            "pod_components": K,
            "input_scaler_scope": "grouped training rows only",
            "pod_fit_scope": "grouped training S rows only",
            "seed": C.SEED,
        },
        results={
            "pod": {
                "explained_variance_fraction": float(pod.explained_variance_ratio_.sum()),
                "explained_variance_ratio": pod.explained_variance_ratio_,
            },
            "input_scaler": {
                "mean_sha256": sha256_array(xs.mean_),
                "scale_sha256": sha256_array(xs.scale_),
            },
            "configurations": configurations,
            "comparison_rows": rows,
        },
    )
    document["artefacts"] = {
        "comparison_table": artifact_record(comparison_path, C.RUN_ROOT),
        "kernel_ridge_subsample": artifact_record(subsample_path, C.RUN_ROOT),
    }
    write_json(summary_path, document)
    print(f"\nsaved {comparison_path}, {subsample_path} and {summary_path}")


if __name__ == "__main__":
    main()
