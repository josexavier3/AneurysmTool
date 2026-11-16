"""Remaining figures: peak-S1 Bland--Altman, UQ coverage, and spatial error."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.inference import load_run_bundle, predict_raw
from ataa_surrogate.metrics import bland_altman
from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal

NN, S_LEN, TO_KPA = C.N_NODES, C.N_NODES * 6, DYNE_CM2_TO_KPA
OUT = C.FIGURES_DIR
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
MEMBERS = [f"cardiac_grouped_free_seed{seed}" for seed in (42, 43, 44, 45, 46)]
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
                     "figure.autolayout": True})


def ensemble_S(paths_loader):
    """Return ensemble mean & std of the raw S block over members."""
    s1 = s2 = None
    for arr in paths_loader:
        S = arr[:, :S_LEN].astype(np.float64)
        s1 = S if s1 is None else s1 + S
        s2 = S*S if s2 is None else s2 + S*S
    m = len(MEMBERS); mean = s1/m; std = np.sqrt(np.maximum(s2/m - mean*mean, 0))
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    C.init_run()
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    sp = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz")); te = sp["test"]
    pts = np.load(os.path.join(C.OUTPUTS_DIR, "mesh_points.npy"))

    # grouped ensemble (saved preds)
    gmean, gstd = ensemble_S(np.load(os.path.join(C.MODELS_DIR, f"{m}_pred_test.npy")) for m in MEMBERS)
    ref_g = np.asarray(Y[te], np.float32)[:, :S_LEN]

    # patient ensemble (recompute)
    PS = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    Xp = np.loadtxt(os.path.join(PS, "ps_sim_inputs_coupled.txt"), delimiter=",").astype("float32")
    Yp = pd.read_csv(os.path.join(PS, "ps_sim_outputs_coupled.txt"), header=None, sep=",",
                     dtype=np.float32).to_numpy()
    good = ~np.isnan(Yp).any(1)

    def patient_preds():
        for m in MEMBERS:
            yield predict_raw(load_run_bundle(C.MODELS_DIR, m), Xp[good])
    pmean, pstd = ensemble_S(patient_preds())
    ref_p = Yp[good][:, :S_LEN]

    # ---- Fig: peak first-principal PK2 stress Bland--Altman ----
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=False)
    for ax, mean, ref, color, name in [(axes[0], gmean, ref_g, BLUE, "Synthetic"),
                                       (axes[1], pmean, ref_p, VERM, "Patient")]:
        pws_p = first_principal(mean, n_nodes=NN, scale=TO_KPA).max(1)
        pws_r = first_principal(ref, n_nodes=NN, scale=TO_KPA).max(1)
        avg = (pws_p + pws_r)/2; diff = pws_p - pws_r
        ba = bland_altman(pws_p, pws_r)
        bias, lo, hi = ba["bias"], ba["lower_loa"], ba["upper_loa"]
        ax.scatter(avg, diff, s=6, alpha=0.3, color=color, edgecolors="none")
        ax.axhline(bias, color="black", lw=1.2); ax.axhline(lo, color="black", lw=0.8, ls="--")
        ax.axhline(hi, color="black", lw=0.8, ls="--")
        ax.set_title(f"{name}: bias {bias:+.1f} kPa"); ax.set_xlabel("Mean peak S1 (kPa)")
    axes[0].set_ylabel("Predicted − reference peak S1 (kPa)")
    fig.savefig(os.path.join(OUT, "fig_pws_bland_altman.pdf")); plt.close(fig)

    # ---- Fig: ensemble UQ coverage curve ----
    zz = np.linspace(0.25, 3.0, 30)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for mean, std, ref, color, name in [(gmean, gstd, ref_g, BLUE, "Synthetic"),
                                        (pmean, pstd, ref_p, VERM, "Patient")]:
        err = np.abs(ref - mean); s = std + 1e-9
        cov = [np.mean(err <= z*s) for z in zz]
        ax.plot(zz, cov, color=color, lw=2, label=name)
    ax.plot(zz, 2*(0.5*(1+np.vectorize(lambda z: __import__("math").erf(z/np.sqrt(2)))(zz))-0.5),
            color="gray", lw=1, ls=":", label="ideal Gaussian")
    ax.set_xlabel("z  (interval = mean ± z·σ)"); ax.set_ylabel("Node coverage")
    ax.set_ylim(0, 1); ax.legend(loc="lower right")
    fig.savefig(os.path.join(OUT, "fig_uq_coverage.pdf")); plt.close(fig)

    # ---- Fig: spatial error field, representative patient case ----
    fp_p = first_principal(pmean, n_nodes=NN, scale=TO_KPA)
    fp_pr = first_principal(ref_p, n_nodes=NN, scale=TO_KPA)
    case = np.argsort(np.abs(fp_p - fp_pr).mean(1))[len(fp_p)//2]   # median-error case
    fig = plt.figure(figsize=(7.2, 3.3))
    for k, (vals, title, cmap) in enumerate([
            (fp_pr[case], "Reference first principal stress (kPa)", "viridis"),
            (np.abs(fp_p[case]-fp_pr[case]), "Absolute error (kPa)", "magma")]):
        ax = fig.add_subplot(1, 2, k+1, projection="3d")
        sctr = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap=cmap, s=6)
        ax.set_title(title, fontsize=9); ax.set_axis_off()
        fig.colorbar(sctr, ax=ax, shrink=0.6, pad=0.02)
    fig.savefig(os.path.join(OUT, "fig_spatial_error.pdf")); plt.close(fig)

    print("saved: fig_pws_bland_altman.pdf, fig_uq_coverage.pdf, fig_spatial_error.pdf ->", OUT)


if __name__ == "__main__":
    main()
