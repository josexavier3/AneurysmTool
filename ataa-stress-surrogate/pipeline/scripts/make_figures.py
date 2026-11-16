"""Generate publication figures from the saved results (Okabe-Ito CVD-safe palette)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT = C.FIGURES_DIR
# Okabe-Ito palette (colorblind-safe)
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
SYNTH, PATIENT = BLUE, VERM

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.autolayout": True,
})


def fig_cdf():
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for label, color in [("grouped", SYNTH), ("patient", PATIENT)]:
        d = pd.read_csv(os.path.join(C.OUTPUTS_DIR, f"cdf_relL2_{label}.csv"))
        name = "Synthetic (held-out)" if label == "grouped" else "Patient (out-of-distribution)"
        ax.plot(d["relL2"], d["cdf"], color=color, lw=2, label=name)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1)
    ax.set_xlabel("Relative $L_2$ error (first principal stress)")
    ax.set_ylabel("Cumulative fraction of cases")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(OUT, "fig_generalisation_cdf.pdf")); plt.close(fig)


def fig_error_dist():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    g = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_grouped.csv"))
    p = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_patient.csv"))
    for ax, col, ylab in [(axes[0], "relL2", "Relative $L_2$ error"),
                          (axes[1], "mae", "MAE (kPa)")]:
        bp = ax.boxplot([g[col], p[col]], widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, c in zip(bp["boxes"], [SYNTH, PATIENT]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Synthetic", "Patient"])
        ax.set_ylabel(ylab)
    fig.savefig(os.path.join(OUT, "fig_error_distribution.pdf")); plt.close(fig)


def fig_baselines():
    df = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "baselines_comparison.csv"))
    order = df.sort_values("p_MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    colors = [GREEN if m == "DNN(ours)" else "#9e9e9e" for m in order["model"]]
    ax.barh(order["model"], order["p_MAE"], color=colors, height=0.7)
    ax.set_xlabel("Patient MAE (kPa) — first principal stress")
    ax.invert_yaxis()
    for y, v in enumerate(order["p_MAE"]):
        ax.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_title("Deep net vs classical ROM (patient set)", fontsize=10)
    fig.savefig(os.path.join(OUT, "fig_baselines.pdf")); plt.close(fig)


def fig_pressure():
    X = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype("float32")
    te = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))["test"]
    g = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_grouped.csv"))
    PS = os.path.join(C.DATA_ROOT, "7.PSdatabase", "3.Inputs_outputs")
    Xp = np.loadtxt(os.path.join(PS, "ps_sim_inputs_coupled.txt"), delimiter=",").astype("float32")
    Yp = pd.read_csv(os.path.join(PS, "ps_sim_outputs_coupled.txt"), header=None, sep=",",
                     dtype=np.float32).to_numpy()
    good = ~np.isnan(Yp).any(1)
    p = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "principal_patient.csv"))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for press, mae, color, name in [(X[te, 9]/1e3, g["mae"], SYNTH, "Synthetic"),
                                     (Xp[good, 9]/1e3, p["mae"], PATIENT, "Patient")]:
        q = np.quantile(press, np.linspace(0, 1, 7))
        cx, cy = [], []
        for i in range(6):
            m = (press >= q[i]) & (press <= q[i+1])
            if m.sum() > 5:
                cx.append(press[m].mean()); cy.append(np.median(mae[m]))
        ax.plot(cx, cy, "o-", color=color, lw=2, ms=5, label=name)
    ax.set_xlabel("Instantaneous pressure (kPa-equivalent, ×10$^3$ dyne/cm$^2$)")
    ax.set_ylabel("Median MAE (kPa)")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_pressure.pdf")); plt.close(fig)


def main() -> None:
    C.init_run()
    os.makedirs(OUT, exist_ok=True)
    fig_cdf(); print("fig_generalisation_cdf.pdf")
    fig_error_dist(); print("fig_error_distribution.pdf")
    fig_baselines(); print("fig_baselines.pdf")
    fig_pressure(); print("fig_pressure.pdf")
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
