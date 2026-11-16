"""Component-wise mean-field baseline under the original (v0) agreement thresholds.

The original study scored agreement per sample over a whole 6-component tensor block,
separately for S and C, using Pearson R and a range-normalised MAE:

    moderate:  R > 0.8  and NMAE < 0.2
    high:      R > 0.95 and NMAE < 0.05

This script asks what those criteria reward. It scores a baseline that ignores the inputs
entirely and always predicts the TRAINING-MEAN field, plus (for C) the constant identity
tensor. If those already clear the thresholds, the corresponding agreement percentage
measures the structure of the tensor rather than the skill of any model.

Usage (needs ATAA_DATA_ROOT / ATAA_WORK_ROOT, and the converted array + splits):
    python scripts/meanfield_componentwise.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from result_provenance import (  # noqa: E402
    artifact_record,
    finite_summary,
    result_document,
    sha256_array,
    write_csv,
    write_json,
)

S_LEN = C.N_NODES * C.N_COMP        # 8124: S occupies [0:S_LEN), C occupies [S_LEN:2*S_LEN)
CHUNK = 512

# Original thresholds, kept here so the comparison is explicit and auditable.
MODERATE = dict(r=0.8, nmae=0.2)
HIGH = dict(r=0.95, nmae=0.05)


def mean_field(Y, idx):
    """Chunked column mean over the given rows (the array is far larger than RAM)."""
    acc = np.zeros(Y.shape[1], dtype=np.float64)
    idx = np.sort(idx)
    for i in range(0, len(idx), CHUNK):
        acc += np.asarray(Y[idx[i:i + CHUNK]], dtype=np.float64).sum(axis=0)
    return (acc / len(idx)).astype(np.float32)


def score(Y, test_idx, pred_vec, block, label):
    """Per-sample Pearson R and range-NMAE of one fixed prediction against every test sample."""
    sl = slice(0, S_LEN) if block == "S" else slice(S_LEN, 2 * S_LEN)
    p = pred_vec[sl].astype(np.float64)
    pc = p - p.mean()
    pss = np.sqrt((pc ** 2).sum())

    R, NM = [], []
    idx = np.sort(test_idx)
    for i in range(0, len(idx), CHUNK):
        ref = np.asarray(Y[idx[i:i + CHUNK], sl], dtype=np.float64)
        rc = ref - ref.mean(axis=1, keepdims=True)
        denom = np.sqrt((rc ** 2).sum(axis=1)) * pss
        correlation = np.full(len(ref), np.nan, dtype=np.float64)
        np.divide(rc @ pc, denom, out=correlation, where=denom > 0)
        R.append(correlation)
        rng = ref.max(axis=1) - ref.min(axis=1)
        NM.append(np.abs(ref - p).mean(axis=1) / np.maximum(rng, 1e-300))

    R, NM = np.concatenate(R), np.concatenate(NM)
    mod = float(np.mean((R > MODERATE["r"]) & (NM < MODERATE["nmae"])) * 100)
    high = float(np.mean((R > HIGH["r"]) & (NM < HIGH["nmae"])) * 100)
    print(f"  {label:<38} moderate {mod:6.2f}%   high {high:6.2f}%"
          f"   (median R {np.nanmedian(R):.4f}, NMAE {np.median(NM):.4f})")
    return ({"label": label, "block": block, "n": int(len(R)),
             "moderate_fraction": mod / 100.0, "high_fraction": high / 100.0,
             "pearson": finite_summary(R),
             "nmae_range": finite_summary(NM)}, R, NM)


def main():
    C.init_run()
    Y = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    sp = np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz"))
    tr, te = sp["train"], sp["test"]
    print(f"outputs {Y.shape}; train {len(tr)}, test {len(te)}")

    print("computing training mean field ...")
    mf = mean_field(Y, tr)

    print("\nMean-field baseline under the original thresholds"
          f" (moderate R>{MODERATE['r']} & NMAE<{MODERATE['nmae']};"
          f" high R>{HIGH['r']} & NMAE<{HIGH['nmae']}):")
    stress, stress_r, stress_nmae = score(
        Y, te, mf, "S", "S component-wise, mean-field"
    )
    deformation, deformation_r, deformation_nmae = score(
        Y, te, mf, "C", "C component-wise, mean-field"
    )

    # Constant identity for C. Component order is [C11, C22, C33, C12, C23, C13] per node,
    # so the identity is ones on the first three and zeros on the shear terms.
    ident = np.zeros(2 * S_LEN, dtype=np.float32)
    ident[S_LEN:].reshape(C.N_NODES, C.N_COMP)[:, :3] = 1.0
    idx = np.sort(te)
    nm = []
    for i in range(0, len(idx), CHUNK):
        ref = np.asarray(Y[idx[i:i + CHUNK], S_LEN:2 * S_LEN], dtype=np.float64)
        rng = ref.max(axis=1) - ref.min(axis=1)
        nm.append(np.abs(ref - ident[S_LEN:]).mean(axis=1) / np.maximum(rng, 1e-300))
    nm = np.concatenate(nm)
    print(f"\n  C, constant identity predictor: median range-NMAE {np.median(nm):.4f};"
          f" {100 * np.mean(nm < HIGH['nmae']):.2f}% of samples below the high NMAE bound")
    print("   (Pearson is undefined against a constant prediction, so no R-based rate)")
    identity = {
        "label": "C component-wise, constant identity",
        "block": "C",
        "pearson": "undefined because the prediction is constant",
        "nmae_range": finite_summary(nm),
        "fraction_nmae_below_high_bound": float(np.mean(nm < HIGH["nmae"])),
    }
    cases_path = write_csv(
        os.path.join(C.OUTPUTS_DIR, "meanfield_componentwise_case_metrics.csv"),
        pd.DataFrame(
            {
                "source_row": np.sort(te),
                "stress_meanfield_pearson": stress_r,
                "stress_meanfield_nmae_range": stress_nmae,
                "deformation_meanfield_pearson": deformation_r,
                "deformation_meanfield_nmae_range": deformation_nmae,
                "deformation_identity_nmae_range": nm,
            }
        ),
    )
    out = os.path.join(C.OUTPUTS_DIR, "meanfield_componentwise_summary.json")
    document = result_document(
        run_id=C.RUN_ID,
        analysis="legacy_componentwise_meanfield_diagnostic",
        definitions={
            "role": "legacy diagnostic only; not a primary performance endpoint",
            "moderate": MODERATE,
            "high": HIGH,
            "pearson": "per case over the complete six-component tensor block",
            "nmae": "per-case MAE divided by the reference component range",
            "meanfield_fit_scope": "grouped training rows only",
        },
        results={
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "training_mean_sha256": sha256_array(mf),
            "stress_meanfield": stress,
            "deformation_meanfield": deformation,
            "deformation_identity": identity,
        },
    )
    document["artefacts"] = {"per_case_metrics": artifact_record(cases_path, C.RUN_ROOT)}
    write_json(out, document)
    print(f"\nwrote {cases_path} and {out}")


if __name__ == "__main__":
    main()
