"""Stage 1 -- build the leakage-free geometry-grouped split (and a random split).

The full training file `sim_inputs_coupled_extended.txt` (79 626 x 10) has no ID column,
but the first six columns are the PCA shape modes, which are identical for every loading
state of a given geometry. The (rounded) PCA(6) tuple is therefore a geometry fingerprint:
grouping on it groups all 20 loading states -- and all simulations -- of one geometry.

Produces:
  <run>/splits/grouped_split.npz   train/val/test row indices, grouped by geometry (no leakage)
  <run>/splits/random_split.npz    train/val/test row indices, sample-level (legacy contrast)
  <run>/outputs/split_manifest.json   counts, seed and overlap checks

Only the 5.6 MB inputs file is read; the 24.86 GB outputs file is NOT touched here.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from evidence_hashing import sha256_file
from result_provenance import artifact_record, write_csv, write_json
from split_validation import load_bad_rows, save_or_verify_split, validate_split_partition

PCA_DECIMALS = 2  # rounding for the geometry fingerprint; PCA cols are O(10-100)


def main():
    C.init_run()
    print(f"Loading inputs: {C.INPUTS_FULL}")
    X = np.loadtxt(C.INPUTS_FULL, delimiter=",")
    n = X.shape[0]
    print(f"  inputs shape: {X.shape}")

    # Geometry fingerprint = rounded PCA(6) tuple -> integer group id
    pca = np.round(X[:, : C.N_PCA], PCA_DECIMALS)
    unique_pca, groups = np.unique(pca, axis=0, return_inverse=True)
    groups = np.asarray(groups).reshape(-1)
    n_groups = int(groups.max()) + 1
    counts = np.bincount(groups)
    print(f"  geometries (unique PCA tuples): {n_groups}")
    print(f"  samples/geometry: min={counts.min()} max={counts.max()} "
          f"median={int(np.median(counts))}")

    rng = C.SEED

    # --- Require a passing non-finite-data diagnosis before splitting. ---
    diagnosis_path = Path(C.OUTPUTS_DIR) / "data_diagnosis.json"
    if not diagnosis_path.is_file():
        raise FileNotFoundError(f"Required diagnosis is missing: {diagnosis_path}")
    with diagnosis_path.open("r", encoding="utf-8") as stream:
        diagnosis = json.load(stream)
    checks = diagnosis["outputs"]
    required_true = (
        "shape_matches_expected",
        "dtype_is_float32",
        "bad_row_count_matches_stage0",
        "all_bad_rows_are_fully_nan",
    )
    failed = [name for name in required_true if checks.get(name) is not True]
    if failed:
        raise RuntimeError(f"Data diagnosis did not pass required checks: {failed}")

    # --- Exclude all-NaN (failed-simulation) rows found by diagnose_data.py ---
    bad_path = os.path.join(C.OUTPUTS_DIR, "bad_rows.npy")
    bad = load_bad_rows(bad_path, n)
    if diagnosis["bad_rows"]["sha256"] != sha256_file(bad_path):
        raise RuntimeError("bad_rows.npy no longer matches data_diagnosis.json")
    valid = np.setdiff1d(np.arange(n), bad)
    if np.unique(groups[valid]).size != n_groups:
        raise RuntimeError("At least one geometry has no valid states after bad-row exclusion")
    print(f"  excluding {len(bad)} all-NaN rows -> {len(valid)} valid samples")

    # Validate the PCA proxy against raw-to-matrix synthetic geometry IDs when available.
    provenance_path = Path(C.AUDIT_DIR) / "row_provenance.csv"
    fingerprint_validation: dict[str, object] = {"available": False}
    fingerprint_map_path = Path(C.OUTPUTS_DIR) / "geometry_fingerprint_map.csv"
    if provenance_path.is_file():
        provenance = pd.read_csv(provenance_path)
        if not np.array_equal(provenance["source_row"].to_numpy(), np.arange(n)):
            raise RuntimeError("row_provenance.csv does not cover source rows 0..n-1 in order")
        crosswalk = pd.DataFrame(
            {"group_id": groups, "sample_id": provenance["sample_id"].to_numpy()}
        )
        groups_per_geometry = crosswalk.groupby("sample_id")["group_id"].nunique()
        geometries_per_group = crosswalk.groupby("group_id")["sample_id"].nunique()
        one_to_one = bool(
            groups_per_geometry.eq(1).all() and geometries_per_group.eq(1).all()
        )
        if not one_to_one:
            raise RuntimeError("PCA fingerprints are not one-to-one with provenance geometry IDs")
        mapping = crosswalk.drop_duplicates().sort_values("group_id").reset_index(drop=True)
        for index in range(C.N_PCA):
            mapping[f"pca{index + 1}"] = unique_pca[
                mapping["group_id"].to_numpy(dtype=np.int64), index
            ]
        write_csv(fingerprint_map_path, mapping)
        sensitivity = {
            str(decimals): int(
                np.unique(np.round(X[:, : C.N_PCA], decimals), axis=0).shape[0]
            )
            for decimals in range(7)
        }
        sensitivity["unrounded"] = int(np.unique(X[:, : C.N_PCA], axis=0).shape[0])
        fingerprint_validation = {
            "available": True,
            "one_to_one_with_provenance_geometry_id": True,
            "n_provenance_geometry_ids": int(provenance["sample_id"].nunique()),
            "n_fingerprints": n_groups,
            "rounding_sensitivity_unique_counts": sensitivity,
            "mapping": artifact_record(fingerprint_map_path, C.RUN_ROOT),
        }

    # --- Grouped split over VALID rows: 20% of GEOMETRIES to test, then 10% to val ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=rng)
    tv_pos, te_pos = next(gss.split(valid, groups=groups[valid]))
    trainval_idx, test_idx = valid[tv_pos], valid[te_pos]
    g_tv = groups[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=rng)
    rel_train, rel_val = next(gss2.split(trainval_idx, groups=g_tv))
    train_idx = trainval_idx[rel_train]
    val_idx = trainval_idx[rel_val]

    # --- Shuffled diagnostic comparator; this is not the legacy shuffle=False protocol. ---
    r_trainval, r_test = train_test_split(valid, test_size=0.20, random_state=rng)
    r_train, r_val = train_test_split(r_trainval, test_size=0.10, random_state=rng)

    # --- Verify complete accounting and no geometry crosses the grouped split ---
    gtr, gva, gte = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    grouped_validation = validate_split_partition(
        {"train": train_idx, "val": val_idx, "test": test_idx},
        valid,
        groups=groups,
        require_group_disjoint=True,
    )
    overlap_grouped = grouped_validation["group_overlap"]
    validate_split_partition(
        {"train": r_train, "val": r_val, "test": r_test},
        valid,
        groups=groups,
        require_group_disjoint=False,
    )
    # For contrast, count geometries shared across the RANDOM split (expected > 0 = leakage)
    overlap_random = len((set(groups[r_train]) & set(groups[r_test])))

    grouped_path = os.path.join(C.SPLITS_DIR, "grouped_split.npz")
    random_path = os.path.join(C.SPLITS_DIR, "random_split.npz")
    os.makedirs(C.SPLITS_DIR, exist_ok=True)
    grouped_archive_state = save_or_verify_split(
        grouped_path, train=train_idx, val=val_idx, test=test_idx, groups=groups
    )
    random_archive_state = save_or_verify_split(
        random_path, train=r_train, val=r_val, test=r_test, groups=groups
    )

    manifest = {
        "n_samples": int(n),
        "n_bad_rows_excluded": int(len(bad)),
        "n_valid": int(len(valid)),
        "n_geometries": n_groups,
        "samples_per_geometry": {"min": int(counts.min()), "max": int(counts.max()),
                                  "median": int(np.median(counts))},
        "pca_round_decimals": PCA_DECIMALS,
        "seed": rng,
        "archive_state": {
            "grouped": grouped_archive_state,
            "random": random_archive_state,
        },
        "geometry_fingerprint_validation": fingerprint_validation,
        "grouped": {"n_train": int(train_idx.size), "n_val": int(val_idx.size),
                     "n_test": int(test_idx.size),
                     "n_geom_train": len(gtr), "n_geom_val": len(gva), "n_geom_test": len(gte),
                     "geometries_shared_across_split": overlap_grouped},
        "random": {"n_train": int(r_train.size), "n_val": int(r_val.size),
                    "n_test": int(r_test.size),
                    "geometries_shared_train_test": overlap_random},
        "artifacts": {
            "grouped_split": artifact_record(grouped_path, C.RUN_ROOT),
            "random_split": artifact_record(random_path, C.RUN_ROOT),
            "bad_rows": artifact_record(bad_path, C.RUN_ROOT),
            "data_diagnosis": artifact_record(diagnosis_path, C.RUN_ROOT),
        },
    }
    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    split_manifest_path = write_json(
        os.path.join(C.OUTPUTS_DIR, "split_manifest.json"), manifest
    )
    C.update_manifest(
        {
            "i07_splits": {
                "passed": True,
                "manifest": artifact_record(split_manifest_path, C.RUN_ROOT),
                "grouped_split": manifest["artifacts"]["grouped_split"],
                "random_split": manifest["artifacts"]["random_split"],
                "geometry_fingerprint_validation": fingerprint_validation,
            }
        }
    )

    print("\n=== manifest ===")
    print(json.dumps(manifest, indent=2))
    print("\nGrouped split leakage (geometries shared across split):", overlap_grouped,
          "(must be 0)")
    print("Random split leakage (geometries shared train<->test):", overlap_random,
          "(shuffled diagnostic comparator; not the legacy protocol)")


if __name__ == "__main__":
    main()
