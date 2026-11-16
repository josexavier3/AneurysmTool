"""Prepare run-scoped release artefacts for later author-approved promotion."""

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.scaling import load_training_scalers
from result_provenance import sha256_file


STRESS_WIDTH = C.N_NODES * 6
PRIMARY_RUN_NAME = "cardiac_grouped_free_seed42"
def main() -> None:
    C.init_run()
    release = C.RELEASE_STAGING_DIR
    shutil.copytree(
        C.PUBLIC_RELEASE_ROOT,
        release,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "models", "data", "SHA256SUMS", "work"
        ),
    )
    for subdirectory in ("models", "data", "src"):
        os.makedirs(os.path.join(release, subdirectory), exist_ok=True)
    # A public package may be refreshed in place. Remove any stale checkpoints
    # from an earlier staging policy: trained weights remain in the frozen run.
    public_models = os.path.join(release, "models")
    for filename in os.listdir(public_models):
        path = os.path.join(public_models, filename)
        if os.path.isfile(path) and filename.lower().endswith((".pth", ".pt", ".ckpt")):
            os.remove(path)

    inputs = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype("float32")
    outputs = np.load(os.path.join(C.DATA_CACHE, "sim_outputs_full.npy"), mmap_mode="r")
    with np.load(os.path.join(C.SPLITS_DIR, "grouped_split.npz")) as archive:
        test_indices = archive["test"].copy()

    scalers = load_training_scalers(C.MODELS_DIR, PRIMARY_RUN_NAME)
    np.savez(
        os.path.join(release, "models", "cardiac_grouped_free_input_scaler.npz"),
        mean=scalers.input_mean,
        scale=scalers.input_scale,
    )
    np.savez(
        os.path.join(release, "models", "cardiac_grouped_free_output_scaler.npz"),
        mean=scalers.output_mean,
        std=scalers.output_scale,
    )

    sample_count = 50
    rng = np.random.default_rng(0)
    selected = np.sort(rng.choice(test_indices, size=sample_count, replace=False))
    np.savez_compressed(
        os.path.join(release, "data", "example_synthetic.npz"),
        inputs=inputs[selected],
        # The source unit audit identifies solver stress as dyne/cm^2.
        S_ref=np.asarray(outputs[selected], dtype=np.float32)[:, :STRESS_WIDTH],
        C_ref=np.asarray(outputs[selected], dtype=np.float32)[:, STRESS_WIDTH:],
    )

    shutil.copytree(
        C.PUBLIC_SRC_DIR,
        os.path.join(release, "src"),
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    hashed_assets = (
        "models/cardiac_grouped_free_input_scaler.npz",
        "models/cardiac_grouped_free_output_scaler.npz",
        "data/example_synthetic.npz",
    )
    with open(os.path.join(release, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as stream:
        for relative in hashed_assets:
            stream.write(f"{sha256_file(os.path.join(release, relative))}  {relative}\n")

    print("staged release artefacts at:", release)
    for root, _, files in os.walk(release):
        for filename in files:
            path = os.path.join(root, filename)
            print(f"  {os.path.relpath(path, release)}  ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
