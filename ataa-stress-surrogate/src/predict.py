"""Evaluate a compatible, user-supplied checkpoint on the patient-free example.

Trained weights are intentionally excluded from the public package. Produce a
checkpoint with the released training pipeline or supply a compatible private one.

Usage (from the component root):
    python src/predict.py --checkpoint /path/to/cardiac_grouped_free.pth
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S_LEN = 1354 * 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a compatible, user-supplied ATAA surrogate checkpoint on "
            "the bundled patient-free synthetic example."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="path to a compatible SurrogateMLP state-dict checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    import numpy as np
    import torch

    from ataa_surrogate.metrics import field_metrics
    from ataa_surrogate.model import SurrogateMLP
    from ataa_surrogate.scaling import (
        load_training_scalers,
        restore_outputs,
        standardise_inputs,
    )
    from ataa_surrogate.tensors import DYNE_CM2_TO_KPA, first_principal

    example = np.load(ROOT / "data" / "example_synthetic.npz", allow_pickle=False)
    inputs = example["inputs"].astype("float32")
    stress_reference = example["S_ref"]
    scalers = load_training_scalers(ROOT / "models", "cardiac_grouped_free")
    standardised = standardise_inputs(inputs, scalers)

    model = SurrogateMLP()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        prediction = restore_outputs(
            model(torch.tensor(standardised, dtype=torch.float32)).numpy(), scalers
        )

    principal_prediction = first_principal(
        prediction[:, :S_LEN], scale=DYNE_CM2_TO_KPA
    )
    principal_reference = first_principal(
        stress_reference, scale=DYNE_CM2_TO_KPA
    )
    metrics = field_metrics(principal_prediction, principal_reference)

    print(
        f"Evaluation on {len(inputs)} held-out SYNTHETIC loading-state rows "
        "(first principal stress, kPa):"
    )
    print(
        f"   R2 = {metrics['R2']:.3f}   relative L2 = {metrics['relL2']:.3f}   "
        f"MAE = {metrics['MAE']:.2f} kPa   Pearson = {metrics['pearson']:.3f}"
    )
    print("   NOTE: this example is patient-free and does not establish clinical validity;")
    print("         see the paper for the imaging-derived domain-transfer evaluation.")


if __name__ == "__main__":
    main()
