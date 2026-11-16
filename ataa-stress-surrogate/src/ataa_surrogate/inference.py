"""Load one persisted training run and apply it without refitting any scaler."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .model import SurrogateMLP, load_state_dict_file
from .scaling import ScalerBundle, load_training_scalers, restore_outputs, standardise_inputs


@dataclass(frozen=True)
class SurrogateBundle:
    run_name: str
    model: SurrogateMLP
    scalers: ScalerBundle
    training_log: dict


def load_run_bundle(
    models_dir: str | Path,
    run_name: str,
    *,
    device: str | torch.device = "cpu",
) -> SurrogateBundle:
    """Load architecture, checkpoint and scalers from the same named run."""

    root = Path(models_dir)
    with (root / f"{run_name}_train_log.json").open("r", encoding="utf-8") as stream:
        log = json.load(stream)
    if log.get("model_run_name") not in (None, run_name):
        raise ValueError("training log run name does not match the requested run")
    architecture = log["architecture"]
    model = SurrogateMLP(
        int(architecture["input_dim"]),
        int(architecture["output_dim"]),
        tuple(int(width) for width in architecture["hidden"]),
        float(architecture["dropout"]),
    )
    model.load_state_dict(load_state_dict_file(root / f"{run_name}.pth"))
    model.to(device)
    model.eval()
    scalers = load_training_scalers(root, run_name)
    if scalers.input_mean.size != int(architecture["input_dim"]):
        raise ValueError("saved input scaler is incompatible with the model architecture")
    if scalers.output_mean.size != int(architecture["output_dim"]):
        raise ValueError("saved output scaler is incompatible with the model architecture")
    return SurrogateBundle(run_name, model, scalers, log)


def predict_raw(
    bundle: SurrogateBundle,
    raw_inputs: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict original-unit fields using the run's persisted scalers."""

    scaled_inputs = standardise_inputs(raw_inputs, bundle.scalers)
    with torch.no_grad():
        scaled_prediction = bundle.model(
            torch.from_numpy(scaled_inputs).to(device)
        ).detach().cpu().numpy()
    return restore_outputs(scaled_prediction, bundle.scalers)
