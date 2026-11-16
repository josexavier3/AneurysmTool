"""Train the cardiac-cycle surrogate with complete run-scoped provenance.

The production protocol predicts the concatenated S and C fields. Every artefact
stem includes its split and seed. A completed training command saves the cloned
best checkpoint, fitted scalers, split hashes, predictions, complete loss history
and a fixed reload fixture, then verifies the saved artefacts in a fresh process.

Examples:
  python train_cardiac_cycle.py --split grouped --seed 42
  python train_cardiac_cycle.py --split random --seed 42
  python train_cardiac_cycle.py --verify-run cardiac_grouped_free_seed42
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Pin native and PyTorch thread pools before importing NumPy, scikit-learn or PyTorch.
EVIDENCE_DEVICE = os.environ.get("ATAA_EVIDENCE_DEVICE", "cpu").strip().lower()
try:
    EVIDENCE_TORCH_THREADS = int(os.environ.get("ATAA_TORCH_THREADS", "4"))
    EVIDENCE_TORCH_INTEROP_THREADS = int(os.environ.get("ATAA_TORCH_INTEROP_THREADS", "1"))
except ValueError as error:
    raise RuntimeError("ATAA_TORCH_THREADS and ATAA_TORCH_INTEROP_THREADS must be integers") from error
if EVIDENCE_TORCH_THREADS <= 0 or EVIDENCE_TORCH_INTEROP_THREADS <= 0:
    raise RuntimeError("ATAA thread counts must be positive")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = str(EVIDENCE_TORCH_THREADS)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))
import config as C  # noqa: E402
from evidence_gates import load_data_audit, load_loss_ab_protocol, load_loss_policy  # noqa: E402
sys.path.insert(0, str(C.PUBLIC_SRC_DIR))
from ataa_surrogate.losses import (  # noqa: E402
    GHAZI_LOSS_DECISION,
    LEGACY_STANDARDISED_SIGMA_DECISION,
    LOSS_POLICY_IDS,
    group_balanced_ghazi_mse,
    group_balanced_ghazi_weights,
    legacy_standardised_sigma_weights,
    weighted_mse,
    weighted_field_mse,
)
from ataa_surrogate.model import (  # noqa: E402
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN,
    SurrogateMLP,
    load_state_dict_file,
)
from ataa_surrogate.scaling import (  # noqa: E402
    load_training_scalers,
    restore_outputs,
    standardise_inputs,
)
from training_provenance import (  # noqa: E402
    ARRAY_HASH_ALGORITHM,
    ENSEMBLE_SEEDS,
    FILE_HASH_ALGORITHM,
    clone_state_dict,
    model_run_name,
    save_input_scaler,
    save_output_scaler,
    set_determinism,
    sha256_array,
    sha256_file,
)


DEFAULT_PATIENCE = 15
DEFAULT_SCHEDULER_PATIENCE = 5
OUTPUT_SCALER_EPSILON = 1e-8
RELOAD_FIXTURE_SIZE = 4
RELOAD_RTOL = 1e-5
RELOAD_ATOL = 1e-4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_hidden(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden widths must be comma-separated integers") from error
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("hidden widths must be positive")
    return widths


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    return dict(load_state_dict_file(path))


def _artefact_paths(
    run_name: str, evaluation_partition: str = "test"
) -> dict[str, Path]:
    root = Path(C.MODELS_DIR)
    return {
        "checkpoint": root / f"{run_name}.pth",
        "input_scaler": root / f"{run_name}_input_scaler.npz",
        "output_scaler": root / f"{run_name}_output_scaler.npz",
        "predictions": root / f"{run_name}_pred_{evaluation_partition}.npy",
        "reload_fixture": root / f"{run_name}_reload_fixture.npz",
        "reload_check": root / f"{run_name}_reload_check.json",
        "training_log": root / f"{run_name}_train_log.json",
    }


def _validate_model_run_name(run_name: str) -> None:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?", run_name):
        raise ValueError(
            "model run name must be 1-128 lowercase letters/digits with optional '_' or '-', "
            "and must start and end with a letter or digit"
        )


def _refuse_existing_training_run(run_name: str, paths: dict[str, Path]) -> None:
    """Prevent an active run from silently replacing model evidence."""

    existing_paths = [path for path in paths.values() if path.exists()]
    with Path(C.MANIFEST_PATH).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest_record_exists = run_name in manifest.get("training_runs", {})
    if existing_paths or manifest_record_exists:
        details = [str(path.relative_to(C.RUN_ROOT)) for path in existing_paths]
        if manifest_record_exists:
            details.append(f"run_manifest.json:training_runs.{run_name}")
        raise FileExistsError(
            f"Refusing to overwrite existing training evidence for {run_name!r}: "
            f"{', '.join(details)}. Use a new ATAA_RUN_ID or a new --tag."
        )


def verify_saved_run(
    run_name: str,
    *,
    rtol: float = RELOAD_RTOL,
    atol: float = RELOAD_ATOL,
) -> dict:
    """Reload a saved checkpoint/scalers and verify a fixed prediction fixture."""

    _validate_model_run_name(run_name)
    paths = _artefact_paths(run_name)
    with paths["training_log"].open("r", encoding="utf-8") as stream:
        training_log = json.load(stream)
    paths = _artefact_paths(
        run_name, training_log.get("arguments", {}).get("evaluation_partition", "test")
    )
    declared_device = training_log.get("determinism", {}).get("evidence_device_policy")
    if declared_device not in {None, "cpu"}:
        raise RuntimeError(
            f"Saved run declares unsupported evidence device {declared_device!r}; "
            "the audited verifier is CPU-only."
        )
    architecture = training_log["architecture"]
    model = SurrogateMLP(
        int(architecture["input_dim"]),
        int(architecture["output_dim"]),
        tuple(int(width) for width in architecture["hidden"]),
        float(architecture["dropout"]),
    )
    model.load_state_dict(_load_state_dict(paths["checkpoint"]))
    model.eval()

    scalers = load_training_scalers(C.MODELS_DIR, run_name)
    with np.load(paths["reload_fixture"]) as fixture:
        raw_inputs = fixture["raw_inputs"].astype(np.float32, copy=False)
        expected = fixture["expected_predictions"].astype(np.float32, copy=False)

    scaled_inputs = standardise_inputs(raw_inputs, scalers)
    with torch.no_grad():
        scaled_prediction = model(torch.from_numpy(scaled_inputs)).numpy()
    actual = restore_outputs(scaled_prediction, scalers)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), atol)
    predictions_match = bool(
        np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    )
    actual_hashes = {
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "input_scaler_sha256": sha256_file(paths["input_scaler"]),
        "output_scaler_sha256": sha256_file(paths["output_scaler"]),
        "reload_fixture_sha256": sha256_file(paths["reload_fixture"]),
    }
    expected_hashes = training_log.get("hashes", {})
    hashes_match = all(
        expected_hashes.get(name) == digest for name, digest in actual_hashes.items()
    )
    passed = predictions_match and hashes_match
    result = {
        "schema_version": 1,
        "run_id": C.RUN_ID,
        "model_run_name": run_name,
        "verified_utc": _utc_now(),
        "passed": passed,
        "predictions_match": predictions_match,
        "hashes_match_training_log": hashes_match,
        "declared_training_device": declared_device or "legacy-unspecified-cpu-verification",
        "verification_device": "cpu",
        "rtol": rtol,
        "atol": atol,
        "fixture_samples": int(raw_inputs.shape[0]),
        "max_absolute_difference": float(absolute.max(initial=0.0)),
        "max_relative_difference": float(relative.max(initial=0.0)),
        "hashes": actual_hashes,
    }
    if paths["reload_check"].exists():
        verified_token = re.sub(r"[^0-9]", "", result["verified_utc"])
        history_dir = Path(C.MODELS_DIR) / "verification_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{run_name}_reload_check_{verified_token}.json"
        if history_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite reload-verification history: {history_path}"
            )
        _atomic_json(history_path, result)
    else:
        _atomic_json(paths["reload_check"], result)
    if not passed:
        raise RuntimeError(
            f"Fresh-process reload check failed for {run_name}: "
            f"predictions_match={predictions_match}, hashes_match={hashes_match}, "
            f"max_abs={result['max_absolute_difference']:.6g}, "
            f"max_rel={result['max_relative_difference']:.6g}."
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("grouped", "random"), default="grouped")
    parser.add_argument("--variant", choices=("free", "cholesky"), default="free")
    parser.add_argument("--epochs", type=int, default=152)
    parser.add_argument("--batch", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, choices=ENSEMBLE_SEEDS, default=C.SEED)
    parser.add_argument("--hidden", type=_parse_hidden, default=DEFAULT_HIDDEN)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_SCHEDULER_PATIENCE)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require deterministic PyTorch algorithms (default: true)",
    )
    parser.add_argument("--tag", default="", help="optional lowercase artefact suffix")
    parser.add_argument(
        "--loss-policy",
        choices=LOSS_POLICY_IDS,
        default=GHAZI_LOSS_DECISION["id"],
    )
    parser.add_argument(
        "--evaluation-partition",
        choices=("test", "validation"),
        default="test",
        help="validation is restricted to prospective I07A loss selection",
    )
    parser.add_argument("--verify-run", help="reload and verify an existing run stem, then exit")
    parser.add_argument("--reload-rtol", type=float, default=RELOAD_RTOL)
    parser.add_argument("--reload-atol", type=float, default=RELOAD_ATOL)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    C.init_run()
    if args.reload_rtol < 0 or args.reload_atol < 0:
        raise SystemExit("reload tolerances must be non-negative.")
    if args.verify_run:
        result = verify_saved_run(args.verify_run, rtol=args.reload_rtol, atol=args.reload_atol)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.variant == "cholesky":
        raise SystemExit("The cholesky variant is not implemented and must not be selected.")
    if args.epochs <= 0 or args.batch <= 0 or args.lr <= 0:
        raise SystemExit("epochs, batch and learning rate must be positive.")
    if not 0 <= args.dropout < 1:
        raise SystemExit("dropout must be in [0, 1).")
    if args.tag and not re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]{0,30}[a-z0-9])?", args.tag):
        raise SystemExit("tag must be 1-32 lowercase letters/digits with optional '_' or '-'.")
    selection_run = args.evaluation_partition == "validation"
    if selection_run and (
        args.split != "grouped" or args.seed != 42 or args.tag != args.loss_policy
    ):
        raise SystemExit(
            "I07A validation-only runs require --split grouped --seed 42 and a tag exactly "
            "matching --loss-policy."
        )
    evidence_gates = {"audited_evidence_mode": C.AUDITED_EVIDENCE_MODE}
    if C.AUDITED_EVIDENCE_MODE:
        data_audit_path = Path(C.AUDIT_DIR) / "data_audit.json"
        data_audit = load_data_audit(
            data_audit_path,
            expected_run_id=C.RUN_ID,
            import_record_path=Path(C.AUDIT_DIR) / "imported_evidence.json",
        )
        manifest_gates = {"data_audit_gate": data_audit}
        evidence_gates["data_audit_sha256"] = sha256_file(data_audit_path)
        if selection_run:
            protocol_path = Path(C.AUDIT_DIR) / "loss_ab_protocol.json"
            load_loss_ab_protocol(protocol_path, expected_loss_id=args.loss_policy)
            evidence_gates["loss_ab_protocol_sha256"] = sha256_file(protocol_path)
        else:
            loss_policy_path = Path(C.AUDIT_DIR) / "loss_policy.json"
            loss_policy = load_loss_policy(
                loss_policy_path, expected_loss_id=args.loss_policy
            )
            manifest_gates["loss_policy"] = loss_policy
            evidence_gates["loss_policy_sha256"] = sha256_file(loss_policy_path)
        C.update_manifest(manifest_gates)

    run_name = model_run_name(args.split, args.variant, args.seed, args.tag)
    _validate_model_run_name(run_name)
    paths = _artefact_paths(run_name, args.evaluation_partition)
    _refuse_existing_training_run(run_name, paths)
    start_time = time.perf_counter()
    started_utc = _utc_now()
    determinism = set_determinism(
        args.seed,
        args.deterministic,
        evidence_device=EVIDENCE_DEVICE,
        torch_threads=EVIDENCE_TORCH_THREADS,
        torch_interop_threads=EVIDENCE_TORCH_INTEROP_THREADS,
    )
    device = torch.device("cpu")
    determinism.update(
        {
            "device": str(device),
            "cuda_available_but_not_selected": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    print(f"run: {run_name}\ndevice: {device}")

    inputs = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype(np.float32)
    target_path = Path(C.DATA_CACHE) / "sim_outputs_full.npy"
    if not target_path.exists():
        raise SystemExit(f"Missing {target_path}. Run convert_outputs.py first.")
    targets = np.load(target_path, mmap_mode="r")
    if inputs.shape[0] != targets.shape[0]:
        raise RuntimeError(f"Input/target row mismatch: {inputs.shape[0]} != {targets.shape[0]}")
    expected_width = C.N_NODES * 12
    if targets.shape[1] != expected_width:
        raise RuntimeError(f"Target width {targets.shape[1]} != expected {expected_width}")

    split_path = Path(C.SPLITS_DIR) / f"{args.split}_split.npz"
    with np.load(split_path) as split_archive:
        train_indices = split_archive["train"].astype(np.int64, copy=True)
        validation_indices = split_archive["val"].astype(np.int64, copy=True)
        evaluation_indices = (
            validation_indices
            if selection_run
            else split_archive["test"].astype(np.int64, copy=True)
        )
    split_hashes = {
        "array_hash_algorithm": ARRAY_HASH_ALGORITHM,
        "file_hash_algorithm": FILE_HASH_ALGORITHM,
        "archive_sha256": sha256_file(split_path),
        "train_indices_sha256": sha256_array(train_indices),
        "validation_indices_sha256": sha256_array(validation_indices),
        "evaluation_partition": args.evaluation_partition,
        "evaluation_indices_sha256": sha256_array(evaluation_indices),
    }
    if not selection_run:
        split_hashes["test_indices_sha256"] = split_hashes["evaluation_indices_sha256"]
    split_counts = {
        "train": int(train_indices.size),
        "validation": int(validation_indices.size),
        "evaluation": int(evaluation_indices.size),
        "test": "sealed" if selection_run else int(evaluation_indices.size),
    }

    input_fit = inputs[train_indices].astype(np.float64)
    input_mean = input_fit.mean(axis=0)
    input_scale = input_fit.std(axis=0, ddof=0)
    input_scale[input_scale == 0] = 1.0

    def scale_inputs(indices: np.ndarray) -> np.ndarray:
        return ((inputs[indices] - input_mean) / input_scale).astype(np.float32)

    train_inputs = scale_inputs(train_indices)
    validation_inputs = scale_inputs(validation_indices)
    evaluation_inputs = scale_inputs(evaluation_indices)
    save_input_scaler(paths["input_scaler"], input_mean, input_scale)

    train_targets = np.asarray(targets[train_indices], dtype=np.float32)
    output_mean = train_targets.mean(axis=0)
    raw_output_sd = train_targets.std(axis=0, ddof=0)
    output_std = raw_output_sd + OUTPUT_SCALER_EPSILON
    del train_targets
    save_output_scaler(paths["output_scaler"], output_mean, output_std)
    field_width = C.N_NODES * C.N_COMP
    loss_group_widths = (field_width, field_width)
    if args.loss_policy == GHAZI_LOSS_DECISION["id"]:
        loss_definition = GHAZI_LOSS_DECISION
        output_weights_array = group_balanced_ghazi_weights(
            raw_output_sd, output_std, loss_group_widths
        )
    else:
        loss_definition = LEGACY_STANDARDISED_SIGMA_DECISION
        output_weights_array = legacy_standardised_sigma_weights(output_std)
    output_weights = torch.tensor(
        output_weights_array,
        dtype=torch.float32,
        device=device,
    )

    def objective_components(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stress_loss, deformation_loss = weighted_field_mse(
            prediction, target, output_weights, loss_group_widths
        )
        if args.loss_policy == GHAZI_LOSS_DECISION["id"]:
            total = group_balanced_ghazi_mse(
                prediction, target, output_weights, loss_group_widths
            )
        else:
            total = weighted_mse(prediction, target, output_weights)
        return total, stress_loss, deformation_loss

    def scale_targets(indices: np.ndarray) -> np.ndarray:
        values = np.asarray(targets[indices], dtype=np.float32)
        return ((values - output_mean) / output_std).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(train_inputs), torch.from_numpy(scale_targets(train_indices))
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_inputs), torch.from_numpy(scale_targets(validation_indices))
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch, shuffle=False, num_workers=0
    )

    model = SurrogateMLP(C.N_INPUTS, expected_width, args.hidden, args.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=args.scheduler_patience
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters: {parameter_count:,}")

    best_loss = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        training_loss_sum = 0.0
        training_stress_loss_sum = 0.0
        training_deformation_loss_sum = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimiser.zero_grad()
            loss, stress_loss, deformation_loss = objective_components(
                model(batch_inputs), batch_targets
            )
            loss.backward()
            optimiser.step()
            training_loss_sum += loss.item() * len(batch_inputs)
            training_stress_loss_sum += stress_loss.item() * len(batch_inputs)
            training_deformation_loss_sum += deformation_loss.item() * len(batch_inputs)

        model.eval()
        validation_loss_sum = 0.0
        validation_stress_loss_sum = 0.0
        validation_deformation_loss_sum = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in validation_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                loss, stress_loss, deformation_loss = objective_components(
                    model(batch_inputs), batch_targets
                )
                validation_loss_sum += loss.item() * len(batch_inputs)
                validation_stress_loss_sum += stress_loss.item() * len(batch_inputs)
                validation_deformation_loss_sum += deformation_loss.item() * len(batch_inputs)
        training_loss = training_loss_sum / len(train_dataset)
        validation_loss = validation_loss_sum / len(validation_dataset)
        scheduler.step(validation_loss)
        epoch_record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_stress_contribution": training_stress_loss_sum / len(train_dataset),
            "training_deformation_contribution": (
                training_deformation_loss_sum / len(train_dataset)
            ),
            "validation_stress_contribution": (
                validation_stress_loss_sum / len(validation_dataset)
            ),
            "validation_deformation_contribution": (
                validation_deformation_loss_sum / len(validation_dataset)
            ),
            "learning_rate": float(optimiser.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:3d} train {training_loss:.6g} val {validation_loss:.6g} "
            f"({epoch_record['seconds']:.1f}s)",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = clone_state_dict(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("early stopping", flush=True)
                break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Training completed without a valid best checkpoint.")
    model.load_state_dict(best_state)
    torch.save(clone_state_dict(model.state_dict()), paths["checkpoint"])

    model.eval()
    with torch.no_grad():
        scaled_prediction = model(
            torch.from_numpy(evaluation_inputs).to(device)
        ).cpu().numpy()
    prediction = (scaled_prediction * output_std + output_mean).astype(np.float32)
    np.save(paths["predictions"], prediction)

    fixture_count = min(RELOAD_FIXTURE_SIZE, len(evaluation_indices))
    source_index_key = f"source_{args.evaluation_partition}_indices"
    np.savez_compressed(
        paths["reload_fixture"],
        raw_inputs=inputs[evaluation_indices[:fixture_count]].astype(np.float32),
        expected_predictions=prediction[:fixture_count],
        **{source_index_key: evaluation_indices[:fixture_count]},
    )

    training_log = {
        "schema_version": 3,
        "run_id": C.RUN_ID,
        "model_run_name": run_name,
        "started_utc": started_utc,
        "training_and_prediction_finished_utc": _utc_now(),
        "arguments": {
            "split": args.split,
            "variant": args.variant,
            "epochs_requested": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "seed": args.seed,
            "hidden": list(args.hidden),
            "dropout": args.dropout,
            "early_stopping_patience": args.patience,
            "scheduler_patience": args.scheduler_patience,
            "tag": args.tag,
            "loss_policy": args.loss_policy,
            "evaluation_partition": args.evaluation_partition,
            "grouped_test_set_sealed": selection_run,
            "deterministic_requested": args.deterministic,
            "evidence_device": EVIDENCE_DEVICE,
            "torch_threads": EVIDENCE_TORCH_THREADS,
            "torch_interop_threads": EVIDENCE_TORCH_INTEROP_THREADS,
            "reload_rtol": args.reload_rtol,
            "reload_atol": args.reload_atol,
        },
        "architecture": {
            "class": "SurrogateMLP",
            "input_dim": C.N_INPUTS,
            "output_dim": expected_width,
            "hidden": list(args.hidden),
            "dropout": args.dropout,
            "parameter_count": parameter_count,
        },
        "optimisation": {
            "loss": args.loss_policy,
            "loss_definition": loss_definition,
            "loss_group_widths": list(loss_group_widths),
            "loss_weights_sha256": sha256_array(output_weights_array),
            "loss_weight_summary": {
                "S": {
                    "minimum": float(output_weights_array[:field_width].min()),
                    "maximum": float(output_weights_array[:field_width].max()),
                    "mean": float(output_weights_array[:field_width].mean()),
                },
                "C": {
                    "minimum": float(output_weights_array[field_width:].min()),
                    "maximum": float(output_weights_array[field_width:].max()),
                    "mean": float(output_weights_array[field_width:].mean()),
                },
            },
            "optimiser": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_patience": args.scheduler_patience,
            "early_stopping_patience": args.patience,
            "output_scaler_epsilon": OUTPUT_SCALER_EPSILON,
        },
        "determinism": determinism,
        "split_counts": split_counts,
        "split_hashes": split_hashes,
        "hash_algorithms": {
            "array": ARRAY_HASH_ALGORITHM,
            "file": FILE_HASH_ALGORITHM,
        },
        "evidence_gates": evidence_gates,
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "stopping_epoch": history[-1]["epoch"],
        "epochs_completed": len(history),
        "validation_history": history,
        "training_and_prediction_wall_time_seconds": time.perf_counter() - start_time,
        "code_hashes": {
            "train_cardiac_cycle.py": sha256_file(Path(__file__).resolve()),
            "evidence_hashing.py": sha256_file(ANALYSIS_DIR / "evidence_hashing.py"),
            "evidence_gates.py": sha256_file(ANALYSIS_DIR / "evidence_gates.py"),
            "training_provenance.py": sha256_file(ANALYSIS_DIR / "training_provenance.py"),
            "config.py": sha256_file(ANALYSIS_DIR / "config.py"),
            "ataa_surrogate/constants.py": sha256_file(
                C.PUBLIC_SRC_DIR / "ataa_surrogate" / "constants.py"
            ),
            "ataa_surrogate/model.py": sha256_file(C.PUBLIC_SRC_DIR / "ataa_surrogate" / "model.py"),
            "ataa_surrogate/losses.py": sha256_file(C.PUBLIC_SRC_DIR / "ataa_surrogate" / "losses.py"),
            "ataa_surrogate/scaling.py": sha256_file(C.PUBLIC_SRC_DIR / "ataa_surrogate" / "scaling.py"),
        },
        "artefacts": {name: str(path.relative_to(C.RUN_ROOT)) for name, path in paths.items()},
        "hashes": {
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "input_scaler_sha256": sha256_file(paths["input_scaler"]),
            "output_scaler_sha256": sha256_file(paths["output_scaler"]),
            "predictions_sha256": sha256_file(paths["predictions"]),
            "reload_fixture_sha256": sha256_file(paths["reload_fixture"]),
        },
    }
    _atomic_json(paths["training_log"], training_log)

    verification_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verify-run",
        run_name,
        "--reload-rtol",
        str(args.reload_rtol),
        "--reload-atol",
        str(args.reload_atol),
    ]
    verification = subprocess.run(
        verification_command,
        cwd=C.REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if verification.returncode != 0:
        raise RuntimeError(
            "Fresh-process reload verification failed.\n"
            f"stdout:\n{verification.stdout}\nstderr:\n{verification.stderr}"
        )
    with paths["reload_check"].open("r", encoding="utf-8") as stream:
        reload_check = json.load(stream)
    training_log["reload_verification"] = reload_check
    training_log["hashes"]["reload_check_sha256"] = sha256_file(paths["reload_check"])
    training_log["finished_utc"] = _utc_now()
    training_log["wall_time_seconds"] = time.perf_counter() - start_time
    _atomic_json(paths["training_log"], training_log)
    training_log_sha256 = sha256_file(paths["training_log"])

    C.update_manifest(
        {
            "environment": {"gpu_and_torch_runtime": determinism},
            "seeds_and_determinism": {
                "training_runtime": determinism,
                "ensemble_seed_policy": list(ENSEMBLE_SEEDS),
            },
            "training_configuration": {
                "model_runs": {
                    run_name: {
                        "arguments": training_log["arguments"],
                        "architecture": training_log["architecture"],
                        "optimisation": training_log["optimisation"],
                    }
                }
            },
            "training_runs": {
                run_name: {
                    "training_log": str(paths["training_log"].relative_to(C.RUN_ROOT)),
                    "checkpoint": str(paths["checkpoint"].relative_to(C.RUN_ROOT)),
                    "seed": args.seed,
                    "split": args.split,
                    "variant": args.variant,
                    "best_epoch": best_epoch,
                    "stopping_epoch": history[-1]["epoch"],
                    "parameter_count": parameter_count,
                    "split_hashes": split_hashes,
                    "code_hashes": training_log["code_hashes"],
                    "hashes": training_log["hashes"],
                    "training_log_sha256": training_log_sha256,
                    "reload_verification_passed": reload_check["passed"],
                    "audited_evidence_mode": C.AUDITED_EVIDENCE_MODE,
                    "evidence_gates": evidence_gates,
                }
            },
        }
    )
    print(f"saved and fresh-process verified all artefacts for {run_name}")


if __name__ == "__main__":
    main()
