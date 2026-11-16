"""Measured inference cost of the trained surrogate, for the clinical-workflow claim.

What this script does and does not establish:

* It MEASURES the surrogate side end to end on declared hardware: checkpoint and
  scaler load, forward pass, inverse output scaling, and the eigenvalue
  post-processing that turns raw tensor components into the reported first
  principal stress. That is the honest "time to a usable result".
* It does NOT re-time the finite-element reference. SimVascular is not run here,
  and no CSM solve time was recorded in the archive. The reference figure is
  inherited from the original study and is reported as such, on its own stated
  hardware. The comparison is therefore an order-of-magnitude statement between
  two different machines, not a controlled hardware-matched benchmark, and the
  manuscript must say so.

The trained model is read from the frozen evidence run; this script writes only
into its own new run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))
import config as C  # noqa: E402
from result_provenance import artifact_record, result_document, write_json  # noqa: E402

sys.path.insert(
    0,
    str(ANALYSIS_DIR.parent / "release" / "ataa-stress-surrogate" / "src"),
)

FROZEN_RUN = Path(
    os.environ.get(
        "ATAA_FROZEN_RUN",
        C.RUN_ROOT,
    )
)
PRIMARY = "cardiac_grouped_free_seed42"
N_NODES = 1354
S_LEN = N_NODES * 6

# Inherited reference, not measured here. Source: original study, Section 2.4.
CSM_REFERENCE = {
    "measured_by_this_script": False,
    "source": "inherited from the original submission; not re-measured",
    "solver": "SimVascular",
    "hardware": "Intel Xeon Gold 6242R 3.10 GHz, 40 cores",
    "wall_time_minutes_range": [20, 60],
}


def timed(function, repeats: int, warmup: int = 2):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return samples


def main() -> None:
    C.init_run()
    import torch  # noqa: E402  (after config pins the thread policy)
    from ataa_surrogate.inference import load_run_bundle, predict_raw  # noqa: E402
    from ataa_surrogate.tensors import first_principal

    inputs = np.loadtxt(C.INPUTS_FULL, delimiter=",").astype(np.float32)
    split = np.load(FROZEN_RUN / "splits" / "grouped_split.npz")
    test = np.sort(split["test"])
    raw_test = inputs[test]

    # --- checkpoint and scaler load ----------------------------------------
    load_samples = timed(lambda: load_run_bundle(FROZEN_RUN / "models", PRIMARY), repeats=5)
    bundle = load_run_bundle(FROZEN_RUN / "models", PRIMARY)

    # --- single-case latency: one loading state, the clinical unit of work ---
    single = raw_test[:1]
    single_forward = timed(lambda: predict_raw(bundle, single), repeats=50)
    single_full = timed(
        lambda: first_principal(predict_raw(bundle, single)[:, :S_LEN], n_nodes=N_NODES, scale=1e-4),
        repeats=50,
    )

    # --- one 20-state loading sequence --------------------------------------
    twenty = raw_test[:20]
    geometry_full = timed(
        lambda: first_principal(predict_raw(bundle, twenty)[:, :S_LEN], n_nodes=N_NODES, scale=1e-4),
        repeats=20,
    )

    # --- whole held-out test set --------------------------------------------
    batch_samples = timed(lambda: predict_raw(bundle, raw_test), repeats=3, warmup=1)

    def summarise(samples):
        return {
            "n_repeats": len(samples),
            "mean_s": float(statistics.mean(samples)),
            "median_s": float(statistics.median(samples)),
            "min_s": float(min(samples)),
            "stdev_s": float(statistics.stdev(samples)) if len(samples) > 1 else None,
        }

    single_median = statistics.median(single_full)
    sequence_median = statistics.median(geometry_full)
    ratio_low = CSM_REFERENCE["wall_time_minutes_range"][0] * 60 / sequence_median
    ratio_high = CSM_REFERENCE["wall_time_minutes_range"][1] * 60 / sequence_median

    results = {
        "hardware": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "run_name": PRIMARY,
            "source_run": FROZEN_RUN.name,
            "parameters": int(sum(p.numel() for p in bundle.model.parameters())),
        },
        "checkpoint_and_scaler_load": summarise(load_samples),
        "single_state_forward_only": summarise(single_forward),
        "single_state_to_first_principal_kpa": summarise(single_full),
        "one_sequence_20_states_to_first_principal_kpa": summarise(geometry_full),
        "full_test_set_forward_only": {
            **summarise(batch_samples),
            "n_rows": int(raw_test.shape[0]),
            "per_row_s": float(statistics.median(batch_samples) / raw_test.shape[0]),
        },
        "csm_reference": CSM_REFERENCE,
        "speedup_versus_inherited_csm_reference": {
            "definition": (
                "inherited wall time for one 20-state CSM sequence divided by the measured "
                "20-state surrogate time to first principal stress; the two were obtained on "
                "different machines"
            ),
            "low_estimate": float(ratio_low),
            "high_estimate": float(ratio_high),
            "hardware_matched": False,
        },
    }

    document = result_document(
        run_id=C.RUN_ID,
        analysis="surrogate_inference_timing_benchmark",
        model_runs=[PRIMARY],
        definitions={
            "timing_method": "time.perf_counter around warmed-up repeated calls",
            "amortisation": "checkpoint and scaler load timed separately, excluded from per-case cost",
            "post_processing_included": (
                "symmetric eigenvalue solve over 1,354 nodes and conversion to kPa"
            ),
            "csm_side": "inherited figure, not measured by this script",
        },
        results=results,
    )
    path = os.path.join(C.OUTPUTS_DIR, "timing_benchmark.json")
    write_json(path, document)
    document["artefacts"] = {"summary": artifact_record(path, C.RUN_ROOT)}
    write_json(path, document)

    print(f"hardware: {results['hardware']['logical_cpu_count']} logical CPUs, "
          f"{results['hardware']['torch_threads']} torch threads, CPU only")
    print(f"load checkpoint + scalers : {summarise(load_samples)['median_s']*1e3:8.1f} ms (once)")
    print(f"one loading state -> kPa  : {single_median*1e3:8.1f} ms")
    print(f"one loading sequence (20 states): {sequence_median*1e3:8.1f} ms")
    print(f"full test set ({raw_test.shape[0]} rows; forward + inverse scaling only): "
          f"{statistics.median(batch_samples):8.2f} s "
          f"({results['full_test_set_forward_only']['per_row_s']*1e3:.2f} ms/row)")
    print(f"versus inherited CSM 20-60 min: {ratio_low:,.0f}x to {ratio_high:,.0f}x "
          f"(different hardware, not matched)")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
