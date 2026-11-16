"""Declarative, fail-fast command graph for the ATAA evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: str
    arguments: tuple[str, ...] = ()
    requires_locked_loss: bool = False
    outputs: tuple[str, ...] = ()

    def command(self, scripts_dir: Path, python_executable: str = sys.executable) -> list[str]:
        return [python_executable, str(scripts_dir / self.script), *self.arguments]


PIPELINE_STEPS = (
    PipelineStep("convert", "convert_outputs.py", outputs=("data/sim_outputs_full.npy",)),
    PipelineStep("diagnose", "diagnose_data.py", outputs=("outputs/bad_rows.npy",)),
    PipelineStep("splits", "build_splits.py", outputs=("splits/grouped_split.npz",)),
    *tuple(
        PipelineStep(
            f"train-grouped-seed{seed}",
            "train_cardiac_cycle.py",
            ("--split", "grouped", "--seed", str(seed)),
            requires_locked_loss=True,
            outputs=(f"models/cardiac_grouped_free_seed{seed}_train_log.json",),
        )
        for seed in range(42, 47)
    ),
    PipelineStep(
        "train-random-seed42",
        "train_cardiac_cycle.py",
        ("--split", "random", "--seed", "42"),
        requires_locked_loss=True,
        outputs=("models/cardiac_random_free_seed42_train_log.json",),
    ),
    PipelineStep("metrics-grouped", "stage4_metrics.py", ("--split", "grouped"), outputs=("outputs/metrics_grouped_summary.json",)),
    PipelineStep("metrics-random", "stage4_metrics.py", ("--split", "random"), outputs=("outputs/metrics_random_summary.json",)),
    PipelineStep("transfer", "ps_transfer.py", outputs=("outputs/metrics_patient_summary.json",)),
    PipelineStep("principal-pws-spd", "stage4_principal.py", outputs=("outputs/pws_spd_summary.json",)),
    PipelineStep("mesh", "mesh_prep.py", outputs=("outputs/mesh_points.npy",)),
    PipelineStep("boundary", "boundary_pws.py", outputs=("outputs/boundary_pws_summary.json",)),
    PipelineStep("pressure", "pressure_dependence.py", outputs=("outputs/pressure_dependence_summary.json",)),
    PipelineStep("percentiles", "percentiles.py", outputs=("outputs/percentiles_summary.json",)),
    PipelineStep("mc-dropout", "mc_dropout.py", ("--seed", "42", "--passes", "50"), outputs=("outputs/mc_dropout_summary.json",)),
    PipelineStep("ensemble", "ensemble_uq.py", outputs=("outputs/ensemble_summary.json",)),
    PipelineStep("meanfield", "baseline_meanfield.py", outputs=("outputs/meanfield_summary.json",)),
    PipelineStep("legacy-meanfield", "meanfield_componentwise.py", outputs=("outputs/meanfield_componentwise_summary.json",)),
    PipelineStep("baselines", "baselines.py", outputs=("outputs/baselines_summary.json",)),
    PipelineStep("figures-primary", "make_figures.py", outputs=("figures/fig_generalisation_cdf.pdf",)),
    PipelineStep("figures-secondary", "make_figures2.py", outputs=("figures/fig_pws_bland_altman.pdf",)),
    PipelineStep("timing", "timing_benchmark.py", outputs=("outputs/timing_benchmark.json",)),
    PipelineStep("results-bundle", "build_results_report.py", outputs=("outputs/manuscript_values.json",)),
    PipelineStep("release-staging", "prepare_release.py", outputs=("release/ataa-stress-surrogate/models/cardiac_grouped_free.pth",)),
    PipelineStep("privacy-audit", "audit_release.py", outputs=("audit/privacy_audit.json",)),
    PipelineStep("finalise", "finalize_run.py", outputs=("RUN_COMPLETE.json",)),
)


def selected_steps(start: str | None = None, stop: str | None = None) -> tuple[PipelineStep, ...]:
    names = [step.name for step in PIPELINE_STEPS]
    if len(names) != len(set(names)):
        raise RuntimeError("pipeline step names are not unique")
    first = 0 if start is None else names.index(start)
    last = len(names) - 1 if stop is None else names.index(stop)
    if first > last:
        raise ValueError(f"pipeline start {start!r} occurs after stop {stop!r}")
    return PIPELINE_STEPS[first : last + 1]


def output_collisions(run_root: str | Path, steps: tuple[PipelineStep, ...]) -> list[str]:
    root = Path(run_root)
    return [
        str(root / relative)
        for step in steps
        for relative in step.outputs
        if (root / relative).exists()
    ]
