"""Single NumPy implementation of the manuscript's field and agreement metrics."""

from __future__ import annotations

from typing import Mapping

import numpy as np


METRIC_POLICY = {
    "sample_weighting": "each case contributes one equally weighted metric value",
    "finite_input_policy": "raise on any NaN or infinity",
    "degenerate_denominator_policy": "return NaN for the affected per-case metric",
    "summary_nan_policy": "omit only metric-specific degeneracies; never omit non-finite inputs",
    "variance_convention": "population variance/covariance (ddof=0) for CCC",
    "summary_convention": "median, arithmetic mean, and linear-interpolated 25th/75th percentiles",
    "mask_convention": "apply the same declared node mask to prediction and reference before case metrics",
    "bland_altman_convention": "predicted-reference; bias=mean; LoA=bias±1.96*sample-SD (ddof=1)",
    "pws_location_convention": "Euclidean distance between first argmax nodes on the common mesh",
}


def _paired_fields(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != ref.shape or pred.shape[1] == 0:
        raise ValueError("predicted and reference fields must be matching non-empty 2D arrays")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("metric inputs contain NaN or infinity")
    return pred, ref


def per_sample_field_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate one metric value per case; cases are never pooled across nodes."""

    pred, ref = _paired_fields(predicted, reference)
    diff = pred - ref
    ss_res = np.sum(diff * diff, axis=1)
    ref_centred = ref - ref.mean(axis=1, keepdims=True)
    pred_centred = pred - pred.mean(axis=1, keepdims=True)
    ss_tot = np.sum(ref_centred * ref_centred, axis=1)
    ref_norm = np.linalg.norm(ref, axis=1)
    ranges = np.ptp(ref, axis=1)
    covariance = np.mean(pred_centred * ref_centred, axis=1)
    pred_variance = np.mean(pred_centred * pred_centred, axis=1)
    ref_variance = np.mean(ref_centred * ref_centred, axis=1)
    pearson_denom = np.sqrt(pred_variance * ref_variance)
    ccc_denom = pred_variance + ref_variance + (pred.mean(1) - ref.mean(1)) ** 2

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.full(numerator.shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    return {
        "relL2": divide(np.linalg.norm(diff, axis=1), ref_norm),
        "R2": 1.0 - divide(ss_res, ss_tot),
        "pearson": divide(covariance, pearson_denom),
        "ccc": divide(2.0 * covariance, ccc_denom),
        "rmse": np.sqrt(np.mean(diff * diff, axis=1)),
        "mae": np.mean(np.abs(diff), axis=1),
        "nmae_range": divide(np.mean(np.abs(diff), axis=1), ranges),
    }


def summarise_metric_arrays(metrics: Mapping[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    """Summarise per-case metrics with equal case weights and explicit finite counts."""

    summary: dict[str, dict[str, float | int]] = {}
    for name, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            stats = {"median": np.nan, "q1": np.nan, "q3": np.nan, "mean": np.nan}
        else:
            stats = {
                "median": float(np.median(finite)),
                "q1": float(np.quantile(finite, 0.25)),
                "q3": float(np.quantile(finite, 0.75)),
                "mean": float(np.mean(finite)),
            }
        summary[name] = {
            **stats,
            "n_total": int(array.size),
            "n_finite": int(finite.size),
            "n_degenerate": int(array.size - finite.size),
        }
    return summary


def field_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compatibility summary: medians of the four public demonstration metrics."""

    summary = summarise_metric_arrays(per_sample_field_metrics(predicted, reference))
    return {
        "R2": float(summary["R2"]["median"]),
        "relL2": float(summary["relL2"]["median"]),
        "MAE": float(summary["mae"]["median"]),
        "pearson": float(summary["pearson"]["median"]),
    }


def bland_altman(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    """Bland--Altman bias and 95% limits for paired scalar measurements."""

    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if pred.shape != ref.shape or pred.size < 2:
        raise ValueError("Bland--Altman analysis requires at least two paired values")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Bland--Altman inputs contain NaN or infinity")
    differences = pred - ref
    bias = float(differences.mean())
    sd = float(differences.std(ddof=1))
    return {
        "n": int(pred.size),
        "bias": bias,
        "difference_sd_ddof1": sd,
        "lower_loa": bias - 1.96 * sd,
        "upper_loa": bias + 1.96 * sd,
    }


def euclidean_peak_location_error(
    predicted_field: np.ndarray,
    reference_field: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Peak-S1 node disagreement on one common template; not anatomical or geodesic."""

    pred, ref = _paired_fields(predicted_field, reference_field)
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape != (pred.shape[1], 3) or not np.isfinite(coordinates).all():
        raise ValueError("points must be a finite (nodes, 3) common-mesh coordinate array")
    pred_nodes = np.argmax(pred, axis=1)
    ref_nodes = np.argmax(ref, axis=1)
    return np.linalg.norm(coordinates[pred_nodes] - coordinates[ref_nodes], axis=1)
