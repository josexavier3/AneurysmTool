"""Audited Ghazi-derived objective for jointly predicted ATAA tensor fields.

Ghazi et al. define a raw-output objective proportional to
``sigma_i * (prediction_i - target_i)^2``. If target ``y`` is trained as
``z=(y-mu)/scale``, the exactly equivalent standardised-space weight is
``sigma_i * scale_i**2`` -- approximately ``sigma_i**3`` when scale is the
population standard deviation. The previous ATAA weight ``sigma_i`` in z-space
instead produced an effective raw-space weight approximately ``1/sigma_i``.

Stress and strain have different physical units. The ATAA adaptation therefore
normalises the equivalent weights within each field and averages the two field
losses equally. Relative Ghazi weighting is preserved within S and within C,
without allowing the arbitrary choice of stress unit to determine task balance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


GHAZI_LOSS_DECISION = {
    "id": "ataa_group_balanced_ghazi_v1",
    "source_equation": "Ghazi et al. 2021 Eq. (1): raw-output SD-weighted MSE",
    "standard_deviation": "population (ddof=0), fitted on training rows only",
    "standardised_equivalent_weight": "raw_sd * output_scaler_scale**2",
    "field_balance": "normalise weights to mean 1 within S and C; average fields equally",
    "rationale": "preserve within-field Ghazi weighting without mixing stress and strain units",
}
LEGACY_STANDARDISED_SIGMA_DECISION = {
    "id": "legacy_standardised_sigma_v0",
    "source": "former rebuild policy before I03",
    "weight": "output_scaler_scale / mean(output_scaler_scale)",
    "space": "standardised target space",
    "interpretation": "not equivalent to Ghazi raw-output SD weighting",
}

LOSS_POLICY_IDS = (
    LEGACY_STANDARDISED_SIGMA_DECISION["id"],
    GHAZI_LOSS_DECISION["id"],
)


def _group_bounds(width: int, group_widths: Sequence[int]) -> tuple[tuple[int, int], ...]:
    widths = tuple(int(value) for value in group_widths)
    if not widths or any(value <= 0 for value in widths) or sum(widths) != width:
        raise ValueError("group widths must be positive and sum to the output width")
    bounds: list[tuple[int, int]] = []
    start = 0
    for value in widths:
        bounds.append((start, start + value))
        start += value
    return tuple(bounds)


def group_balanced_ghazi_weights(
    raw_output_sd: np.ndarray,
    output_scaler_scale: np.ndarray,
    group_widths: Sequence[int],
) -> np.ndarray:
    """Return standardised-space weights equivalent to Ghazi within each field."""

    raw_sd = np.asarray(raw_output_sd, dtype=np.float64)
    scaler_scale = np.asarray(output_scaler_scale, dtype=np.float64)
    if raw_sd.ndim != 1 or raw_sd.shape != scaler_scale.shape:
        raise ValueError("raw SD and output scale must be matching one-dimensional arrays")
    if not np.isfinite(raw_sd).all() or not np.isfinite(scaler_scale).all():
        raise ValueError("loss scales must be finite")
    if np.any(raw_sd < 0) or np.any(scaler_scale <= 0):
        raise ValueError("raw SD must be non-negative and scaler scale strictly positive")
    weights = raw_sd * scaler_scale**2
    for start, stop in _group_bounds(weights.size, group_widths):
        group_mean = float(weights[start:stop].mean())
        if group_mean <= 0:
            raise ValueError("every output field must contain at least one varying component")
        weights[start:stop] /= group_mean
    return weights.astype(np.float32)


def legacy_standardised_sigma_weights(output_scaler_scale: np.ndarray) -> np.ndarray:
    """Reproduce the former rebuild's sigma weights in standardised target space."""

    scale = np.asarray(output_scaler_scale, dtype=np.float64)
    if scale.ndim != 1 or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("output scale must be a finite, strictly positive vector")
    return (scale / scale.mean()).astype(np.float32)


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return the former rebuild's globally averaged weighted MSE."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching two-dimensional tensors")
    if weights.ndim != 1 or weights.numel() != prediction.shape[1]:
        raise ValueError("weight vector is incompatible with the prediction width")
    return ((prediction - target) ** 2 * weights).mean()


def weighted_field_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    group_widths: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Return each field's weighted MSE for contribution tracking."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching two-dimensional tensors")
    if weights.ndim != 1 or weights.numel() != prediction.shape[1]:
        raise ValueError("weight vector is incompatible with the prediction width")
    squared_error = (prediction - target) ** 2
    return tuple(
        (squared_error[:, start:stop] * weights[start:stop]).mean()
        for start, stop in _group_bounds(prediction.shape[1], group_widths)
    )


def group_balanced_ghazi_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    group_widths: Sequence[int],
) -> torch.Tensor:
    """Mean of the equally weighted, within-field Ghazi losses."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching two-dimensional tensors")
    if weights.ndim != 1 or weights.numel() != prediction.shape[1]:
        raise ValueError("weight vector is incompatible with the prediction width")
    field_losses = weighted_field_mse(
        prediction, target, weights, group_widths
    )
    return torch.stack(field_losses).mean()
