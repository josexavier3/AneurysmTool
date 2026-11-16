"""Canonical PyTorch model definition for ATAA wall-mechanics inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from .constants import IN_DIM, N_NODES, OUT_DIM

DEFAULT_HIDDEN = (64, 128, 256, 512, 1024)
DEFAULT_DROPOUT = 0.2


class SurrogateMLP(nn.Module):
    """Fully connected surrogate used by training and every inference consumer."""

    def __init__(
        self,
        in_dim: int = IN_DIM,
        out_dim: int = OUT_DIM,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or not hidden or any(width <= 0 for width in hidden):
            raise ValueError("model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        layers: list[nn.Module] = []
        width = int(in_dim)
        for next_width in hidden:
            layers.extend((nn.Linear(width, int(next_width)), nn.ReLU()))
            width = int(next_width)
        layers.extend((nn.Dropout(float(dropout)), nn.Linear(width, int(out_dim))))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def load_state_dict_file(path: str | Path) -> Mapping[str, torch.Tensor]:
    """Load a weights-only state dict, retaining compatibility with older PyTorch."""

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch only
        return torch.load(Path(path), map_location="cpu")
