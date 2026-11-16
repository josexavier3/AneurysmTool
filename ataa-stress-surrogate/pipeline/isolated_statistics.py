"""Statistics evaluated outside processes that have initialised PyTorch OpenMP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def spearmanr_isolated(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return SciPy's Spearman statistic and p-value from a clean child process."""

    with tempfile.TemporaryDirectory() as temporary:
        values_path = Path(temporary) / "values.npz"
        np.savez(values_path, x=np.asarray(x), y=np.asarray(y))
        code = (
            "import json, numpy as np, sys; "
            "from scipy.stats import spearmanr; "
            "z=np.load(sys.argv[1]); r=spearmanr(z['x'], z['y']); "
            "print(json.dumps([float(r.statistic), float(r.pvalue)]))"
        )
        environment = os.environ.copy()
        environment.pop("KMP_DUPLICATE_LIB_OK", None)
        completed = subprocess.run(
            [sys.executable, "-c", code, str(values_path)],
            capture_output=True,
            check=True,
            env=environment,
            text=True,
            timeout=120,
        )
    rho, p_value = json.loads(completed.stdout)
    return float(rho), float(p_value)
