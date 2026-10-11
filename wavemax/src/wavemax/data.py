"""Loading the benchmark radar pulse set (``pulse_set.mat``)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

__all__ = ["DEFAULT_PULSES", "load_pulses", "load_pulse"]

DEFAULT_PULSES = Path(__file__).resolve().parents[2] / "data" / "pulse_set.mat"


def load_pulses(path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """The ``(num_pulses, N)`` complex ``pulse_set`` array from ``pulse_set.mat``."""
    from scipy.io import loadmat

    return np.asarray(loadmat(str(path or DEFAULT_PULSES))["pulse_set"])


def load_pulse(
    index: int,
    path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Pulse ``index`` (0-based row of ``pulse_set``) as a complex128 tensor."""
    return torch.as_tensor(load_pulses(path)[index], dtype=torch.complex128, device=device)
