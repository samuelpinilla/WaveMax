"""Figures for pulse recovery: intensity/phase comparisons and solver convergence."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

__all__ = ["plot_comparison", "plot_convergence"]


def _masked_phase(signal: np.ndarray, reference_intensity: np.ndarray) -> np.ndarray:
    """Unwrapped phase, NaN where the reference intensity is < 5e-3 of its peak."""
    valid = reference_intensity >= reference_intensity.max() * 5e-3
    phase = np.full(signal.shape, np.nan)
    if valid.any():
        idx = np.flatnonzero(valid)
        phase[idx] = np.unwrap(np.angle(signal[idx]))
    return phase


def plot_comparison(original, estimate, title: str, path: Optional[Path] = None):
    """Normalized intensity and unwrapped phase of ``original`` vs ``estimate``.

    ``original`` should already be aligned to ``estimate`` (e.g. ``Alignment.z``).
    """
    import matplotlib.pyplot as plt

    original = np.asarray(original)
    estimate = np.asarray(estimate)
    io = np.abs(original) ** 2
    ie = np.abs(estimate) ** 2
    n = np.arange(1, len(original) + 1)

    fig, (ax_i, ax_p) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax_i.plot(n, io / io.max(), label="original (aligned)")
    ax_i.plot(n, ie / ie.max(), "--", label="estimate")
    ax_i.set(title=f"{title}: intensity", xlabel="sample n", ylabel="|x[n]|² (normalized)")
    ax_i.legend()
    ax_p.plot(n, _masked_phase(original, io), label="original (aligned)")
    ax_p.plot(n, _masked_phase(estimate, io), "--", label="estimate")
    ax_p.set(title=f"{title}: phase", xlabel="sample n", ylabel="unwrapped phase (rad)")
    ax_p.legend()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_convergence(rounds: List[dict], path: Optional[Path] = None):
    """FASTA residuals of every WaveMax round, concatenated (log scale)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    start = 0
    for r, info in enumerate(rounds, start=1):
        res = np.asarray(info["residuals"])
        ax.semilogy(np.arange(start, start + len(res)) + 1, res, label=f"round {r}")
        start += len(res)
    ax.set(title="Convergence curve: wavemax", xlabel="FASTA iterations", ylabel="residual")
    ax.legend()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
