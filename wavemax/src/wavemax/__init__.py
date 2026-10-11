"""WaveMax: radar waveform recovery from its FrFT-based ambiguity function.

Implementation of Pinilla et al., "WaveMax: Radar Waveform Design via Convex
Maximization", IEEE TSP 73 (2025): spectral initialization, the lifted convex
program (14) and a local polish, built on the operators, penalties and solvers
of :mod:`invexapi`.
"""

from .data import DEFAULT_PULSES, load_pulse, load_pulses
from .distance import Alignment, align, distance, relative_error
from .initialization import estimate_norm, spectral_init
from .solver import WaveMaxResult, leading_component, refine, trace_weight, wavemax

__all__ = [
    "DEFAULT_PULSES",
    "load_pulse",
    "load_pulses",
    "Alignment",
    "align",
    "distance",
    "relative_error",
    "estimate_norm",
    "spectral_init",
    "WaveMaxResult",
    "leading_component",
    "refine",
    "trace_weight",
    "wavemax",
]
