"""WaveMax (paper Algorithm 2): convex lifted recovery of x from its FrFT-based AF.

Solves problem (14) in a penalized, slack-relaxed form::

    min_Z  0.5*|| |B(Z)| + s - b ||^2  -  mu*Re<X0, Z>  +  mu*tr(Z)  +  i_PSD(Z)

with ``b = sqrt(A)``, ``B`` the lifted measurement operator and the slack ``s``
refreshed between FASTA rounds as ``s = max(b - |B(Z)|, 0)``. Each building block
is an invexapi object: ``LiftedFrFTAmbiguity`` (B), ``SlackAmplitudeLeastSquares``
(smooth term), ``LinearTilt(PSDTracePenalty(mu), mu*X0)`` (proximal term) and ``FASTA``.
The rank-1 estimate is then polished by ``LevenbergMarquardt`` on the signal-domain
``FrFTAmbiguity`` operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from invexapi import SlackAmplitudeLeastSquares, LinearTilt, PSDTracePenalty
from invexapi.operators import LiftedFrFTAmbiguity
from invexapi.optim import FASTA, LevenbergMarquardt

__all__ = ["WaveMaxResult", "trace_weight", "leading_component", "refine", "wavemax"]


@dataclass
class WaveMaxResult:
    x: torch.Tensor
    Z: torch.Tensor
    mu: float
    x_convex: Optional[torch.Tensor] = None
    rounds: List[dict] = field(default_factory=list)
    refine_history: List[float] = field(default_factory=list)

    @property
    def residuals(self) -> List[float]:
        return [r for info in self.rounds for r in info["residuals"]]


def trace_weight(op: LiftedFrFTAmbiguity, b: torch.Tensor, tol: float) -> float:
    """``mu = tol * ||eig(B^*(b))||_1``: the alignment weight, scaled to the data."""
    return tol * torch.linalg.eigvals(op.adjoint(b)).abs().sum().item()


def leading_component(Z: torch.Tensor) -> torch.Tensor:
    """``sqrt(lambda_1) v_1`` of the Hermitian part of ``Z`` (Algorithm 2, line 14)."""
    d, V = torch.linalg.eigh(0.5 * (Z + Z.conj().T))
    return torch.sqrt(d[-1].clamp(min=0.0)) * V[:, -1]


def refine(
    op: LiftedFrFTAmbiguity, af: torch.Tensor, x: torch.Tensor, max_iter: int = 150
) -> tuple[torch.Tensor, List[float]]:
    """Polish ``x`` by Levenberg-Marquardt on the intensity misfit ``A(x) - A``.

    The lifted convex stage only needs to land in the basin of this local
    problem (roughly 30% relative error); the polish then fits the data to the
    level the discrete FrFT allows.
    """
    solver = LevenbergMarquardt(op.signal_operator(), af, max_iter=max_iter)
    return solver.run(x)


def wavemax(
    op: LiftedFrFTAmbiguity,
    af: torch.Tensor,
    x0: torch.Tensor,
    outer_iters: int = 3,
    max_iter: int = 200,
    tol: float = 1e-5,
    mu: Optional[float] = None,
    mu_decay: float = 0.1,
    slack: str = "refresh",
    refine_iters: int = 150,
    verbose: int = 0,
    generator: Optional[torch.Generator] = None,
) -> WaveMaxResult:
    """Recover ``x`` (up to trivial ambiguities) from ``af = op.ambiguity(x)``.

    ``x0`` is the anchor of Eq. (14), normally from
    :func:`wavemax.initialization.spectral_init`. Each of ``outer_iters`` rounds
    runs FASTA on the lifted problem; between rounds ``mu`` is multiplied by
    ``mu_decay`` (continuation: a fixed mu biases the minimizer by O(mu)) and the
    slack is refreshed as ``max(b - |B(Z)|, 0)`` (``slack="refresh"``) or kept
    at zero (``slack="none"``). ``refine_iters > 0`` then
    polishes the rank-1 estimate with :func:`refine`.
    """
    if slack not in ("refresh", "none"):
        raise ValueError(f"slack must be 'refresh' or 'none', got {slack!r}")
    b = torch.sqrt(af.clamp(min=0.0))
    X0 = torch.outer(x0, x0.conj())
    if mu is None:
        mu = trace_weight(op, b, tol)

    s = torch.zeros_like(b)
    Z = X0.clone()
    rounds = []
    mu_r = mu
    for r in range(outer_iters):
        solver = FASTA(
            SlackAmplitudeLeastSquares(b, slack=s),
            LinearTilt(PSDTracePenalty(lamb=mu_r), mu_r * X0),
            A=op,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            generator=generator,
        )
        Z, _ = solver.run(Z)
        rounds.append(solver.info)
        if slack == "refresh":
            s = torch.clamp(b - op.apply(Z).abs(), min=0.0)
        mu_r *= mu_decay
        if verbose:
            print(f"WaveMax round {r + 1}/{outer_iters}: {solver.info['iterations']} FASTA iterations")

    x_convex = leading_component(Z)
    x_hat, history = x_convex, []
    if refine_iters > 0:
        x_hat, history = refine(op, af, x_convex, max_iter=refine_iters)
    return WaveMaxResult(x=x_hat, Z=Z, mu=mu, x_convex=x_convex, rounds=rounds, refine_history=history)
