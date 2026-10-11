"""Spectral initialization x^(0) of WaveMax (paper Algorithm 1, Eqs. 15-21)."""

from __future__ import annotations

import math
from typing import Optional

import torch

from invexapi.operators import LiftedFrFTAmbiguity

__all__ = ["spectral_init", "estimate_norm"]


def _lag_domain(af: torch.Tensor) -> torch.Tensor:
    """``Y[alpha, l] = (1/N) sum_k A[alpha, k] w^{kl}`` (Eq. 15)."""
    return torch.fft.ifft(af, dim=-1).real


def _centered_lags(Y: torch.Tensor) -> torch.Tensor:
    """Move lag 0 of ``Y[alpha, l]`` to index N//2, the centre of the FrFT grid.

    ``Y[alpha, .]`` is the autocorrelation of the intensity ``|F_alpha x|^2``, so
    its mass sits at small lags around l = 0. Read as FrFT sample indices (as
    Algorithm 1 does), uncentred lags select the edges of the centred FrFT grid;
    centred, they select a signal concentrated at time 0 and frequency 0 - itself
    a trivial ambiguity of x, since shifts and modulations leave the AF unchanged.
    """
    return torch.roll(Y, Y.shape[-1] // 2, dims=-1)


def estimate_norm(af: torch.Tensor) -> float:
    """``lambda_0 ~ ||x||_2`` from the ambiguity function (Eqs. 20-21).

    ``sum_l Y[alpha, l] = ||F_alpha x||^4``, so each order gives one estimate of
    ``||x||``; they are averaged over the K orders. (Eq. 21 writes the prefactor
    as 1/N, which coincides with the mean only when K = N orders are sampled.)
    """
    per_order = _lag_domain(af).sum(dim=-1).clamp(min=0.0) ** 0.25
    return per_order.mean().item()


def spectral_init(
    op: LiftedFrFTAmbiguity,
    af: torch.Tensor,
    keep_fraction: float = 1.0 / 6.0,
    iters: int = 100,
    generator: Optional[torch.Generator] = None,
    center_lags: bool = True,
) -> torch.Tensor:
    """Algorithm 1: power iteration on ``G_0 = (1/|I0c|) sum_{(n,a) in I0c} u u^H``.

    ``I0c`` keeps the ``floor(m * keep_fraction)`` largest ``Y[alpha, l]`` (the
    paper's ``floor(m/6)``) with the lag ``l`` taken as the FrFT sample ``n``.
    With ``center_lags`` (default) lag 0 is mapped to the grid centre first (see
    :func:`_centered_lags`); ``center_lags=False`` uses the lags as indices
    directly. The result is scaled by :func:`estimate_norm`.
    """
    k, m = op.shape
    Y = _lag_domain(af)
    if center_lags:
        Y = _centered_lags(Y)
    count = max(1, math.floor(k * m * keep_fraction))
    mask = torch.zeros(k * m, dtype=torch.float64, device=af.device)
    mask[torch.topk(Y.reshape(-1), count).indices] = 1.0
    mask = mask.reshape(k, m).to(op.matrices.dtype)

    z = torch.randn(op.n, dtype=op.matrices.dtype, device=af.device, generator=generator)
    z = z / torch.linalg.vector_norm(z)
    F = op.matrices
    for _ in range(iters):
        # G_0 z = (1/|I0c|) sum_alpha F_alpha^H (mask_alpha * F_alpha z); the 1/|I0c|
        # factor is dropped since the iterate is renormalized anyway.
        z = (F.conj().transpose(-1, -2) @ (mask * (F @ z)).unsqueeze(-1)).sum(dim=0).squeeze(-1)
        z = z / torch.linalg.vector_norm(z)
    return estimate_norm(af) * z
