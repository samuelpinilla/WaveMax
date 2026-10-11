"""Distance up to the trivial ambiguities of the FrFT-based AF (paper Eq. 11 / 35).

Every row of the FrFT-based AF is a Fourier magnitude ``|fft(|F_alpha x|^2)|``,
which is invariant to any shift of the intensity ``|F_alpha x|^2``. A time shift
``a`` and Doppler modulation ``b`` of ``x`` shift every such intensity, so in the
continuous model both are exact ambiguities for any real value. Eq. (35) states
the shift for integer ``a``; by default the shift here is continuous
(``a in R``, applied as a band-limited FFT phase ramp), so that sub-sample
delays - which no AF measurement can resolve - are not counted as error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

__all__ = ["Alignment", "align", "distance", "relative_error", "fractional_shift"]


@dataclass(frozen=True)
class Alignment:
    """The element ``z = e^{i beta} e^{i b n} x[eps*n - a]`` of T(x) closest to q.

    ``a`` is a float when the shift is continuous (integer part = cyclic shift,
    remainder = sub-sample shift applied after the reflection).
    """

    distance: float
    relative_error: float
    z: np.ndarray
    beta: float
    b: float
    eps: int
    a: float


def fractional_shift(x: np.ndarray, tau: float) -> np.ndarray:
    """Band-limited cyclic shift ``x[n - tau]`` for real ``tau`` (FFT phase ramp)."""
    w = np.fft.fftfreq(len(x)) * 2 * np.pi
    return np.fft.ifft(np.fft.fft(x) * np.exp(-1j * w * tau))


def _candidates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ``x[(eps*n - a) mod N]`` for eps in {+1, -1}, a in 0..N-1."""
    n = len(x)
    idx = np.arange(n)
    eps = np.repeat([1, -1], n)
    a = np.tile(idx, 2)
    rows = (eps[:, None] * idx[None, :] - a[:, None]) % n
    return x[rows], eps, a


def _best_modulation(z0: np.ndarray, q: np.ndarray, center: float, step: float) -> tuple[float, float]:
    """``max_b |sum conj(z0) q e^{-ibn}|`` near ``center``: returns (b, correlation)."""
    ns = np.arange(len(q))
    p = np.conj(z0) * q
    res = minimize_scalar(lambda b: -np.abs(np.sum(p * np.exp(-1j * b * ns))),
                          bounds=(center - step, center + step), method="bounded",
                          options={"xatol": 1e-13})
    return float(res.x), float(-res.fun)


def align(x, q, oversample: int = 16, refine: int = 4, continuous_shift: bool = True) -> Alignment:
    """Minimize ``||q - z||_2`` over ``z in T(x)``: global phase beta, Doppler
    modulation e^{ibn} (b continuous), reflection eps and shift a.

    For fixed (eps, a), ``min_{beta,b} ||q - e^{i beta} e^{ibn} z0||^2 =
    ||q||^2 + ||z0||^2 - 2 max_b |sum_n conj(z0[n]) q[n] e^{-ibn}|``, so beta is
    closed-form and b is found on an ``oversample``-times zero-padded FFT grid,
    then refined for the ``refine`` best integer (eps, a). With
    ``continuous_shift`` (default) a sub-sample shift in [-1/2, 1/2] is then
    optimized jointly with b; ``continuous_shift=False`` is Eq. (35) verbatim.
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    q = np.asarray(q, dtype=np.complex128).ravel()
    n = len(x)
    ns = np.arange(n)
    cands, eps, a = _candidates(x)
    size = n * oversample
    spectra = np.abs(np.fft.fft(np.conj(cands) * q[None, :], size, axis=-1))
    grid_best = spectra.argmax(axis=-1)
    order = np.argsort(-spectra[np.arange(len(cands)), grid_best])[:refine]
    step = 2 * np.pi / size

    best = None  # (correlation, row, tau, b)
    for row in order:
        b, corr = _best_modulation(cands[row], q, grid_best[row] * step, step)
        tau = 0.0
        if continuous_shift:
            def neg_corr(p, base=cands[row]):
                z0 = fractional_shift(base, p[0])
                return -np.abs(np.sum(np.conj(z0) * q * np.exp(-1j * p[1] * ns)))

            res = minimize(neg_corr, [0.0, b], method="Nelder-Mead",
                           options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 4000})
            if -res.fun > corr and abs(res.x[0]) <= 0.5:
                tau, b, corr = float(res.x[0]), float(res.x[1]), float(-res.fun)
        if best is None or corr > best[0]:
            best = (corr, row, tau, b)

    _, row, tau, b = best
    z0 = fractional_shift(cands[row], tau) if tau else cands[row]
    inner = np.sum(np.conj(z0) * q * np.exp(-1j * b * ns))
    beta = float(np.angle(inner))
    z = np.exp(1j * beta) * np.exp(1j * b * ns) * z0
    dist = float(np.linalg.norm(q - z))
    return Alignment(
        distance=dist,
        relative_error=dist / float(np.linalg.norm(x)),
        z=z,
        beta=beta,
        b=float(np.mod(b, 2 * np.pi)),
        eps=int(eps[row]),
        a=float(a[row]) + tau,
    )


def distance(x, q, **kwargs) -> float:
    """``dist(x, q)`` of Eq. (11)."""
    return align(x, q, **kwargs).distance


def relative_error(x, q, **kwargs) -> float:
    """``dist(x, q) / ||x||_2``, the relative error reported in the paper."""
    return align(x, q, **kwargs).relative_error
