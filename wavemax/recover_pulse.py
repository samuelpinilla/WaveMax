"""Recover a radar pulse from its noiseless FrFT-based ambiguity function.

Loads a pulse from ``pulse_set.mat`` (pulse 3 by default), simulates its
ambiguity function over ceil(N/L) FrFT orders in [0.5, 1.5], builds the
spectral initialization (Algorithm 1) and runs WaveMax
(Algorithm 2) - FASTA on the lifted problem (14) with three slack rounds and
mu continuation - then polishes the estimate by Levenberg-Marquardt. Relative
errors are dist(x, q)/||x|| from Eq. (11), reported with a continuous time shift
(default) and with the integer shift of Eq. (35).

    python wavemax/recover_pulse.py                                 # centred init, continuation, polish
    python wavemax/recover_pulse.py --mu-decay 1 --refine-iters 0   # convex stage alone, fixed mu
    python wavemax/recover_pulse.py --pulse-index 10 --decimation 2 --show

Expect invexapi's "no convex/invex certificate" warnings: the amplitude loss is
nonconvex and the tilted PSD penalty is deliberately left uncertified.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from invexapi.operators import LiftedFrFTAmbiguity, frft_orders
from wavemax import DEFAULT_PULSES, align, load_pulse, spectral_init, wavemax
from wavemax.plotting import plot_comparison, plot_convergence


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=DEFAULT_PULSES, help="pulse_set.mat path")
    p.add_argument("--pulse-index", type=int, default=2, help="0-based row of pulse_set (2 = pulse 3)")
    p.add_argument("--decimation", type=int, default=3, help="L: use ceil(N/L) FrFT orders")
    p.add_argument("--pad", type=int, default=1, help="zero-pad factor of the FrFT grid (2 = more accurate FrFT)")
    p.add_argument("--init-iters", type=int, default=100, help="power iterations (Algorithm 1)")
    p.add_argument("--keep-fraction", type=float, default=1 / 6, help="|I0c| / m (paper: 1/6)")
    p.add_argument("--outer-iters", type=int, default=3, help="slack rounds")
    p.add_argument("--max-iter", type=int, default=1000, help="FASTA iterations per round")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--mu-decay", type=float, default=0.1, help="mu continuation factor per round (1 = fixed)")
    p.add_argument("--slack", choices=("refresh", "none"), default="refresh")
    p.add_argument("--refine-iters", type=int, default=150, help="Levenberg-Marquardt polish (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("runs/recover_pulse"), help="figure directory")
    p.add_argument("--show", action="store_true", help="show figures interactively")
    p.add_argument("--verbose", type=int, default=1, help="0 quiet, 1 per round, 2 per iteration")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    x = load_pulse(args.pulse_index, args.data, device=device)
    n = x.shape[0]
    op = LiftedFrFTAmbiguity(n, frft_orders(n, args.decimation), pad=args.pad, device=device)
    af = op.ambiguity(x)
    print(f"pulse {args.pulse_index}: N = {n}, {len(op.orders)} FrFT orders, pad {args.pad}, m = {af.numel()}")

    x0 = spectral_init(op, af, args.keep_fraction, args.init_iters, generator=generator)
    init = align(x.cpu().numpy(), x0.cpu().numpy())
    print(f"init error           {init.relative_error:.3e}")

    start = time.perf_counter()
    result = wavemax(
        op, af, x0,
        outer_iters=args.outer_iters, max_iter=args.max_iter, tol=args.tol,
        mu_decay=args.mu_decay, slack=args.slack, refine_iters=args.refine_iters,
        verbose=max(args.verbose - 1, 0), generator=generator,
    )
    if args.verbose:
        for r, info in enumerate(result.rounds, start=1):
            print(f"  round {r}: {info['iterations']} iterations, final residual {info['residuals'][-1]:.3e}")
    xs = x.cpu().numpy()
    convex = align(xs, result.x_convex.cpu().numpy())
    final = align(xs, result.x.cpu().numpy())
    strict = align(xs, result.x.cpu().numpy(), continuous_shift=False)
    print(f"mu = {result.mu:.3e}, solve time {time.perf_counter() - start:.1f}s")
    print(f"convex-stage error   {convex.relative_error:.3e}")
    if result.refine_history:
        print(f"refined error        {final.relative_error:.3e}  (misfit {result.refine_history[-1]:.1e}, "
              f"time shift {final.a:.4f} samples)")
    print(f"Eq. (35) integer-shift error {strict.relative_error:.3e}")

    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    plot_comparison(init.z, x0.cpu().numpy(), "initial point", args.out / "initial_point.png")
    plot_comparison(final.z, result.x.cpu().numpy(), "reconstruction", args.out / "reconstruction.png")
    plot_convergence(result.rounds, args.out / "convergence.png")
    print(f"figures saved to {args.out}/")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
