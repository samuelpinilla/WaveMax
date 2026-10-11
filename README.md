# WaveMax

**Recover a radar waveform from its ambiguity function by convex optimization.**

WaveMax implements Pinilla et al., *WaveMax: Radar Waveform Design via Convex
Maximization* (IEEE Transactions on Signal Processing, vol. 73, 2025). Given only
the fractional-Fourier-transform (FrFT) based ambiguity function (AF) of a pulse,
which records how its energy spreads over delay and Doppler, WaveMax reconstructs
the pulse itself, magnitude *and* phase, up to the AF's unavoidable trivial
ambiguities (global phase, time shift, Doppler shift, time reversal).

The AF is quadratic in the pulse and discards all phase information, so this is a
hard phase-retrieval problem. WaveMax *lifts* the unknown pulse `x` to the
rank-one matrix `X = x xᴴ`, where the measurements become linear, and solves a
convex program anchored at a spectral estimate. A final local polish then fits
the data to the precision the discrete model allows.

All numerical building blocks (operators, penalties and solvers) come from
[invexapi](https://github.com/samuelpinilla/invexapi). This repository is a
complete worked example of that library.

## How it works

Write `F_α` for the FrFT of order α and `A[α, k]` for the measured AF. Every AF
sample is a linear function of the lifted matrix:

```
A[α, k] = |Tr(B_{α,k} X)|²,     B_{α,k} = F_αᴴ diag(DFT row k) F_α,     X = x xᴴ.
```

WaveMax runs in three stages:

1. **Spectral initialization** (paper Algorithm 1). The lag-domain data
   `Y[α, l] = IDFT_k A[α, k]` reveal where the pulse's FrFT energy concentrates.
   A power iteration on the matrix built from the strongest entries gives an
   anchor `x₀`, which is scaled to the energy estimated from the AF (Eq. 21).
2. **Convex lifted recovery** (paper Algorithm 2, problem 14). Find the PSD
   matrix `Z` most aligned with the anchor `X₀ = x₀x₀ᴴ` whose measurements match
   the data, solved by FASTA (adaptive forward-backward splitting):

   ```
   min_Z  ½‖ |B(Z)| + s − √A ‖²  −  μ·Re⟨X₀, Z⟩  +  μ·tr(Z)  +  𝟙{Z ⪰ 0}
   ```

   Three rounds refresh the slack `s = max(√A − |B(Z)|, 0)` and shrink μ
   tenfold. The estimate is `√λ₁·v₁`, the leading eigenpair of `Z`.
3. **Local polish.** Levenberg–Marquardt on the intensity misfit `A(x) − A`
   in the signal domain. The convex stage only has to land within about 30%
   of the truth; the polish then converges rapidly.

### Building blocks

| Role | Object |
|---|---|
| Fast fractional Fourier transform | `invexapi.operators.frft`, `FractionalFourier` |
| Lifted AF measurement `Z ↦ Tr(B_{α,k} Z)` (linear) | `invexapi.operators.LiftedFrFTAmbiguity` |
| Signal-domain AF `x ↦ A[α, k]` (nonlinear) | `invexapi.operators.FrFTAmbiguity` |
| Data fidelity with slack | `invexapi.SlackAmplitudeLeastSquares` |
| Anchor alignment, trace and PSD constraint | `invexapi.LinearTilt(PSDTracePenalty(μ), μ·X₀)` |
| Convex solver | `invexapi.optim.FASTA` |
| Local polish | `invexapi.optim.LevenbergMarquardt` |
| Initialization, pipeline, polish | `wavemax.spectral_init`, `wavemax.wavemax`, `wavemax.refine` |
| Error up to trivial ambiguities | `wavemax.align` |

The lifted operator is matrix-free: it never forms the N·K matrices `B_{α,k}`
(about 1.4 GB at N = 128), only the K FrFT matrices.

## Install and run

```bash
pip install -e ../invexapi      # or wherever invexapi lives
pip install -e .
python wavemax/recover_pulse.py                                 # pulse 3, full pipeline
python wavemax/recover_pulse.py --pulse-index 10 --show         # another pulse, show figures
python wavemax/recover_pulse.py --mu-decay 1 --refine-iters 0   # convex stage alone, fixed mu
python wavemax/recover_pulse.py --pad 2                         # more accurate FrFT model
pytest
```

The driver prints the relative error of the initialization, the convex stage and
the polished estimate, and saves intensity/phase and convergence figures to
`runs/recover_pulse/`. Main options:

| Option | Default | Meaning |
|---|---|---|
| `--pulse-index` | 2 | 0-based row of `pulse_set` |
| `--decimation` | 3 | keep every L-th FrFT order: `ceil(N/L)` orders in [0.5, 1.5] |
| `--pad` | 1 | zero-pad factor of the FrFT grid (see *Accuracy*) |
| `--outer-iters`, `--max-iter` | 3, 200 | convex rounds × FASTA iterations per round |
| `--mu-decay` | 0.1 | μ continuation factor per round (1 = fixed μ) |
| `--slack` | `refresh` | refresh the slack between rounds, or keep it at zero (`none`) |
| `--refine-iters` | 150 | Levenberg–Marquardt iterations (0 = no polish) |

FASTA prints invexapi's "no convex/invex certificate" warnings. These are
deliberate: the amplitude-fit term is nonconvex and the tilted penalty carries no
proof of convexity, so invexapi declines to promise a global optimum.

### From Python

```python
import torch
from invexapi.operators import LiftedFrFTAmbiguity, frft_orders
from wavemax import align, load_pulse, spectral_init, wavemax

x = load_pulse(2)                                     # ground truth, for scoring only
op = LiftedFrFTAmbiguity(len(x), frft_orders(len(x), 3))
af = op.ambiguity(x)                                  # the measurements

x0 = spectral_init(op, af, generator=torch.Generator().manual_seed(0))
result = wavemax(op, af, x0)
print(align(x.numpy(), result.x.numpy()).relative_error)
```

## Design notes

- **Centred lags in the initialization.** `Y[α, ·]` is the autocorrelation of
  the FrFT intensity, so its mass sits at small lags around 0. Moving lag 0 to the
  centre of the FrFT grid before selecting entries yields an anchor concentrated
  at time 0 and frequency 0, which is itself a trivial ambiguity of the pulse.
  Across the 12 benchmark pulses this lowers the anchor error from 0.64–1.02 to
  0.49–0.69, which is what lets the convex stage reach the polish basin.
- **Exact PSD projection.** The gradient step leaves `Z` non-Hermitian. The prox
  projects its Hermitian part with `eigh` and soft-thresholds the eigenvalues by
  μτ (Algorithm 2, lines 10–13), which is the exact Frobenius projection.
- **μ continuation.** With a fixed μ the penalized minimizer sits O(μ) away from
  the true matrix (about 2e-3 even when started at the truth). Shrinking μ
  tenfold per round removes that bias.
- **Correctly scaled estimate.** `√λ₁·v₁` preserves the pulse energy. The error
  measure of Eq. 11 is scale-sensitive, so a unit-norm eigenvector would be
  penalized for its norm alone.
- **Polish on intensities, not magnitudes.** `|z|` is not differentiable at zero
  and many AF samples are near zero. The intensity residual is a polynomial in x,
  which keeps Gauss–Newton steps well behaved, and each step is solved by QR,
  never by the normal equations.

## Measuring error

`wavemax.align` computes the distance of Eq. (11) by minimizing over the trivial
ambiguities `z = e^{iβ} e^{ibn} x[εn − a]`: the global phase β in closed form,
the Doppler b on an oversampled FFT grid followed by refinement, and the
reflection ε and shift a exhaustively. Every AF row is a Fourier magnitude of an
FrFT intensity, which no shift can change, so in the continuous model *any*
real delay is an ambiguity. The shift is therefore continuous by default
(`a ∈ ℝ`, a band-limited phase ramp). `align(..., continuous_shift=False)`
restricts it to integers, exactly as written in Eq. (35).

## Accuracy

The discrete fast FrFT reproduces the continuous model's shift invariance only
approximately. Delaying a pulse by one sample changes its computed AF by about
1e-7 (relative). This leaves a shallow valley of near-solutions that solvers
cannot traverse in double precision, and it bounds the reachable error at about
1e-5.

Zero-padding the signal before each FrFT (`pad=2`) keeps its time–frequency
support well inside the fast algorithm's accurate region. The shift then
changes the AF by only about 1e-10, and the error floor near the truth drops to
about 5e-8. Padding also changes the transform's time scaling, so it is a
different (more accurate) measurement model, not a resampling. On that model the
convex stage currently reaches the polish basin less often (6/12 pulses), so
`pad=1` stays the default.

## Layout

```
wavemax/
  recover_pulse.py       command-line driver
  data/pulse_set.mat     benchmark pulses (100 x 128, complex)
  src/wavemax/
    initialization.py    spectral initialization (Algorithm 1)
    solver.py            convex lifted stage (Algorithm 2) and polish
    distance.py          Eq. (11) distance up to trivial ambiguities
    data.py, plotting.py
tests/test_wavemax.py
```

## Citation

```bibtex
@article{pinilla2025wavemax,
  author  = {Pinilla, Samuel and others},
  title   = {{WaveMax}: Radar Waveform Design via Convex Maximization},
  journal = {IEEE Transactions on Signal Processing},
  volume  = {73},
  year    = {2025}
}
```
