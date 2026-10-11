import numpy as np
import pytest
import torch

from invexapi.operators import LiftedFrFTAmbiguity, frft_orders
from wavemax import align, estimate_norm, leading_component, load_pulses, refine, spectral_init, wavemax
from wavemax.distance import fractional_shift


def _pulse():
    return load_pulses()[2]


@pytest.mark.parametrize("eps,a,b,beta", [(1, 0, 0.0, 0.0), (1, 5, 0.3, 1.0), (-1, 17, 2.0, -2.0)])
def test_align_inverts_every_trivial_ambiguity(eps, a, b, beta):
    x = _pulse()
    n = np.arange(len(x))
    q = np.exp(1j * beta) * np.exp(1j * b * n) * x[(eps * n - a) % len(x)]
    for continuous in (True, False):
        result = align(x, q, continuous_shift=continuous)
        assert result.relative_error < 1e-6
        assert result.eps == eps and result.a == pytest.approx(a, abs=1e-6)
        assert np.allclose(result.z, q, atol=1e-6)


def test_align_factors_out_fractional_shifts_only_when_continuous():
    x = _pulse()
    q = np.exp(0.4j) * fractional_shift(x, 3.3)
    continuous = align(x, q)
    assert continuous.relative_error < 1e-6 and continuous.a == pytest.approx(3.3, abs=1e-6)
    assert align(x, q, continuous_shift=False).relative_error > 1e-2


def test_distance_is_scale_sensitive():
    x = _pulse()
    assert align(x, 0.5 * x).relative_error == pytest.approx(0.5)


@pytest.mark.parametrize("pad", [1, 2])
def test_norm_estimate_and_lifting_consistency(pad):
    x = torch.as_tensor(_pulse(), dtype=torch.complex128) * 3.0
    op = LiftedFrFTAmbiguity(len(x), frft_orders(len(x), 3), pad=pad)
    af = op.ambiguity(x)
    assert af.shape == (43, pad * len(x))
    # the fast FrFT is only approximately unitary, so lambda_0 ~ ||x|| loosely
    assert estimate_norm(af) == pytest.approx(3.0, rel=0.2)
    X = torch.outer(x, x.conj())
    assert torch.allclose(op.apply(X).abs() ** 2, af, rtol=1e-8, atol=1e-8)


def test_leading_component_of_rank_one_matrix():
    x = torch.as_tensor(_pulse(), dtype=torch.complex128) * 2.0
    v = leading_component(torch.outer(x, x.conj()))
    assert align(x.numpy(), v.numpy()).relative_error < 1e-8


def test_centred_lags_improve_the_spectral_init():
    x = torch.as_tensor(_pulse(), dtype=torch.complex128)
    op = LiftedFrFTAmbiguity(len(x), frft_orders(len(x), 3))
    af = op.ambiguity(x)
    errs = {
        c: align(x.numpy(), spectral_init(op, af, generator=torch.Generator().manual_seed(0),
                                          center_lags=c).numpy()).relative_error
        for c in (True, False)
    }
    assert errs[True] < errs[False]


def test_refine_converges_from_a_nearby_start():
    x = torch.as_tensor(_pulse(), dtype=torch.complex128)
    op = LiftedFrFTAmbiguity(len(x), frft_orders(len(x), 3), pad=2)
    g = torch.Generator().manual_seed(0)
    d = torch.randn(len(x), dtype=torch.complex128, generator=g)
    with pytest.warns(UserWarning):
        q, history = refine(op, op.ambiguity(x), x + 0.05 * d / d.norm(), max_iter=60)
    assert history[-1] < 1e-9
    assert align(x.numpy(), q.numpy()).relative_error < 1e-6


def test_wavemax_pipeline_improves_on_its_initialization():
    x = torch.as_tensor(_pulse(), dtype=torch.complex128)
    op = LiftedFrFTAmbiguity(len(x), frft_orders(len(x), 3))
    af = op.ambiguity(x)
    g = torch.Generator().manual_seed(0)
    x0 = spectral_init(op, af, generator=g)
    init_err = align(x.numpy(), x0.numpy()).relative_error
    with pytest.warns(UserWarning, match="certificate"):
        result = wavemax(op, af, x0, outer_iters=2, max_iter=60, refine_iters=20, generator=g)
    assert align(x.numpy(), result.x_convex.numpy()).relative_error < init_err
    assert align(x.numpy(), result.x.numpy()).relative_error <= align(
        x.numpy(), result.x_convex.numpy()).relative_error
    assert len(result.residuals) == sum(r["iterations"] for r in result.rounds)
    with pytest.raises(ValueError):
        wavemax(op, af, x0, slack="bogus")
