"""Lab 2 — verify conditional probability paths, schedules, and score identities."""

import torch

from flow_matching_labs.distributions import Gaussian, GaussianMixture, CheckerboardSampleable
from flow_matching_labs.paths import (
    LinearAlpha, SquareRootBeta, LinearBeta,
    GaussianConditionalProbabilityPath, LinearConditionalProbabilityPath,
)
from flow_matching_labs.models import (
    MLPVectorField, ConditionalFlowMatchingTrainer, ScoreFromVectorField,
)

torch.manual_seed(0)
DEV = "cpu"


def test_alpha_beta_endpoints_and_derivatives():
    a, b = LinearAlpha(), SquareRootBeta()
    assert torch.allclose(a(torch.zeros(1, 1)), torch.zeros(1, 1))
    assert torch.allclose(a(torch.ones(1, 1)), torch.ones(1, 1))
    assert torch.allclose(b(torch.zeros(1, 1)), torch.ones(1, 1))
    assert torch.allclose(b(torch.ones(1, 1)), torch.zeros(1, 1))
    # Analytic dt matches finite differences for alpha (linear) exactly.
    t = torch.rand(32, 1) * 0.9 + 0.05
    assert torch.allclose(a.dt(t), torch.ones_like(t))


def test_squareroot_beta_dt_matches_autodiff():
    b = SquareRootBeta()
    t = torch.rand(32, 1) * 0.8 + 0.05
    t.requires_grad_(True)
    y = b(t).sum()
    (grad,) = torch.autograd.grad(y, t)
    # Analytic dt uses a +1e-4 stabiliser; compare loosely.
    assert torch.allclose(b.dt(t.detach()), grad.detach(), atol=1e-2)


def test_gaussian_path_conditional_score_is_analytic():
    # For p_t(x|z) = N(alpha_t z, beta_t^2 I): score = (alpha_t z - x)/beta_t^2.
    path = GaussianConditionalProbabilityPath(
        p_data=GaussianMixture.symmetric_2D(5, 1.0, 10.0),
        alpha=LinearAlpha(), beta=SquareRootBeta(),
    )
    z = path.p_data.sample(64)
    t = torch.rand(64, 1) * 0.9 + 0.05
    x = path.sample_conditional_path(z, t)
    a, b = path.alpha(t), path.beta(t)
    expected = (a * z - x) / b ** 2
    assert torch.allclose(path.conditional_score(x, z, t), expected, atol=1e-5)


def test_gaussian_path_vector_field_transports_endpoints():
    # At t->0 the conditional path is near the Gaussian source; at t->1 it is z.
    path = GaussianConditionalProbabilityPath(
        p_data=GaussianMixture.symmetric_2D(5, 0.5, 8.0),
        alpha=LinearAlpha(), beta=SquareRootBeta(),
    )
    z = path.p_data.sample(1000)
    t1 = torch.full((1000, 1), 0.999)
    x1 = path.sample_conditional_path(z, t1)
    # Near t=1, samples concentrate on z.
    assert (x1 - z).abs().mean().item() < 0.2


def test_linear_path_vector_field_formula():
    # u_t(x|z) = (z - x)/(1 - t) and X_t = (1-t) X_0 + t z, so dX/dt = z - X_0
    # and (z - X_t)/(1-t) = z - X_0.  Check consistency directly.
    path = LinearConditionalProbabilityPath(
        p_simple=Gaussian.isotropic(2, 1.0),
        p_data=CheckerboardSampleable("cpu", grid_size=4),
    )
    z = path.p_data.sample(128)
    x0 = path.p_simple.sample(128)
    t = torch.rand(128, 1) * 0.8 + 0.1
    xt = (1 - t) * x0 + t * z
    vf = path.conditional_vector_field(xt, z, t)
    assert torch.allclose(vf, z - x0, atol=1e-4)


def test_score_from_vector_field_recovers_conditional_score_after_training():
    # After brief flow-matching training on a single Gaussian target, the
    # ScoreFromVectorField reconstruction should have the right shape and be
    # finite; exact match is only asymptotic, so we check a coarse property:
    # the recovered score points roughly toward the data mean at small t.
    target = Gaussian(torch.zeros(2), torch.eye(2) * 0.25)
    path = GaussianConditionalProbabilityPath(
        p_data=target, alpha=LinearAlpha(), beta=SquareRootBeta(),
    )
    fm = MLPVectorField(dim=2, hiddens=[64, 64])
    trainer = ConditionalFlowMatchingTrainer(path, fm)
    trainer.train(num_epochs=400, device=DEV, lr=1e-3, progress=False, batch_size=512)
    sfvf = ScoreFromVectorField(fm, path.alpha, path.beta)
    x = torch.randn(256, 2) * 2.0
    t = torch.full((256, 1), 0.5)
    s = sfvf(x, t)
    assert s.shape == (256, 2)
    assert torch.isfinite(s).all()
    # Score should on average point inward (negative dot with x) for a target at 0.
    inward = (-(s * x).sum(-1)).mean().item()
    assert inward > 0
