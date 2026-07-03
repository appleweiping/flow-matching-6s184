"""Lab 1 — verify the numerical simulators and SDEs against analytic facts."""

import math

import torch

from flow_matching_labs.core import EulerSimulator, EulerMaruyamaSimulator
from flow_matching_labs.lab1 import BrownianMotion, OUProcess, LangevinSDE
from flow_matching_labs.distributions import Gaussian, GaussianMixture

torch.manual_seed(0)


class _ConstantODE:
    """dX/dt = c."""

    def __init__(self, c):
        self.c = c

    def drift_coefficient(self, xt, t, **kw):
        return self.c * torch.ones_like(xt)


def test_euler_integrates_constant_ode_exactly():
    # X(0)=0, dX/dt=2, integrate to t=1  => X(1) = 2.
    sim = EulerSimulator(_ConstantODE(2.0))
    x0 = torch.zeros(16, 1)
    ts = torch.linspace(0, 1, 101)
    xf = sim.simulate(x0, ts)
    assert torch.allclose(xf, 2.0 * torch.ones_like(xf), atol=1e-5)


def test_brownian_motion_variance_grows_linearly():
    # Var[X_t] = sigma^2 * t for standard BM started at 0.
    sigma, T = 1.0, 5.0
    sim = EulerMaruyamaSimulator(BrownianMotion(sigma))
    x0 = torch.zeros(20000, 1)
    ts = torch.linspace(0, T, 500)
    xf = sim.simulate(x0, ts)
    assert abs(xf.var().item() - sigma ** 2 * T) < 0.25
    assert abs(xf.mean().item()) < 0.1


def test_ou_process_reaches_stationary_variance():
    # Stationary Var = sigma^2 / (2 theta).
    theta, sigma = 0.25, 1.0
    sim = EulerMaruyamaSimulator(OUProcess(theta, sigma))
    x0 = torch.randn(20000, 1) * 8
    ts = torch.linspace(0, 60, 6000)
    xf = sim.simulate(x0, ts)
    expected = sigma ** 2 / (2 * theta)
    assert abs(xf.var().item() - expected) < 0.2


def test_langevin_moves_mass_toward_high_density():
    # Langevin dynamics targeting a GMM should raise the average log-density.
    target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3)
    sde = LangevinSDE(sigma=0.6, density=target)
    sim = EulerMaruyamaSimulator(sde)
    src = Gaussian(torch.zeros(2), 20 * torch.eye(2))
    x0 = src.sample(3000)
    ts = torch.linspace(0, 5.0, 1000)
    xf = sim.simulate(x0, ts)
    ld_before = target.log_density(x0).mean().item()
    ld_after = target.log_density(xf).mean().item()
    assert ld_after > ld_before + 5.0  # substantial improvement


def test_density_score_matches_analytic_gaussian():
    # For N(mu, sigma^2 I): score(x) = -(x - mu)/sigma^2.
    g = Gaussian(torch.tensor([1.0, -2.0]), torch.eye(2) * 4.0)
    x = torch.randn(64, 2)
    analytic = -(x - g.mean) / 4.0
    assert torch.allclose(g.score(x), analytic, atol=1e-4)
