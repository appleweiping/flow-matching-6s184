"""Lab 1 — Simulating ODEs and SDEs.

Implements the SDEs of Lab 1:
    * Q2.1 ``BrownianMotion``  — ``dX = σ dW``.
    * Q2.2 ``OUProcess``       — ``dX = -θ X dt + σ dW`` (Ornstein–Uhlenbeck).
    * Q3.1 ``LangevinSDE``     — ``dX = ½σ² ∇log p(X) dt + σ dW`` (Langevin dynamics).

The numerical simulators used to integrate these live in
:mod:`flow_matching_labs.core` (``EulerSimulator`` / ``EulerMaruyamaSimulator``).
"""

from __future__ import annotations

import torch

from .core import SDE
from .distributions import Density


class BrownianMotion(SDE):
    """Scaled Brownian motion ``dX_t = σ dW_t``  (Lab 1, Q2.1)."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(xt)

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sigma * torch.ones_like(xt)


class OUProcess(SDE):
    """Ornstein–Uhlenbeck process ``dX_t = -θ X_t dt + σ dW_t``  (Lab 1, Q2.2).

    Its stationary distribution is ``N(0, σ² / (2θ))``.
    """

    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return -self.theta * xt

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sigma * torch.ones_like(xt)


class LangevinSDE(SDE):
    """Langevin dynamics for a target ``Density``  (Lab 1, Q3.1).

    ``dX_t = ½ σ² ∇_x log p(X_t) dt + σ dW_t``. Its stationary distribution is
    exactly ``p`` regardless of ``σ`` — this is what lets us *sample* from any
    density whose score we can evaluate.
    """

    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return 0.5 * self.sigma ** 2 * self.density.score(xt)

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sigma * torch.ones_like(xt)


__all__ = ["BrownianMotion", "OUProcess", "LangevinSDE"]
