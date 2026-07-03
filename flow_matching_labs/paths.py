"""Conditional probability paths — Lab 2 (2-D) and Lab 3 (image space).

A *conditional probability path* ``p_t(x | z)`` interpolates from a simple source
``p_0 = p_simple`` to a data-conditioned target ``p_1(·|z) = δ_z``.  Flow /
score matching learns the *marginal* vector field / score of the mixture
``p_t(x) = ∫ p_t(x|z) p_data(z) dz`` by regressing against the (tractable)
conditional quantities.

Implements:
    * Q2.1 ``LinearAlpha`` / ``SquareRootBeta`` — the noise schedule with
      analytic and autodiff-checked derivatives.
    * Q2.2–2.4 ``GaussianConditionalProbabilityPath`` — Gaussian path
      ``p_t(x|z) = N(α_t z, β_t² I)`` with conditional vector field and score.
    * Q4.1 ``LinearConditionalProbabilityPath`` — the straight-line path
      ``X_t = (1-t) X_0 + t z`` bridging *arbitrary* source/target.

The time argument ``t`` may be shape ``(b, 1)`` (Lab 2) or ``(b,)`` (Lab 3);
scalars are broadcast against the data shape ``(b, ...)`` automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from .distributions import Gaussian, IsotropicGaussian, Sampleable


# --------------------------------------------------------------------------- #
#  Noise schedules  α_t, β_t                                                   #
# --------------------------------------------------------------------------- #
class Alpha(ABC):
    """Signal schedule with ``α_0 = 0``, ``α_1 = 1``."""

    def __init__(self):
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1))
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ...

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """``d/dt α_t`` via autodiff (overridable with an analytic form).

        Expects ``t`` of shape ``(b, 1)`` and returns ``(b, 1)``.
        """
        t = t.unsqueeze(1)
        d = vmap(jacrev(self))(t)
        return d.view(-1, 1)


class Beta(ABC):
    """Noise schedule with ``β_0 = 1``, ``β_1 = 0``."""

    def __init__(self):
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1))
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ...

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        d = vmap(jacrev(self))(t)
        return d.view(-1, 1)


class LinearAlpha(Alpha):
    """``α_t = t``  (Lab 2 Q2.1)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    """``β_t = sqrt(1 - t)``  (Lab 2 Q2.1)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)


class LinearBeta(Beta):
    """``β_t = 1 - t``  (Lab 3)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


# --------------------------------------------------------------------------- #
#  Abstract conditional probability path                                       #
# --------------------------------------------------------------------------- #
class ConditionalProbabilityPath(nn.Module, ABC):
    """Abstract base class for conditional probability paths."""

    def __init__(self, p_simple, p_data):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """Draw ``x ~ p_t(x) = ∫ p_t(x|z) p_data(z) dz``."""
        num_samples = t.shape[0]
        z = self.sample_conditioning_variable(num_samples)
        if isinstance(z, tuple):  # labelled data (Lab 3)
            z = z[0]
        return self.sample_conditional_path(z, t)

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        ...

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def conditional_vector_field(self, x, z, t) -> torch.Tensor:
        ...

    @abstractmethod
    def conditional_score(self, x, z, t) -> torch.Tensor:
        ...


def _match_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a time tensor of shape ``(b,)`` or ``(b, 1)`` so that it
    broadcasts against a data tensor ``x`` of shape ``(b, ...)``."""
    t = t.reshape(t.shape[0], *([1] * (x.dim() - 1)))
    return t


# --------------------------------------------------------------------------- #
#  Gaussian conditional probability path (Lab 2 Q2.2–2.4, Lab 3)              #
# --------------------------------------------------------------------------- #
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """``p_t(x|z) = N(α_t z, β_t² I)``.

    Works for both flat 2-D data (Lab 2) and image tensors (Lab 3); the time
    argument is broadcast to the data shape.  ``p_data`` may be ``None`` when the
    path is only used as a *latent* source (Lab 3, Part 5).
    """

    def __init__(self, p_data, alpha: Alpha, beta: Beta, p_simple_shape: List[int] | None = None):
        if p_simple_shape is not None:
            p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        else:
            p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = _match_time(self.alpha(t), z)
        b = _match_time(self.beta(t), z)
        return a * z + b * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = _match_time(self.alpha(t), x)
        b = _match_time(self.beta(t), x)
        da = _match_time(self.alpha.dt(t), x)
        db = _match_time(self.beta.dt(t), x)
        return (da - db / b * a) * z + db / b * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = _match_time(self.alpha(t), x)
        b = _match_time(self.beta(t), x)
        return (z * a - x) / b ** 2


# --------------------------------------------------------------------------- #
#  Linear conditional probability path (Lab 2 Q4.1)                            #
# --------------------------------------------------------------------------- #
class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    """Straight-line path ``X_t = (1-t) X_0 + t z``, ``X_0 ~ p_simple``.

    Bridges an *arbitrary* source ``p_simple`` to an *arbitrary* target
    ``p_data``.  Its conditional vector field is ``u_t(x|z) = (z - x)/(1-t)``.
    The conditional score is not tractable and raises.
    """

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.p_simple.sample(z.shape[0])
        t = _match_time(t, z)
        return (1 - t) * x0 + t * z

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _match_time(t, x)
        return (z - x) / (1 - t)

    def conditional_score(self, x, z, t) -> torch.Tensor:
        raise NotImplementedError(
            "The conditional score of a LinearConditionalProbabilityPath is not tractable."
        )


__all__ = [
    "Alpha",
    "Beta",
    "LinearAlpha",
    "SquareRootBeta",
    "LinearBeta",
    "ConditionalProbabilityPath",
    "GaussianConditionalProbabilityPath",
    "LinearConditionalProbabilityPath",
]
