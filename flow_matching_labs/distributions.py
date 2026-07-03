"""Sampleable / Density distributions used throughout the labs.

Implements the shared distribution zoo:
    * ``Gaussian`` — multivariate Gaussian (Sampleable + Density) with an
      ``isotropic`` constructor.
    * ``GaussianMixture`` — 2-D GMM with ``random_2D`` / ``symmetric_2D`` factories.
    * ``MoonsSampleable`` / ``CirclesSampleable`` / ``CheckerboardSampleable`` —
      the toy 2-D targets of Lab 2, Part 4.
    * ``IsotropicGaussian`` / ``GMM`` — the shape-agnostic image-space versions
      used in Lab 3.
    * ``score`` — a generic autodiff score ``∇_x log p(x)`` for any ``Density``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from torch.func import jacrev, vmap


# --------------------------------------------------------------------------- #
#  Abstract interfaces                                                         #
# --------------------------------------------------------------------------- #
class Sampleable(ABC):
    """A distribution that can be sampled from."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        ...


class Density(ABC):
    """A distribution with a tractable (log-)density and autodiff score."""

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Log density at ``x``; returns shape ``(batch_size, 1)``."""

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score ``∇_x log p(x)`` via autodiff (Lab 1, Part 3).

        Returns shape ``(batch_size, dim)``.
        """
        x = x.unsqueeze(1)  # (b, 1, dim)
        s = vmap(jacrev(self.log_density))(x)  # (b, 1, 1, 1, dim)
        return s.squeeze((1, 2, 3))


# --------------------------------------------------------------------------- #
#  Concrete 2-D distributions (Labs 1 & 2)                                     #
# --------------------------------------------------------------------------- #
class Gaussian(nn.Module, Sampleable, Density):
    """Multivariate Gaussian; both Sampleable and Density."""

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        return cls(torch.zeros(dim), torch.eye(dim) * std ** 2)


class GaussianMixture(nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian-mixture model."""

    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs, validate_args=False
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(cls, nmodes: int, std: float, scale: float = 10.0, seed: int = 0) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, std: float, scale: float = 10.0) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)


# --------------------------------------------------------------------------- #
#  Toy 2-D targets for arbitrary flow matching (Lab 2, Part 4)                 #
# --------------------------------------------------------------------------- #
class MoonsSampleable(Sampleable):
    """The two-moons distribution (``sklearn.datasets.make_moons``)."""

    def __init__(self, device: torch.device, noise: float = 0.05, scale: float = 5.0,
                 offset: Optional[torch.Tensor] = None):
        from sklearn.datasets import make_moons  # local import: heavy dep
        self._make_moons = make_moons
        self.noise, self.scale, self.device = noise, scale, device
        self.offset = (torch.zeros(2) if offset is None else offset).to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        s, _ = self._make_moons(n_samples=num_samples, noise=self.noise, random_state=None)
        return self.scale * torch.from_numpy(s.astype(np.float32)).to(self.device) + self.offset


class CirclesSampleable(Sampleable):
    """Concentric-circles distribution (``sklearn.datasets.make_circles``)."""

    def __init__(self, device: torch.device, noise: float = 0.05, scale: float = 5.0,
                 offset: Optional[torch.Tensor] = None):
        from sklearn.datasets import make_circles
        self._make_circles = make_circles
        self.noise, self.scale, self.device = noise, scale, device
        self.offset = (torch.zeros(2) if offset is None else offset).to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        s, _ = self._make_circles(n_samples=num_samples, noise=self.noise, factor=0.5, random_state=None)
        return self.scale * torch.from_numpy(s.astype(np.float32)).to(self.device) + self.offset


class CheckerboardSampleable(Sampleable):
    """Rejection-sampled checkerboard distribution."""

    def __init__(self, device: torch.device, grid_size: int = 3, scale: float = 5.0):
        self.grid_size, self.scale, self.device = grid_size, scale, device

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(self.device)
        while samples.shape[0] < num_samples:
            new = (torch.rand(num_samples, 2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new[:, 0] + self.scale) / grid_length) % 2 == 0
            y_mask = torch.floor((new[:, 1] + self.scale) / grid_length) % 2 == 0
            accept = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new[accept]], dim=0)
        return samples[:num_samples]


# --------------------------------------------------------------------------- #
#  Shape-agnostic distributions for image space (Lab 3)                        #
# --------------------------------------------------------------------------- #
class LabeledSampleable(ABC):
    """A distribution that yields ``(samples, labels)``."""

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...


class IsotropicGaussian(nn.Module, Sampleable):
    """Standard-normal source over an arbitrary tensor shape (Lab 3)."""

    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = list(shape)
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)


class GMM(nn.Module, LabeledSampleable):
    """Labelled Gaussian mixture over a flat ``(dim,)`` space (Lab 3 sanity check)."""

    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
        super().__init__()
        self.means = nn.Buffer(means)
        self.covariances = nn.Buffer(covariances)
        self.weights = nn.Buffer(weights)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.multinomial(
            self.weights.cpu(), num_samples=num_samples, replacement=True
        ).to(self.means.device)
        samples = torch.zeros(num_samples, self.means.shape[1]).to(self.means.device)
        for idx in range(len(self.means)):
            mask = labels == idx
            samples[mask] = torch.randn_like(samples[mask]) * self.covariances[idx] + self.means[idx]
        return samples, labels


__all__ = [
    "Sampleable",
    "Density",
    "Gaussian",
    "GaussianMixture",
    "MoonsSampleable",
    "CirclesSampleable",
    "CheckerboardSampleable",
    "LabeledSampleable",
    "IsotropicGaussian",
    "GMM",
]
