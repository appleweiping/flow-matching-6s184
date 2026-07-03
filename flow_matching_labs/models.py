"""Learned vector-field / score networks and their trainers — Lab 2.

Implements:
    * ``build_mlp`` / ``MLPVectorField`` / ``MLPScore`` — simple MLP
      parameterisations ``u_t^θ(x)`` and ``s_t^θ(x)``.
    * Q3.1 ``ConditionalFlowMatchingTrainer`` — regress the network onto the
      conditional vector field (the flow-matching objective).
    * Q3.2 ``ConditionalScoreMatchingTrainer`` — regress onto the conditional
      score (the denoising score-matching objective).
    * Q3.3 ``ScoreFromVectorField`` — recover the marginal score analytically
      from a trained *flow* network (Gaussian paths only).
    * ``LearnedVectorFieldODE`` / ``LangevinFlowSDE`` — wrap a trained network as
      an ODE (deterministic sampling) or SDE (Langevin-corrected sampling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Type

import torch
from tqdm import tqdm

from .core import ODE, SDE
from .paths import Alpha, Beta, ConditionalProbabilityPath


# --------------------------------------------------------------------------- #
#  Networks                                                                    #
# --------------------------------------------------------------------------- #
def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return torch.nn.Sequential(*layers)


class MLPVectorField(torch.nn.Module):
    """MLP parameterisation of the learned vector field ``u_t^θ(x)``."""

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + list(hiddens) + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class MLPScore(torch.nn.Module):
    """MLP parameterisation of the learned score field ``s_t^θ(x)``."""

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + list(hiddens) + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


# --------------------------------------------------------------------------- #
#  Trainers                                                                    #
# --------------------------------------------------------------------------- #
class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        ...

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3,
              progress: bool = True, **kwargs) -> List[float]:
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        losses: List[float] = []
        it = tqdm(range(num_epochs)) if progress else range(num_epochs)
        for _ in it:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if progress:
                it.set_description(f"loss: {loss.item():.4f}")
        self.model.eval()
        return losses


class ConditionalFlowMatchingTrainer(Trainer):
    """Flow-matching objective (Lab 2, Q3.1).

    ``L(θ) = E_{z,t,x}  ‖u_t^θ(x) − u_t(x|z)‖²`` with ``z ~ p_data``,
    ``t ~ U[0,1]``, ``x ~ p_t(·|z)``.
    """

    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField):
        super().__init__(model)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional_path(z, t)
        ut_theta = self.model(x, t)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        return torch.mean(torch.sum((ut_theta - ut_ref) ** 2, dim=-1))


class ConditionalScoreMatchingTrainer(Trainer):
    """Denoising score-matching objective (Lab 2, Q3.2).

    ``L(θ) = E_{z,t,x}  ‖s_t^θ(x) − ∇log p_t(x|z)‖²``.
    """

    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore):
        super().__init__(model)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional_path(z, t)
        s_theta = self.model(x, t)
        s_ref = self.path.conditional_score(x, z, t)
        return torch.mean(torch.sum((s_theta - s_ref) ** 2, dim=-1))


# --------------------------------------------------------------------------- #
#  Deriving the marginal score from a learned vector field (Lab 2, Q3.3)       #
# --------------------------------------------------------------------------- #
class ScoreFromVectorField(torch.nn.Module):
    """Compute the marginal score from a trained *flow* network, for a Gaussian
    conditional probability path.

    Uses the identity (valid for ``p_t(x|z) = N(α_t z, β_t² I)``):

        ``s_t(x) = (α_t u_t(x) − α̇_t x) / (β_t² α̇_t − α_t β̇_t β_t)``.
    """

    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        b = self.beta(t)
        da = self.alpha.dt(t)
        db = self.beta.dt(t)
        num = a * self.vector_field(x, t) - da * x
        den = b ** 2 * da - a * db * b
        return num / den


# --------------------------------------------------------------------------- #
#  Wrapping trained networks as dynamics                                       #
# --------------------------------------------------------------------------- #
class LearnedVectorFieldODE(ODE):
    """Wrap a trained vector-field network as a (marginal) ODE."""

    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x, t)


class LangevinFlowSDE(SDE):
    """Langevin-corrected sampling SDE combining a flow and a score network.

    ``dX = [u_t^θ(x) + ½σ² s_t^θ(x)] dt + σ dW``  (Lab 2, Part 3).
    """

    def __init__(self, flow_model: MLPVectorField, score_model, sigma: float):
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.flow_model(x, t) + 0.5 * self.sigma ** 2 * self.score_model(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sigma * torch.ones_like(x)


# --------------------------------------------------------------------------- #
#  Conditional ODE / SDE for a *known* path (Lab 2, Q2.3–2.4)                  #
# --------------------------------------------------------------------------- #
class ConditionalVectorFieldODE(ODE):
    """The (ground-truth) conditional ODE ``u_t(x|z)`` for a fixed ``z``."""

    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        z = self.z.expand(x.shape[0], *self.z.shape[1:])
        return self.path.conditional_vector_field(x, z, t)


class ConditionalVectorFieldSDE(SDE):
    """The conditional SDE ``u_t(x|z) + ½σ² s_t(x|z)`` for a fixed ``z``."""

    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        z = self.z.expand(x.shape[0], *self.z.shape[1:])
        return self.path.conditional_vector_field(x, z, t) + \
            0.5 * self.sigma ** 2 * self.path.conditional_score(x, z, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sigma * torch.ones_like(x)


__all__ = [
    "build_mlp",
    "MLPVectorField",
    "MLPScore",
    "Trainer",
    "ConditionalFlowMatchingTrainer",
    "ConditionalScoreMatchingTrainer",
    "ScoreFromVectorField",
    "LearnedVectorFieldODE",
    "LangevinFlowSDE",
    "ConditionalVectorFieldODE",
    "ConditionalVectorFieldSDE",
]
