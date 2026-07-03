"""Classifier-free guidance (CFG) — Lab 3, Part 2.

Implements:
    * ``ConditionalVectorField`` — the abstract ``u_t^θ(x | y)`` interface shared
      by the MLP sanity-check model and the DiT (:mod:`flow_matching_labs.dit`).
    * Q2.1 ``CFGVectorFieldODE`` — the guided ODE that mixes the conditional and
      unconditional (null-label) vector fields:
      ``ũ = (1-w) u(x,t,∅) + w u(x,t,y)``.
    * Q2.2 ``CFGTrainer`` — flow-matching trainer that randomly drops the label
      to the null class with probability ``η`` so a single network learns both
      the conditional and unconditional fields.
    * Q2.3 ``MLPConditionalVectorField`` — a small MLP used for the 2-D GMM
      sanity check before scaling up to the DiT.
    * ``Trainer`` — the Lab-3 trainer base class (AdamW + LR warm-up +
      checkpoint hook), shared by CFG / VAE / latent trainers.
"""

from __future__ import annotations

import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from tqdm import tqdm

from .core import ODE


# --------------------------------------------------------------------------- #
#  Abstract conditional vector field                                           #
# --------------------------------------------------------------------------- #
class ConditionalVectorField(nn.Module, ABC):
    """A conditional vector field ``u_t^θ(x | y)``."""

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...


# --------------------------------------------------------------------------- #
#  Q2.1 — CFG ODE                                                              #
# --------------------------------------------------------------------------- #
class CFGVectorFieldODE(ODE):
    """Classifier-free-guided ODE (Lab 3, Q2.1)."""

    def __init__(self, net: ConditionalVectorField, null_label: int, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        guided = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * self.null_label
        unguided = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided + self.guidance_scale * guided


# --------------------------------------------------------------------------- #
#  Lab-3 trainer base class (AdamW + LR warm-up)                               #
# --------------------------------------------------------------------------- #
MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    size = 0
    for p in model.parameters():
        size += p.nelement() * p.element_size()
    for b in model.buffers():
        size += b.nelement() * b.element_size()
    return size


class Trainer(ABC):
    """Lab-3 trainer: AdamW, linear LR warm-up, optional periodic checkpoint."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.output_dir: Optional[str] = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        ...

    def checkpoint(self, step: int):  # overridden by subclasses if desired
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    @staticmethod
    def random_name() -> str:
        adjectives = ["autumn", "hidden", "bitter", "misty", "silent", "empty", "dry",
                      "dark", "summer", "icy", "delicate", "quiet", "white", "cool",
                      "spring", "winter", "patient"]
        foods = ["apple", "banana", "pear", "plum", "orange", "persimmon", "tangerine",
                 "durian", "jackfruit", "jicama", "cantaloupe", "watermelon", "peach"]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    def train(
        self,
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        ckpt_every: Optional[int] = None,
        run_name: Optional[str] = None,
        out_root: str = "runs",
        progress: bool = True,
        **kwargs,
    ) -> Tuple[List[float], List[int]]:
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join(out_root, run_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = model
        self.opt = self.get_optimizer(lr)
        self.model.train()
        for pg in self.opt.param_groups:
            pg["lr"] = 0.0

        losses: List[float] = []
        it = tqdm(range(num_steps)) if progress else range(num_steps)
        for step in it:
            cur_lr = lr * float(step + 1) / float(warmup_steps) if (warmup_steps > 0 and step < warmup_steps) else lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr

            self.opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            self.opt.step()

            losses.append(float(loss.detach().item()))
            if progress:
                it.set_description(f"step {step} lr={cur_lr:.1e} loss={loss.item():.4f}")

            if ckpt_every is not None and step % ckpt_every == 0 and step > 0:
                self.model.eval()
                self.checkpoint(step)
                self.model.train()

        self.model.eval()
        return losses, list(range(num_steps))


# --------------------------------------------------------------------------- #
#  Q2.2 — CFG trainer                                                          #
# --------------------------------------------------------------------------- #
class CFGTrainer(Trainer):
    """Classifier-free-guidance flow-matching trainer (Lab 3, Q2.2)."""

    def __init__(self, path, eta: float, null_label: int, eps: float = 0.001, **kwargs):
        assert 0 < eta < 1
        super().__init__(**kwargs)
        self.path = path
        self.eta = eta
        self.eps = eps
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, y = self.path.p_data.sample(batch_size)
        # Drop the label to the null class with probability eta.
        xi = torch.rand(y.shape[0]).to(y.device)
        y = y.clone()
        y[xi < self.eta] = self.null_label
        t = torch.rand(batch_size).to(z) * (1 - self.eps)
        x = self.path.sample_conditional_path(z, t)
        ut_theta = self.model(x, t, y)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        return torch.square(ut_theta - ut_ref).mean()


# --------------------------------------------------------------------------- #
#  Q2.3 — MLP conditional vector field (2-D sanity check)                      #
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: Type[nn.Module] = nn.SiLU, final_init: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        if final_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPConditionalVectorField(ConditionalVectorField):
    """MLP conditional vector field over flat data (Lab 3, Q2.3)."""

    def __init__(self, dim: int, hidden_dim: int, class_dim: int, num_classes: int):
        super().__init__()
        self.mlp = MLP([dim + class_dim + 1, hidden_dim, hidden_dim, dim])
        self.class_embedding = nn.Embedding(num_classes + 1, class_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xyt = torch.cat([x, self.class_embedding(y), t.unsqueeze(-1)], dim=-1)
        return self.mlp(xyt)


__all__ = [
    "ConditionalVectorField",
    "CFGVectorFieldODE",
    "Trainer",
    "CFGTrainer",
    "MLP",
    "MLPConditionalVectorField",
    "model_size_b",
    "MiB",
]
