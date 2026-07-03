"""Core ODE / SDE abstractions and numerical simulators.

This module unifies the abstractions that appear (in slightly different forms)
across all three labs.  We adopt the *batched-time* convention of Labs 2 & 3,
in which ``t`` and the step ``h`` are per-sample tensors, and support both the
scalar-time Lab-1 usage and the ``**kwargs`` conditioning of Lab 3.

Implements:
    * Lab 1, Q1.1 — ``EulerSimulator`` (ODE) and ``EulerMaruyamaSimulator`` (SDE).
    * Lab 2/3    — the same simulators generalised to per-sample timesteps and
      arbitrary tensor shapes ``(b, ...)`` with optional conditioning kwargs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Dynamics: ODEs and SDEs                                                     #
# --------------------------------------------------------------------------- #
class ODE(ABC):
    """An ordinary differential equation ``dX_t = drift(X_t, t) dt``."""

    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Drift coefficient. ``xt``: (b, ...), ``t``: (b,) or scalar. Returns (b, ...)."""


class SDE(ABC):
    """A stochastic differential equation ``dX_t = drift dt + diffusion dW_t``."""

    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Drift coefficient. ``xt``: (b, ...), ``t``: (b,) or scalar. Returns (b, ...)."""

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Diffusion coefficient. ``xt``: (b, ...), ``t``: (b,) or scalar. Returns (b, ...)."""


# --------------------------------------------------------------------------- #
#  Simulators                                                                  #
# --------------------------------------------------------------------------- #
def _reshape_step(h: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-sample step ``h`` (shape ``()`` or ``(b,)`` or ``(b,1)``)
    against a state tensor ``xt`` of shape ``(b, ...)``."""
    if h.dim() == 0:
        return h
    return h.view([-1] + [1] * (xt.dim() - 1))


class Simulator(ABC):
    """Discretised integrator for an :class:`ODE` or :class:`SDE`."""

    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Take one integration step from time ``t`` with width ``h``."""

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = False, **kwargs) -> torch.Tensor:
        """Integrate from ``ts[..., 0]`` to ``ts[..., -1]``.

        ``ts`` may be shape ``(nts,)`` (shared schedule) or ``(b, nts)`` /
        ``(b, nts, 1)`` (per-sample schedule).  Returns the final state.
        """
        for t, h in self._iter_steps(ts, use_tqdm):
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(
        self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = False, **kwargs
    ) -> torch.Tensor:
        """Like :meth:`simulate`, but return the full trajectory ``(b, nts, ...)``."""
        xs = [x.clone()]
        for t, h in self._iter_steps(ts, use_tqdm):
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

    @staticmethod
    def _iter_steps(ts: torch.Tensor, use_tqdm: bool):
        """Yield ``(t, h)`` pairs for each integration interval.

        Supports the shared 1-D schedule of Lab 1 as well as the per-sample
        schedule ``(b, nts)`` / ``(b, nts, 1)`` of Labs 2 & 3.
        """
        if ts.dim() == 1:  # shared schedule (Lab 1)
            n = ts.shape[0]
            it = tqdm(range(n - 1)) if use_tqdm else range(n - 1)
            for i in it:
                yield ts[i], ts[i + 1] - ts[i]
        else:  # per-sample schedule (Lab 2 / 3): (b, nts) or (b, nts, 1)
            n = ts.shape[1]
            it = tqdm(range(n - 1)) if use_tqdm else range(n - 1)
            for i in it:
                yield ts[:, i], ts[:, i + 1] - ts[:, i]


class EulerSimulator(Simulator):
    """Forward-Euler integrator for an ODE  (Lab 1, Q1.1)."""

    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        h_ = _reshape_step(h, xt)
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h_


class EulerMaruyamaSimulator(Simulator):
    """Euler–Maruyama integrator for an SDE  (Lab 1, Q1.1)."""

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        h_ = _reshape_step(h, xt)
        drift = self.sde.drift_coefficient(xt, t, **kwargs)
        diff = self.sde.diffusion_coefficient(xt, t, **kwargs)
        return xt + drift * h_ + diff * torch.sqrt(h_) * torch.randn_like(xt)


# --------------------------------------------------------------------------- #
#  Utilities                                                                   #
# --------------------------------------------------------------------------- #
def record_every(num_timesteps: int, every: int) -> torch.Tensor:
    """Indices to keep when sub-sampling a trajectory (always keeps the last)."""
    if every <= 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [torch.arange(0, num_timesteps - 1, every), torch.tensor([num_timesteps - 1])]
    )


def build_ts(num_timesteps: int, num_samples: int, device, t_end: float = 1.0) -> torch.Tensor:
    """Convenience: build a per-sample schedule ``(num_samples, num_timesteps, 1)``."""
    return (
        torch.linspace(0.0, t_end, num_timesteps)
        .view(1, -1, 1)
        .expand(num_samples, -1, 1)
        .to(device)
    )


__all__ = [
    "ODE",
    "SDE",
    "Simulator",
    "EulerSimulator",
    "EulerMaruyamaSimulator",
    "record_every",
    "build_ts",
]
