"""Lab 3 glue — MNIST data source and the pixel/latent CFG trainers.

Implements:
    * ``MNISTSampler`` — a ``LabeledSampleable`` wrapper over torchvision MNIST,
      resized to 32×32 and normalised (Lab 3, Part 1).
    * ``MNISTCFGTrainer`` — pixel-space classifier-free-guidance trainer with a
      checkpoint that saves a sample grid (Part 3).
    * ``LatentCFGTrainer`` — latent-space CFG trainer that encodes MNIST with a
      frozen VAE and trains a DiT in the ``128×4×4`` latent (Part 5).
    * ``visualize_output`` / ``sample_grid`` — sampling + figure helpers.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from .cfg import CFGTrainer, CFGVectorFieldODE, Trainer
from .core import EulerSimulator
from .distributions import LabeledSampleable
from .paths import GaussianConditionalProbabilityPath
from .vae import VAE


# --------------------------------------------------------------------------- #
#  MNIST data source                                                           #
# --------------------------------------------------------------------------- #
class MNISTSampler(nn.Module, LabeledSampleable):
    """Sampleable wrapper for MNIST (resized to 32×32, normalised)."""

    def __init__(self, root: str = "./data"):
        super().__init__()
        self.dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1305,), (0.2891,)),
            ]),
        )
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels


# --------------------------------------------------------------------------- #
#  Sampling / visualisation helpers                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_grid(
    model,
    path,
    device,
    samples_per_class: int = 10,
    num_timesteps: int = 100,
    guidance_scale: float = 3.0,
    null_label: int = 10,
    n_classes: int = 10,
    decode: Optional[VAE] = None,
) -> torch.Tensor:
    """Generate a ``n_classes × samples_per_class`` grid of digits via CFG.

    Returns a ``(N, 1, 32, 32)`` tensor of decoded/pixel samples in ``[0,1]``.
    """
    ode = CFGVectorFieldODE(model, guidance_scale=guidance_scale, null_label=null_label)
    simulator = EulerSimulator(ode)
    y = torch.arange(n_classes, dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
    num_samples = y.shape[0]
    x0 = path.p_simple.sample(num_samples)
    ndim = x0.dim()  # e.g. 4 for (b,c,h,w)
    ts = torch.linspace(0, 0.999, num_timesteps).view([1, -1] + [1] * (ndim - 1))
    ts = ts.expand([num_samples, -1] + [1] * (ndim - 1)).to(device)
    x1 = simulator.simulate(x0, ts, y=y, use_tqdm=False)
    if decode is not None:
        x1, _ = decode.decode(x1)
    v_min, v_max = x1.min(), x1.max()
    return (x1 - v_min) / (v_max - v_min + 1e-8)


@torch.no_grad()
def visualize_output(model, path, device, save_path: str, guidance_scales: List[float] = (1.0, 3.0, 5.0),
                     samples_per_class: int = 10, num_timesteps: int = 100, null_label: int = 10,
                     n_classes: int = 10, decode: Optional[VAE] = None):
    """Save a multi-panel figure of CFG samples for several guidance strengths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(6 * len(guidance_scales), 6))
    if len(guidance_scales) == 1:
        axes = [axes]
    for ax, w in zip(axes, guidance_scales):
        grid_imgs = sample_grid(model, path, device, samples_per_class, num_timesteps,
                                guidance_scale=w, null_label=null_label, n_classes=n_classes, decode=decode)
        grid = make_grid(grid_imgs, nrow=samples_per_class, normalize=True, value_range=(0, 1))
        ax.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Guidance $w={w:.1f}$", fontsize=18)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()


# --------------------------------------------------------------------------- #
#  Pixel-space CFG trainer (Part 3)                                            #
# --------------------------------------------------------------------------- #
class MNISTCFGTrainer(CFGTrainer):
    """CFG trainer with an MNIST-specific sample-grid checkpoint."""

    def __init__(self, path, eta: float, null_label: int, device, **kwargs):
        super().__init__(path=path, eta=eta, null_label=null_label, **kwargs)
        self.device = device

    def checkpoint(self, step: int):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"step_{step:06d}_model.pt"))
        visualize_output(self.model, self.path, self.device,
                         save_path=os.path.join(self.output_dir, f"step_{step:06d}_output.png"))


# --------------------------------------------------------------------------- #
#  VAE trainer (Part 4)                                                        #
# --------------------------------------------------------------------------- #
class MNISTVAETrainer(Trainer):
    """Trains the VAE to reconstruct MNIST (Lab 3, Part 4)."""

    def __init__(self, mnist: MNISTSampler, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.mnist = mnist
        self.batch_size = batch_size

    def get_train_loss(self) -> torch.Tensor:
        x, _ = self.mnist.sample(self.batch_size)
        z_mean, z_logvar, x_mean, x_logvar = self.model(x)
        return self.model.compute_loss(z_mean, z_logvar, x_mean, x_logvar, x)


# --------------------------------------------------------------------------- #
#  Latent-space CFG trainer (Part 5)                                          #
# --------------------------------------------------------------------------- #
class LatentCFGTrainer(Trainer):
    """Latent diffusion: encode MNIST with a frozen VAE, train a DiT in latent space."""

    def __init__(self, mnist: MNISTSampler, vae: VAE, path: GaussianConditionalProbabilityPath,
                 eta: float, null_label: int, device, eps: float = 0.001, **kwargs):
        assert 0 < eta < 1
        super().__init__(**kwargs)
        self.mnist = mnist
        self.vae = vae
        self.path = path
        self.eta = eta
        self.eps = eps
        self.null_label = null_label
        self.device = device

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        with torch.no_grad():
            xx, y = self.mnist.sample(batch_size)
            z_mean, z_logvar = self.vae.encode(xx)
            zz = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        yi = torch.rand(y.shape[0]).to(y.device)
        y = y.clone()
        y[yi < self.eta] = self.null_label
        t = torch.rand(batch_size).to(zz) * (1 - self.eps)
        zx = self.path.sample_conditional_path(zz, t)
        ut_theta = self.model(zx, t, y)
        ut_ref = self.path.conditional_vector_field(zx, zz, t)
        return torch.square(ut_theta - ut_ref).mean()

    def checkpoint(self, step: int):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"step_{step:06d}_model.pt"))
        visualize_output(self.model, self.path, self.device,
                         save_path=os.path.join(self.output_dir, f"step_{step:06d}_output.png"),
                         decode=self.vae)


__all__ = [
    "MNISTSampler",
    "sample_grid",
    "visualize_output",
    "MNISTCFGTrainer",
    "MNISTVAETrainer",
    "LatentCFGTrainer",
]
