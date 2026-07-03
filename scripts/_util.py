"""Shared helpers for the driver scripts: CPU thread setup + 2-D plotting."""

from __future__ import annotations

import os
from typing import Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
DEVICE = torch.device("cpu")


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def hist2d_samples(samples, ax, bins: int = 200, scale: float = 6.0, percentile: int = 99, **kwargs):
    """Plot a 2-D histogram of ``samples`` clipped at a colour percentile."""
    samples = samples.detach().cpu().numpy()
    H, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins,
                               range=[[-scale, scale], [-scale, scale]])
    cmax = np.percentile(H, percentile)
    norm = matplotlib.colors.Normalize(vmax=max(cmax, 1e-8), vmin=0.0)
    ax.imshow(H.T, extent=[xe[0], xe[-1], ye[0], ye[-1]], origin="lower", norm=norm, **kwargs)


def imshow_density(density, ax, scale: float = 15.0, bins: int = 200, **kwargs):
    """Heatmap of a 2-D ``Density``'s log-density."""
    x = torch.linspace(-scale, scale, bins)
    y = torch.linspace(-scale, scale, bins)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    d = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(d.detach().cpu(), extent=[-scale, scale, -scale, scale], origin="lower", **kwargs)


def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")
