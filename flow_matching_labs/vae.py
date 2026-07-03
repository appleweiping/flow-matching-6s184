"""Variational auto-encoder for latent diffusion — Lab 3, Part 4.

Implements the convolutional VAE that compresses ``1×32×32`` MNIST images into a
``128×4×4`` latent, used later as the space in which the latent diffusion model
(``flow_matching_labs.dit`` in latent mode) operates:

    * Q4.1 ``ResidualBlock``  — GroupNorm + conv + SiLU + zero-init residual.
    * Q4.2 ``AttnBlock``      — spatial self-attention block.
    * Q4.3–4 ``EncoderBlock`` / ``Encoder``.
    * Q4.5–6 ``DecoderBlock`` / ``Decoder``.
    * Q4.7 ``VAE``            — encode/decode + the ELBO ``compute_loss``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .dit import MHA, MLP


# --------------------------------------------------------------------------- #
#  Q4.1 — Residual block                                                       #
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """GroupNorm + 3×3 conv + SiLU + 1×1 conv, with a zero-init residual."""

    def __init__(self, channels: int, act: type = nn.SiLU):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.act1 = act()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x_skip + x


# --------------------------------------------------------------------------- #
#  Q4.2 — Attention block                                                      #
# --------------------------------------------------------------------------- #
class AttnBlock(nn.Module):
    """Spatial self-attention over an image feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.reshape1 = Rearrange("b c h w -> b (h w) c")
        self.norm1 = nn.LayerNorm(channels)
        self.mha = MHA(channels, 1)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = MLP([channels, 2 * channels, channels], final_init=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.reshape1(x)
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# --------------------------------------------------------------------------- #
#  Q4.3–4.4 — Encoder                                                          #
# --------------------------------------------------------------------------- #
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, downsample_channels: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        self.downsample = (
            nn.Conv2d(in_channels, downsample_channels, padding=1, stride=2, kernel_size=3)
            if downsample_channels is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int]):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1, stride=1)
        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        self.blocks = nn.ModuleList([EncoderBlock(i, o) for i, o in zip(ch_in, ch_out)])
        z_dim = hidden_channels[-1]
        self.z_mean = nn.Sequential(
            nn.GroupNorm(1, z_dim),
            nn.Conv2d(z_dim, z_dim, kernel_size=1, stride=1, padding=0),
        )
        self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.z_mean(x), self.logvar


# --------------------------------------------------------------------------- #
#  Q4.5–4.6 — Decoder                                                          #
# --------------------------------------------------------------------------- #
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, upsample_channels: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        self.upsample = (
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, upsample_channels, kernel_size=3, padding=1, stride=1),
            )
            if upsample_channels is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: List[int]):
        super().__init__()
        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        self.blocks = nn.ModuleList([DecoderBlock(i, o) for i, o in zip(ch_in, ch_out)])
        x_dim = hidden_channels[-1]
        self.x_mean = nn.Sequential(
            nn.GroupNorm(1, x_dim),
            nn.Conv2d(x_dim, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return self.x_mean(x), self.logvar


# --------------------------------------------------------------------------- #
#  Q4.7 — the VAE                                                              #
# --------------------------------------------------------------------------- #
class VAE(nn.Module):
    """Convolutional VAE with a Gaussian encoder/decoder (Lab 3, Q4.7)."""

    def __init__(self, data_channels: int, hidden_channels: List[int], beta: float = 0.1):
        super().__init__()
        self.beta = beta
        self._encoder = Encoder(data_channels, hidden_channels)
        self._decoder = Decoder(data_channels, list(reversed(hidden_channels)))

    def encode(self, x: torch.Tensor):
        return self._encoder(x)

    def decode(self, z: torch.Tensor):
        return self._decoder(z)

    def forward(self, x: torch.Tensor):
        z_mean, z_logvar = self.encode(x)
        z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        x_mean, x_logvar = self.decode(z)
        return z_mean, z_logvar, x_mean, x_logvar

    def compute_loss(self, z_mean, z_logvar, x_mean, x_logvar, x_true) -> torch.Tensor:
        """Negative ELBO: β·KL(q‖N(0,I)) + Gaussian reconstruction NLL."""
        eps = 1e-6
        kl_loss = self.beta * (z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1).mean()
        mse_term = (x_true - x_mean).pow(2) / (torch.exp(x_logvar) + eps)
        recon_loss = (mse_term + x_logvar).mean()
        return kl_loss + recon_loss


__all__ = [
    "ResidualBlock",
    "AttnBlock",
    "EncoderBlock",
    "Encoder",
    "DecoderBlock",
    "Decoder",
    "VAE",
]
