"""Diffusion Transformer (DiT) for conditional image generation — Lab 3, Part 3.

Implements the full DiT flow model that predicts the marginal vector field
``u_t^θ(x | y)`` for classifier-free guidance:

    * Q3.1 ``FourierEncoder``   — random-Fourier time embedding.
    * Q3.2 ``Patchifier``       — conv-based patch embedding ``(b,c,H,W) → (b,N,d)``.
    * Q3.3 ``MHA`` / ``DiffusionTransformerLayer`` / ``DiffusionTransformer`` —
      multi-head self-attention with adaptive-LayerNorm (adaLN-zero) conditioning.
    * Q3.4 ``Depatchifier``     — MLP + un-patchify + conv back to image space.
    * Q3.5 ``DiffusionTransformerFlowModel`` — the assembled conditional vector
      field, usable in both pixel space (Part 3) and latent space (Part 5).
"""

from __future__ import annotations

import math
from typing import List, Type

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .cfg import ConditionalVectorField


# --------------------------------------------------------------------------- #
#  Small MLP used inside the transformer                                       #
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


# --------------------------------------------------------------------------- #
#  Q3.1 — Fourier time encoder                                                 #
# --------------------------------------------------------------------------- #
class FourierEncoder(nn.Module):
    """Random-Fourier features of a scalar time ``t`` (Lab 3, Q3.1)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        freqs = t * self.weights * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1) * math.sqrt(2)


# --------------------------------------------------------------------------- #
#  Q3.2 — Patchifier                                                           #
# --------------------------------------------------------------------------- #
class Patchifier(nn.Module):
    """Conv-based patch embedding: ``(b, c, H, W) → (b, N, dim)`` (Lab 3, Q3.2)."""

    def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.net = nn.Sequential(
            nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b d h w -> b (h w) d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Q3.3 — Attention + adaLN transformer                                        #
# --------------------------------------------------------------------------- #
class MHA(nn.Module):
    """Multi-headed self-attention (Lab 3, Q3.3)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fold_heads = Rearrange("b n (h d) -> (b h) n d", h=heads)
        self.unfold_heads = Rearrange("(b h) n d -> b n (h d)", h=heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.fold_heads, (q, k, v))
        qk = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)
        x = torch.einsum("bij,bjd->bid", attn, v)
        x = self.unfold_heads(x)
        return self.out(x)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + bias


class DiffusionTransformerLayer(nn.Module):
    """A single DiT block with adaLN-zero conditioning (Lab 3, Q3.3)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6),
        )
        # adaLN-zero: start as identity residual for training stability.
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        self.attn = MHA(dim, heads)
        self.ff = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = rearrange(self.ada_ln(c), "b d -> b 1 d")
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)
        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        x = x + ff_gate * self.ff(modulate(self.norm2(x), ff_scale, ff_bias))
        return x


class DiffusionTransformer(nn.Module):
    """Stack of DiT layers with learned positional encodings (Lab 3, Q3.3)."""

    def __init__(self, depth: int, n_tokens: int, dim: int, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [DiffusionTransformerLayer(dim=dim, **layer_kwargs) for _ in range(depth)]
        )
        self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encodings.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, c)
        return x


# --------------------------------------------------------------------------- #
#  Q3.4 — Depatchifier                                                         #
# --------------------------------------------------------------------------- #
class Depatchifier(nn.Module):
    """Map tokens back to an image ``(b, N, dim) → (b, c_out, H, W)`` (Q3.4)."""

    def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int):
        super().__init__()
        assert img_size % patch_size == 0
        h = w = img_size // patch_size
        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, 4 * dim, final_dim * patch_size ** 2]),
            Rearrange(
                "b (h w) (f ph pw) -> b f (h ph) (w pw)",
                h=h, w=w, f=final_dim, ph=patch_size, pw=patch_size,
            ),
            nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Q3.5 — the assembled conditional vector field                              #
# --------------------------------------------------------------------------- #
class DiffusionTransformerFlowModel(ConditionalVectorField):
    """Full DiT conditional vector field ``u_t^θ(x | y)`` (Lab 3, Q3.5)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        dim: int = 256,
        heads: int = 4,
        final_dim: int = 10,
        n_classes: int = 11,
    ):
        super().__init__()
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = nn.Embedding(num_embeddings=n_classes, embedding_dim=dim)
        self.patchifier = Patchifier(img_size=img_size, patch_size=patch_size, c_in=c, dim=dim)
        n_tokens = (img_size // patch_size) ** 2
        self.dit = DiffusionTransformer(depth=num_layers, n_tokens=n_tokens, dim=dim, heads=heads)
        self.depatchifier = Depatchifier(
            img_size=img_size, patch_size=patch_size, dim=dim, final_dim=final_dim, c_out=c
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embedder(t)
        y_embed = self.y_embedder(y)
        x = self.patchifier(x)
        x = self.dit(x, t_embed + y_embed)
        return self.depatchifier(x)


__all__ = [
    "MLP",
    "FourierEncoder",
    "Patchifier",
    "MHA",
    "modulate",
    "DiffusionTransformerLayer",
    "DiffusionTransformer",
    "Depatchifier",
    "DiffusionTransformerFlowModel",
]
