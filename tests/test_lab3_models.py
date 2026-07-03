"""Lab 3 — verify DiT components, VAE, and CFG guidance behaviour."""

import math

import torch

from flow_matching_labs.dit import (
    FourierEncoder, Patchifier, MHA, Depatchifier,
    DiffusionTransformer, DiffusionTransformerFlowModel,
)
from flow_matching_labs.vae import VAE
from flow_matching_labs.cfg import (
    MLPConditionalVectorField, CFGTrainer, CFGVectorFieldODE,
)
from flow_matching_labs.core import EulerSimulator
from flow_matching_labs.paths import (
    GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta,
)
from flow_matching_labs.distributions import GMM

torch.manual_seed(0)
DEV = "cpu"


def test_fourier_encoder_shape_and_scale():
    enc = FourierEncoder(32)
    out = enc(torch.rand(8))
    assert out.shape == (8, 32)
    assert torch.isfinite(out).all()


def test_patchifier_token_count():
    p = Patchifier(img_size=32, patch_size=8, c_in=1, dim=64)
    toks = p(torch.randn(4, 1, 32, 32))
    assert toks.shape == (4, (32 // 8) ** 2, 64)  # (4, 16, 64)


def test_mha_is_permutation_shape_preserving():
    mha = MHA(64, 4)
    x = torch.randn(2, 10, 64)
    assert mha(x).shape == x.shape


def test_dit_flow_model_end_to_end_shape():
    model = DiffusionTransformerFlowModel(
        img_size=32, patch_size=8, num_layers=2, c=1, dim=64, heads=4,
        final_dim=8, n_classes=11,
    )
    x = torch.randn(3, 1, 32, 32)
    t = torch.rand(3)
    y = torch.randint(0, 11, (3,))
    out = model(x, t, y)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_vae_latent_and_reconstruction_shapes():
    vae = VAE(data_channels=1, hidden_channels=[16, 32, 64, 128], beta=1.0)
    x = torch.randn(2, 1, 32, 32)
    z_mean, z_logvar = vae.encode(x)
    assert z_mean.shape == (2, 128, 4, 4)  # 3 downsamples: 32 -> 16 -> 8 -> 4
    z_mean, z_logvar, x_mean, x_logvar = vae(x)
    assert x_mean.shape == x.shape
    loss = vae.compute_loss(z_mean, z_logvar, x_mean, x_logvar, x)
    assert torch.isfinite(loss) and loss.item() > 0


def test_cfg_ode_reduces_to_conditional_at_w1():
    # With guidance_scale=1, the guided field equals the conditional field.
    model = MLPConditionalVectorField(dim=2, hidden_dim=32, class_dim=2, num_classes=3)
    ode = CFGVectorFieldODE(model, null_label=3, guidance_scale=1.0)
    x = torch.randn(16, 2)
    t = torch.rand(16)
    y = torch.randint(0, 3, (16,))
    guided = ode.drift_coefficient(x, t, y=y)
    cond = model(x, t, y)
    assert torch.allclose(guided, cond, atol=1e-6)


def test_cfg_trains_and_separates_modes_on_gmm():
    # Train a small conditional field on a 3-mode GMM; each conditioned sample
    # cloud should land near its mode.
    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
    covs = torch.tensor([0.2, 0.2, 0.2])
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    gmm = GMM(means, covs, weights)
    path = GaussianConditionalProbabilityPath(
        p_data=gmm, alpha=LinearAlpha(), beta=LinearBeta(), p_simple_shape=[2],
    )
    vf = MLPConditionalVectorField(dim=2, hidden_dim=128, class_dim=2, num_classes=3)
    trainer = CFGTrainer(path=path, eta=0.25, null_label=3)
    losses, _ = trainer.train(model=vf, num_steps=600, lr=1e-3, warmup_steps=100,
                              batch_size=250, progress=False)
    assert losses[-1] < losses[0]

    ode = CFGVectorFieldODE(vf, guidance_scale=2.0, null_label=3)
    sim = EulerSimulator(ode)
    for mode in range(3):
        y = torch.full((400,), mode, dtype=torch.long)
        x0 = path.p_simple.sample(400)
        ts = torch.linspace(0, 1, 100).expand(400, -1)
        xs = sim.simulate(x0, ts, y=y)
        dist = (xs.mean(0) - means[mode]).norm().item()
        assert dist < 0.8, f"mode {mode} landed {dist:.2f} from its centre"
