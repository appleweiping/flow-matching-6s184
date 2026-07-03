"""Generate clean, runnable *completed* lab notebooks that import the
``flow_matching_labs`` package and reproduce each lab's key results.

These complement the original (unsolved) notebooks under ``notebooks/`` — they
show the intended usage of every implemented class.  Run once:

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
import os

NB_DIR = "notebooks"


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


# --------------------------------------------------------------------------- #
LAB1 = nb([
    md("# Lab 1 (completed) — Simulating ODEs and SDEs\n\n"
       "Uses `flow_matching_labs`. Implements the Euler / Euler–Maruyama simulators, "
       "Brownian motion, the Ornstein–Uhlenbeck process, and Langevin dynamics."),
    code("""
import torch
from flow_matching_labs.core import EulerMaruyamaSimulator
from flow_matching_labs.lab1 import BrownianMotion, OUProcess, LangevinSDE
from flow_matching_labs.distributions import Gaussian, GaussianMixture
torch.manual_seed(0)
"""),
    md("## Q2.1 Brownian motion — Var$[X_t] = \\sigma^2 t$"),
    code("""
sim = EulerMaruyamaSimulator(BrownianMotion(sigma=1.0))
xf = sim.simulate(torch.zeros(20000, 1), torch.linspace(0, 5, 500))
print("Var[X_5] =", round(xf.var().item(), 3), " (expected 5.0)")
"""),
    md("## Q2.2 Ornstein–Uhlenbeck — stationary Var $=\\sigma^2/(2\\theta)$"),
    code("""
sim = EulerMaruyamaSimulator(OUProcess(theta=0.25, sigma=1.0))
xf = sim.simulate(torch.randn(20000, 1) * 8, torch.linspace(0, 60, 6000))
print("stationary Var =", round(xf.var().item(), 3), " (expected 2.0)")
"""),
    md("## Q3.1 Langevin dynamics — sample an arbitrary density from its score"),
    code("""
target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3)
sim = EulerMaruyamaSimulator(LangevinSDE(sigma=0.6, density=target))
x0 = Gaussian(torch.zeros(2), 20 * torch.eye(2)).sample(2000)
xf = sim.simulate(x0, torch.linspace(0, 5, 1000))
print("mean log-density:", round(target.log_density(x0).mean().item(), 2),
      "->", round(target.log_density(xf).mean().item(), 2))
"""),
])

LAB2 = nb([
    md("# Lab 2 (completed) — Flow Matching and Score Matching\n\n"
       "Gaussian & linear conditional probability paths, and the flow- / score-matching "
       "training objectives on 2-D toy distributions."),
    code("""
import torch
from flow_matching_labs.core import EulerSimulator, build_ts
from flow_matching_labs.distributions import GaussianMixture
from flow_matching_labs.paths import GaussianConditionalProbabilityPath, LinearAlpha, SquareRootBeta
from flow_matching_labs.models import MLPVectorField, ConditionalFlowMatchingTrainer, LearnedVectorFieldODE
torch.manual_seed(0)
"""),
    md("## Q3.1 Flow matching with a Gaussian conditional path\n\n"
       "Train $u_t^\\theta$ to match the conditional vector field, then integrate the "
       "learned ODE from noise to data."),
    code("""
target = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0)
path = GaussianConditionalProbabilityPath(target, LinearAlpha(), SquareRootBeta())
model = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64])
losses = ConditionalFlowMatchingTrainer(path, model).train(
    num_epochs=2000, device='cpu', lr=1e-3, batch_size=1000, progress=False)
print("final FM loss:", round(losses[-1], 3))

sim = EulerSimulator(LearnedVectorFieldODE(model))
gen = sim.simulate(path.p_simple.sample(5000), build_ts(500, 5000, 'cpu'))
print("generated mean/std:", gen.mean(0).tolist(), gen.std(0).tolist())
"""),
    md("See `scripts/run_lab2.py` for the full experiments (score matching + Langevin "
       "SDE sampling, and the linear-path checkerboard) with saved density-match figures "
       "under `results/lab2/`."),
])

LAB3 = nb([
    md("# Lab 3 (completed) — A Conditional Generative Model for Images\n\n"
       "Classifier-free guidance, a Diffusion Transformer, a VAE, and latent diffusion. "
       "The full MNIST training run lives in `scripts/run_lab3_mnist.py`; here we show the "
       "2-D CFG sanity check and the model wiring."),
    code("""
import math, torch
from flow_matching_labs.distributions import GMM
from flow_matching_labs.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from flow_matching_labs.cfg import MLPConditionalVectorField, CFGTrainer, CFGVectorFieldODE
from flow_matching_labs.core import EulerSimulator
torch.manual_seed(0)
"""),
    md("## Q2 Classifier-free guidance on a 3-mode GMM"),
    code("""
angles = [0, 2*math.pi/3, 4*math.pi/3]
means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
gmm = GMM(means, torch.tensor([0.2, 0.2, 0.2]), torch.tensor([1/3, 1/3, 1/3]))
path = GaussianConditionalProbabilityPath(gmm, LinearAlpha(), LinearBeta(), p_simple_shape=[2])
vf = MLPConditionalVectorField(dim=2, hidden_dim=256, class_dim=2, num_classes=3)
CFGTrainer(path=path, eta=0.25, null_label=3).train(
    model=vf, num_steps=2000, lr=1e-3, warmup_steps=200, batch_size=250, progress=False)

ode = CFGVectorFieldODE(vf, guidance_scale=1.0, null_label=3)
sim = EulerSimulator(ode)
for mode in range(3):
    y = torch.full((300,), mode).long()
    xs = sim.simulate(path.p_simple.sample(300), torch.linspace(0,1,100).expand(300,-1), y=y)
    print(f"class {mode}: sample mean {xs.mean(0).tolist()} (target {means[mode].tolist()})")
"""),
    md("## Q3–Q4 Diffusion Transformer + VAE wiring"),
    code("""
from flow_matching_labs.dit import DiffusionTransformerFlowModel
from flow_matching_labs.vae import VAE
model = DiffusionTransformerFlowModel(img_size=32, patch_size=4, num_layers=4, c=1, dim=96, heads=4)
print("DiT output:", tuple(model(torch.randn(2,1,32,32), torch.rand(2), torch.zeros(2).long()).shape))
vae = VAE(data_channels=1, hidden_channels=[16,32,64,128], beta=10.0)
zmean, _ = vae.encode(torch.randn(2,1,32,32))
print("VAE latent:", tuple(zmean.shape))  # (2, 128, 4, 4)
"""),
    md("Run `python scripts/run_lab3_mnist.py` to train the DiT on MNIST and generate the "
       "labelled digit grids saved under `results/lab3/`."),
])


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    for name, obj in [("lab_one_completed", LAB1), ("lab_two_completed", LAB2), ("lab_three_completed", LAB3)]:
        path = os.path.join(NB_DIR, name + ".ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
