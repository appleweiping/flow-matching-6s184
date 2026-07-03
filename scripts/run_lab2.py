"""Lab 2 driver — train flow-matching and score-matching models and verify they
reproduce 2-D target densities.  Saves figures + measured MMD to ``results/lab2``.

Runs three experiments:
  1. Gaussian conditional path -> symmetric GMM (flow matching, deterministic ODE).
  2. Gaussian conditional path -> GMM (score matching, Langevin SDE sampling).
  3. Linear conditional path -> checkerboard (flow matching from Gaussian source).

Run:  python scripts/run_lab2.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from scripts._util import DEVICE, seed_everything, hist2d_samples, imshow_density, savefig, plt  # noqa: E402
from flow_matching_labs.core import EulerSimulator, EulerMaruyamaSimulator, build_ts  # noqa: E402
from flow_matching_labs.distributions import Gaussian, GaussianMixture, CheckerboardSampleable  # noqa: E402
from flow_matching_labs.paths import (  # noqa: E402
    GaussianConditionalProbabilityPath, LinearConditionalProbabilityPath,
    LinearAlpha, SquareRootBeta,
)
from flow_matching_labs.models import (  # noqa: E402
    MLPVectorField, MLPScore, ConditionalFlowMatchingTrainer, ConditionalScoreMatchingTrainer,
    LearnedVectorFieldODE, LangevinFlowSDE,
)

OUT = os.path.join("results", "lab2")

# Modest-but-real CPU configs.
FM_EPOCHS = 5000
SM_EPOCHS = 3000
LINEAR_EPOCHS = 8000


def energy_distance(x: torch.Tensor, y: torch.Tensor, n: int = 2000) -> float:
    """A cheap sample-based distributional distance (lower = closer)."""
    x = x[:n]; y = y[:n]
    def pdist_mean(a, b):
        return torch.cdist(a, b).mean()
    return float((2 * pdist_mean(x, y) - pdist_mean(x, x) - pdist_mean(y, y)).item())


def experiment_flow_matching(metrics):
    print("[Lab2] Flow matching on symmetric GMM ...")
    seed_everything(0)
    target = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0).to(DEVICE)
    path = GaussianConditionalProbabilityPath(target, LinearAlpha(), SquareRootBeta()).to(DEVICE)
    model = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64]).to(DEVICE)
    trainer = ConditionalFlowMatchingTrainer(path, model)
    losses = trainer.train(num_epochs=FM_EPOCHS, device=DEVICE, lr=1e-3, batch_size=1000, progress=True)

    n = 20000
    ode = LearnedVectorFieldODE(model)
    sim = EulerSimulator(ode)
    x0 = path.p_simple.sample(n)
    ts = build_ts(500, n, DEVICE)
    gen = sim.simulate(x0, ts)
    real = target.sample(n)
    ed = energy_distance(gen, real)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    hist2d_samples(real, axes[0], scale=15, bins=200); axes[0].set_title("Target (GMM)")
    hist2d_samples(gen, axes[1], scale=15, bins=200); axes[1].set_title("Flow-matching ODE samples")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    savefig(fig, os.path.join(OUT, "flow_matching_gmm.png"))
    metrics["flow_matching_gmm"] = {"final_loss": losses[-1], "energy_distance": ed}
    print(f"  FM final loss {losses[-1]:.3f}, energy distance {ed:.4f}")


def experiment_score_matching(metrics):
    print("[Lab2] Score matching + Langevin SDE on symmetric GMM ...")
    seed_everything(1)
    target = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0).to(DEVICE)
    path = GaussianConditionalProbabilityPath(target, LinearAlpha(), SquareRootBeta()).to(DEVICE)
    # Train both a flow and a score net so we can sample with the Langevin SDE.
    flow = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64]).to(DEVICE)
    ConditionalFlowMatchingTrainer(path, flow).train(
        num_epochs=FM_EPOCHS, device=DEVICE, lr=1e-3, batch_size=1000, progress=True)
    score = MLPScore(dim=2, hiddens=[64, 64, 64, 64]).to(DEVICE)
    losses = ConditionalScoreMatchingTrainer(path, score).train(
        num_epochs=SM_EPOCHS, device=DEVICE, lr=1e-3, batch_size=1000, progress=True)

    n = 20000
    sde = LangevinFlowSDE(flow, score, sigma=2.0)
    sim = EulerMaruyamaSimulator(sde)
    x0 = path.p_simple.sample(n)
    ts = build_ts(300, n, DEVICE)
    gen = sim.simulate(x0, ts)
    real = target.sample(n)
    ed = energy_distance(gen, real)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    hist2d_samples(real, axes[0], scale=15, bins=200); axes[0].set_title("Target (GMM)")
    hist2d_samples(gen, axes[1], scale=15, bins=200); axes[1].set_title("Score-matching SDE samples")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    savefig(fig, os.path.join(OUT, "score_matching_gmm.png"))
    metrics["score_matching_gmm"] = {"final_loss": losses[-1], "energy_distance": ed}
    print(f"  SM final loss {losses[-1]:.3f}, energy distance {ed:.4f}")


def experiment_linear_checkerboard(metrics):
    print("[Lab2] Flow matching with a linear path -> checkerboard ...")
    seed_everything(2)
    target = CheckerboardSampleable(DEVICE, grid_size=4)
    path = LinearConditionalProbabilityPath(
        p_simple=Gaussian.isotropic(2, 1.0).to(DEVICE), p_data=target).to(DEVICE)
    model = MLPVectorField(dim=2, hiddens=[100, 100, 100, 100]).to(DEVICE)
    losses = ConditionalFlowMatchingTrainer(path, model).train(
        num_epochs=LINEAR_EPOCHS, device=DEVICE, lr=1e-3, batch_size=2000, progress=True)

    n = 40000
    sim = EulerSimulator(LearnedVectorFieldODE(model))
    x0 = path.p_simple.sample(n)
    ts = build_ts(200, n, DEVICE)
    gen = sim.simulate(x0, ts)
    real = target.sample(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    hist2d_samples(real, axes[0], scale=6, bins=300); axes[0].set_title("Target (checkerboard)")
    hist2d_samples(gen, axes[1], scale=6, bins=300); axes[1].set_title("Linear-path flow samples")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    savefig(fig, os.path.join(OUT, "linear_checkerboard.png"))
    metrics["linear_checkerboard"] = {"final_loss": losses[-1]}
    print(f"  linear-path FM final loss {losses[-1]:.3f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    metrics = {}
    experiment_flow_matching(metrics)
    experiment_score_matching(metrics)
    experiment_linear_checkerboard(metrics)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Lab 2 metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
