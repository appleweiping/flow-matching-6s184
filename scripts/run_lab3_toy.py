"""Lab 3 driver (sanity check) — classifier-free guidance on a 3-mode GMM.

Trains an MLP conditional vector field with label dropout, then draws
class-conditioned samples at several guidance strengths and measures how tightly
each conditioned cloud clusters on its mode.  Saves to ``results/lab3``.

Run:  python scripts/run_lab3_toy.py
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from scripts._util import DEVICE, seed_everything, savefig, plt  # noqa: E402
from flow_matching_labs.core import EulerSimulator  # noqa: E402
from flow_matching_labs.distributions import GMM  # noqa: E402
from flow_matching_labs.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta  # noqa: E402
from flow_matching_labs.cfg import MLPConditionalVectorField, CFGTrainer, CFGVectorFieldODE  # noqa: E402

OUT = os.path.join("results", "lab3")


def main():
    os.makedirs(OUT, exist_ok=True)
    seed_everything(0)

    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
    covs = torch.tensor([0.2, 0.2, 0.2])
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    gmm = GMM(means, covs, weights).to(DEVICE)

    path = GaussianConditionalProbabilityPath(gmm, LinearAlpha(), LinearBeta(), p_simple_shape=[2]).to(DEVICE)
    vf = MLPConditionalVectorField(dim=2, hidden_dim=256, class_dim=2, num_classes=3).to(DEVICE)
    trainer = CFGTrainer(path=path, eta=0.25, null_label=3)
    losses, steps = trainer.train(model=vf, num_steps=3000, lr=1e-3, warmup_steps=300,
                                  batch_size=250, progress=True)

    guidance_scales = [1.0, 2.0, 4.0]
    fig, axes = plt.subplots(1, len(guidance_scales) + 1, figsize=(6 * (len(guidance_scales) + 1), 6))
    xd, _ = gmm.sample(750)
    xd = xd.cpu().numpy()
    axes[0].scatter(xd[:, 0], xd[:, 1], s=6, marker="*", c="grey")
    axes[0].set_title("Target GMM"); axes[0].set_aspect("equal")

    metrics = {"final_loss": losses[-1], "mode_distances": {}}
    for ax, w in zip(axes[1:], guidance_scales):
        ode = CFGVectorFieldODE(vf, guidance_scale=w, null_label=3)
        sim = EulerSimulator(ode)
        dists = []
        for mode in range(3):
            y = torch.full((250,), mode, dtype=torch.long).to(DEVICE)
            x0 = path.p_simple.sample(250)
            ts = torch.linspace(0, 1, 100).expand(250, -1).to(DEVICE)
            xs = sim.simulate(x0, ts, y=y)
            dists.append(float((xs.mean(0).cpu() - means[mode]).norm().item()))
            xs = xs.cpu().numpy()
            ax.scatter(xs[:, 0], xs[:, 1], s=6, marker="*", label=f"class {mode}")
        ax.set_title(f"CFG samples ($w={w:.1f}$)"); ax.legend(); ax.set_aspect("equal")
        metrics["mode_distances"][f"w={w}"] = dists

    savefig(fig, os.path.join(OUT, "cfg_gmm.png"))
    with open(os.path.join(OUT, "cfg_gmm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Lab 3 (toy) metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
