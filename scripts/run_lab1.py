"""Lab 1 driver — simulate Brownian motion, the OU process, and Langevin dynamics,
save figures + measured statistics to ``results/lab1``.

Run:  python scripts/run_lab1.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from scripts._util import DEVICE, seed_everything, imshow_density, savefig, plt  # noqa: E402
from flow_matching_labs.core import EulerMaruyamaSimulator  # noqa: E402
from flow_matching_labs.lab1 import BrownianMotion, OUProcess, LangevinSDE  # noqa: E402
from flow_matching_labs.distributions import Gaussian, GaussianMixture  # noqa: E402

OUT = os.path.join("results", "lab1")


def main():
    seed_everything(0)
    metrics = {}

    # ---- Brownian motion: trajectories + terminal histogram --------------- #
    sigma = 1.0
    sim = EulerMaruyamaSimulator(BrownianMotion(sigma))
    x0 = torch.zeros(500, 1).to(DEVICE)
    ts = torch.linspace(0, 5.0, 500).to(DEVICE)
    traj = sim.simulate_with_trajectory(x0, ts)  # (n, nts, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
    for i in range(traj.shape[0]):
        a1.plot(ts.cpu(), traj[i, :, 0].cpu(), color="#2a6f97", alpha=0.06)
    a1.set_xlabel("time $t$"); a1.set_ylabel("$X_t$")
    a1.set_title(rf"Brownian motion trajectories ($\sigma={sigma}$)")
    a2.hist(traj[:, -1, 0].cpu().numpy(), bins=40, orientation="horizontal", color="#e07a5f")
    a2.set_title("terminal $X_5$")
    savefig(fig, os.path.join(OUT, "brownian_motion.png"))
    metrics["brownian_terminal_var"] = float(traj[:, -1, 0].var().item())
    metrics["brownian_terminal_var_expected"] = sigma ** 2 * 5.0

    # ---- OU process: several (theta, sigma) --------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ou_stats = []
    for ax, (theta, s) in zip(axes, [(0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]):
        sim = EulerMaruyamaSimulator(OUProcess(theta, s))
        x0 = torch.linspace(-10, 10, 200).view(-1, 1).to(DEVICE)
        ts = torch.linspace(0, 15.0, 1500).to(DEVICE)
        tr = sim.simulate_with_trajectory(x0, ts)
        for i in range(tr.shape[0]):
            ax.plot(ts.cpu(), tr[i, :, 0].cpu(), alpha=0.15, color="#3d405b")
        expected = s ** 2 / (2 * theta)
        # measure stationary var from a large batch
        xb = sim.simulate(torch.randn(4000, 1).to(DEVICE) * 8, ts)
        ou_stats.append({"theta": theta, "sigma": s,
                         "stationary_var": float(xb.var().item()),
                         "stationary_var_expected": expected})
        ax.set_title(rf"OU $\theta={theta},\ \sigma={s}$")
        ax.set_xlabel("time $t$")
    savefig(fig, os.path.join(OUT, "ou_process.png"))
    metrics["ou"] = ou_stats

    # ---- Langevin dynamics sampling a GMM ---------------------------------- #
    target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3).to(DEVICE)
    sde = LangevinSDE(sigma=0.6, density=target)
    sim = EulerMaruyamaSimulator(sde)
    src = Gaussian(torch.zeros(2), 20 * torch.eye(2)).to(DEVICE)
    x0 = src.sample(2000)
    ts = torch.linspace(0, 5.0, 1000).to(DEVICE)
    traj = sim.simulate_with_trajectory(x0, ts)
    plot_ts = [0, len(ts) // 3, 2 * len(ts) // 3, len(ts) - 1]
    fig, axes = plt.subplots(1, len(plot_ts), figsize=(5 * len(plot_ts), 5))
    for ax, ti in zip(axes, plot_ts):
        imshow_density(target, ax, scale=18, cmap="Blues", alpha=0.5, vmin=-15)
        xt = traj[:, ti]
        ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=4, c="black", alpha=0.4, marker="x")
        ax.set_title(f"Langevin samples at $t={ts[ti].item():.1f}$")
        ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, os.path.join(OUT, "langevin_gmm.png"))
    metrics["langevin_logdensity_before"] = float(target.log_density(x0).mean().item())
    metrics["langevin_logdensity_after"] = float(target.log_density(traj[:, -1]).mean().item())

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Lab 1 metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
