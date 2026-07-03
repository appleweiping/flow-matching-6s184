"""Lab 3 headline driver — train a Diffusion Transformer flow model on MNIST with
classifier-free guidance and GENERATE real digit samples.

This is the assignment's culminating experiment (pixel-space CFG, Lab 3 Part 3).
Everything is CPU-scale but real: a small DiT trained on the actual MNIST data.
Config is tunable via environment variables so the run fits a CPU budget:

    LAB3_STEPS        number of training steps          (default 5000)
    LAB3_BATCH        batch size                        (default 128)
    LAB3_DIM          transformer width                 (default 128)
    LAB3_LAYERS       number of DiT layers              (default 5)
    LAB3_HEADS        attention heads                   (default 4)
    LAB3_PATCH        patch size                        (default 4)
    LAB3_LR           peak learning rate                (default 6e-4)

Saves loss curve, a labelled 10x10 sample grid at several guidance strengths,
and a metrics JSON to ``results/lab3``.

Run:  python scripts/run_lab3_mnist.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

from scripts._util import DEVICE, seed_everything, savefig, plt  # noqa: E402
from flow_matching_labs.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta  # noqa: E402
from flow_matching_labs.dit import DiffusionTransformerFlowModel  # noqa: E402
from flow_matching_labs.cfg import model_size_b, MiB  # noqa: E402
from flow_matching_labs.lab3 import MNISTSampler, MNISTCFGTrainer, sample_grid  # noqa: E402

OUT = os.path.join("results", "lab3")


def env_int(k, d):
    return int(os.environ.get(k, d))


def main():
    os.makedirs(OUT, exist_ok=True)
    seed_everything(0)

    steps = env_int("LAB3_STEPS", 5000)
    batch = env_int("LAB3_BATCH", 128)
    dim = env_int("LAB3_DIM", 128)
    layers = env_int("LAB3_LAYERS", 5)
    heads = env_int("LAB3_HEADS", 4)
    patch = env_int("LAB3_PATCH", 4)
    lr = float(os.environ.get("LAB3_LR", 6e-4))

    print(f"[Lab3-MNIST] steps={steps} batch={batch} dim={dim} layers={layers} "
          f"heads={heads} patch={patch} lr={lr}")

    path = GaussianConditionalProbabilityPath(
        p_data=MNISTSampler(),
        alpha=LinearAlpha(),
        beta=LinearBeta(),
        p_simple_shape=[1, 32, 32],
    ).to(DEVICE)

    model = DiffusionTransformerFlowModel(
        img_size=32, patch_size=patch, num_layers=layers, c=1,
        dim=dim, heads=heads, final_dim=10, n_classes=11,
    ).to(DEVICE)
    print(f"  model size: {model_size_b(model) / MiB:.2f} MiB, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    trainer = MNISTCFGTrainer(path=path, eta=0.35, null_label=10, device=DEVICE)
    t0 = time.time()
    losses, _ = trainer.train(
        model=model, num_steps=steps, lr=lr, warmup_steps=min(500, steps // 10),
        batch_size=batch, ckpt_every=max(1000, steps // 5), out_root=os.path.join(OUT, "runs"),
        progress=True,
    )
    train_time = time.time() - t0

    torch.save(model.state_dict(), os.path.join(OUT, "dit_mnist.pt"))

    # Loss curve (smoothed).
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.3, color="#2a6f97", label="loss")
    if len(losses) > 50:
        k = 50
        sm = torch.tensor(losses).unfold(0, k, 1).mean(-1)
        ax.plot(range(k - 1, len(losses)), sm.numpy(), color="#e07a5f", label=f"{k}-step MA")
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss"); ax.legend()
    ax.set_title("DiT MNIST training loss")
    savefig(fig, os.path.join(OUT, "mnist_loss.png"))

    # Sample grids at several guidance strengths.
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, w in zip(axes, [1.0, 3.0, 5.0]):
        grid_imgs = sample_grid(model, path, DEVICE, samples_per_class=10,
                                num_timesteps=200, guidance_scale=w, null_label=10, n_classes=10)
        grid = make_grid(grid_imgs, nrow=10, normalize=True, value_range=(0, 1))
        ax.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        ax.axis("off"); ax.set_title(f"Guidance $w={w:.1f}$", fontsize=20)
    savefig(fig, os.path.join(OUT, "mnist_samples.png"))

    metrics = {
        "config": {"steps": steps, "batch": batch, "dim": dim, "layers": layers,
                   "heads": heads, "patch": patch, "lr": lr},
        "model_params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_loss_last100_mean": float(torch.tensor(losses[-100:]).mean().item()),
        "train_time_seconds": train_time,
        "steps_per_second": steps / train_time,
    }
    with open(os.path.join(OUT, "mnist_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Lab 3 MNIST metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
