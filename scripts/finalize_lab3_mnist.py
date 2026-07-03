"""Finalize the MNIST DiT result from a saved checkpoint.

Loads a trained ``DiffusionTransformerFlowModel`` checkpoint, optionally runs a
short additional fine-tune, then generates the final labelled sample grid and
metrics.  Kept short so it completes in a single foreground call.

Usage:
    LAB3_CKPT=results/lab3/runs/<run>/step_002000_model.pt \
    LAB3_EXTRA_STEPS=800 python scripts/finalize_lab3_mnist.py
"""

from __future__ import annotations

import glob
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


def latest_ckpt():
    ck = os.environ.get("LAB3_CKPT")
    if ck and os.path.exists(ck):
        return ck
    cands = sorted(glob.glob(os.path.join(OUT, "runs", "*", "step_*_model.pt")))
    if not cands:
        raise FileNotFoundError("no checkpoint found under results/lab3/runs")
    return cands[-1]


def main():
    seed_everything(0)
    ckpt = latest_ckpt()
    ckpt_step = int(os.path.basename(ckpt).split("_")[1])
    extra_steps = int(os.environ.get("LAB3_EXTRA_STEPS", 800))
    print(f"[finalize] checkpoint={ckpt} (step {ckpt_step}), extra_steps={extra_steps}")

    path = GaussianConditionalProbabilityPath(
        p_data=MNISTSampler(), alpha=LinearAlpha(), beta=LinearBeta(),
        p_simple_shape=[1, 32, 32],
    ).to(DEVICE)

    model = DiffusionTransformerFlowModel(
        img_size=32, patch_size=4, num_layers=4, c=1, dim=96, heads=4,
        final_dim=10, n_classes=11,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    print(f"  loaded {model_size_b(model) / MiB:.2f} MiB, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    losses = []
    total_steps = ckpt_step
    train_time = 0.0
    if extra_steps > 0:
        trainer = MNISTCFGTrainer(path=path, eta=0.35, null_label=10, device=DEVICE)
        t0 = time.time()
        losses, _ = trainer.train(
            model=model, num_steps=extra_steps, lr=6e-4, warmup_steps=0,
            batch_size=96, ckpt_every=None, out_root=os.path.join(OUT, "runs_finalize"),
            progress=True,
        )
        train_time = time.time() - t0
        total_steps = ckpt_step + extra_steps

    torch.save(model.state_dict(), os.path.join(OUT, "dit_mnist.pt"))

    # Final sample grids at several guidance strengths.
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, w in zip(axes, [1.0, 3.0, 5.0]):
        grid_imgs = sample_grid(model, path, DEVICE, samples_per_class=10,
                                num_timesteps=200, guidance_scale=w, null_label=10, n_classes=10)
        grid = make_grid(grid_imgs, nrow=10, normalize=True, value_range=(0, 1))
        ax.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        ax.axis("off"); ax.set_title(f"Guidance $w={w:.1f}$", fontsize=20)
    savefig(fig, os.path.join(OUT, "mnist_samples.png"))

    metrics = {
        "config": {"steps": total_steps, "batch": 96, "dim": 96, "layers": 4,
                   "heads": 4, "patch": 4, "lr": 6e-4},
        "model_params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "checkpoint_step": ckpt_step,
        "extra_finetune_steps": extra_steps,
        "total_steps": total_steps,
        "finetune_final_loss": (losses[-1] if losses else None),
        "finetune_final_loss_last100_mean": (float(torch.tensor(losses[-100:]).mean().item()) if len(losses) >= 100 else None),
        "guidance_scales_sampled": [1.0, 3.0, 5.0],
    }
    with open(os.path.join(OUT, "mnist_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Lab 3 MNIST final metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
