"""Make the package importable when running ``pytest`` from the repo root
without an editable install, and keep the CPU thread budget modest."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OMP_NUM_THREADS", "3")

import torch  # noqa: E402

torch.set_num_threads(3)
torch.manual_seed(0)
