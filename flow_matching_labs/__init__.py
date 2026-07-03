"""flow_matching_labs — clean, importable implementations of the MIT 6.S184 /
6.S975 "Generative AI with Stochastic Differential Equations" labs.

The package is organised by lab:

* ``core``          — shared ODE / SDE / Simulator abstractions (Lab 1 & 2 & 3).
* ``distributions`` — Sampleable / Density objects (Gaussians, mixtures, toy 2D).
* ``lab1``          — numerical simulators, Brownian motion, OU process, Langevin.
* ``paths``         — conditional probability paths (Gaussian & linear) — Lab 2.
* ``models``        — MLP vector-field / score networks + trainers — Lab 2.
* ``cfg``           — classifier-free guidance ODE + trainer — Lab 3.
* ``dit``           — diffusion transformer — Lab 3.
* ``vae``           — variational auto-encoder — Lab 3.

Every function/class corresponds to a specific ``### Question`` / ``### Problem``
in the official notebooks; see the module docstrings for the mapping.
"""

from . import core, distributions  # noqa: F401

__all__ = ["core", "distributions"]
__version__ = "1.0.0"
