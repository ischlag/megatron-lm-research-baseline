"""nGPT-style hyperspherical normalization patch.

Mirrors the structure of ``_research/logging_patch``: a monkey-patch
layer activated by environment variables, leaving upstream Megatron
source files untouched.

Tiers (additive, each tier turns on the previous):

- T1 ``APERTUS_NGPT_WEIGHT_PROJECTION=1``  - row-unit-norm projection on
  every linear weight matrix after each optimizer step.
- T2 ``APERTUS_NGPT_SIGMA_SCALARS=1``      - learnable per-output sigma
  scalars on Q/K, MLP, and logits (eigen-LR style).
- T3 ``APERTUS_NGPT_ARCHITECTURE=1``       - normalized residual
  interpolation, drops all RMSNorm.

See: nGPT (Loshchilov et al, 2024, arXiv:2410.01131).
"""

from .install import install

__all__ = ["install"]
