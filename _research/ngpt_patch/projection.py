"""Unit-row L2 projection on linear weight matrices (nGPT T1)."""

from __future__ import annotations

from typing import Iterable

import torch

# Megatron parameter-name suffixes that should land on the unit hypersphere.
# Each matched tensor is 2D ``[out, in]``; we normalize rows (dim 0) so each
# output direction has unit L2 norm.
_PROJECT_SUFFIXES = (
    "linear_qkv.weight",       # fused QKV proj (rows = Q heads | K heads | V heads, stacked)
    "linear_proj.weight",      # attention output proj
    "linear_fc1.weight",       # MLP up (or fused gate+up for SwiGLU)
    "linear_fc2.weight",       # MLP down
    "word_embeddings.weight",  # input token embeddings
    "output_layer.weight",     # LM head (when --untie-embeddings-and-output-weights)
)


def _matches(name: str) -> bool:
    return any(name.endswith(s) for s in _PROJECT_SUFFIXES)


@torch.no_grad()
def project_unit_row(models: Iterable | torch.nn.Module, eps: float = 1e-8) -> int:
    """In-place: every matched 2D weight has each row L2-normalized to 1.

    Returns number of tensors projected. Safe to call repeatedly.
    """
    if isinstance(models, torch.nn.Module):
        models = [models]

    n = 0
    for model in models:
        for name, param in model.named_parameters():
            if not _matches(name):
                continue
            if param.dim() != 2:
                continue
            norms = param.data.norm(dim=1, keepdim=True).clamp_min(eps)
            param.data.div_(norms)
            n += 1
    return n
