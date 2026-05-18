# Copyright (c) 2026, ETH Zurich.

"""RMNP optimizer (Deng et al., 2026).

Row-Momentum Normalized Preconditioning: replaces Muon's Newton-Schulz
orthogonalization with row L2 normalization. Per-step preconditioning
cost is O(mn) vs Muon's O(mn * min(m, n)), with 13-44x wall-clock
speedup reported on LLM-scale weight matrices. The paper claims
RMNP matches Muon's final perplexity at 60M-1.5B pretraining; the
surrounding literature (Lin, https://nil9.net/posts/rownorm_optimizer/)
disputes the equivalence on theoretical grounds (RMS-close-to-diagonal
in V V^T does not imply equivalent inverse-square-root behavior on
small singular values).

References:
- Paper: https://arxiv.org/abs/2603.20527
- Reference impl: https://github.com/Dominator-Index/RMNP

Algorithm (per 2D matrix parameter):
    V_t = beta * V_{t-1} + (1 - beta) * G_t            # momentum EMA
    D_t = V_t / clamp(||V_{t,i:}||_2, min=eps)         # row L2 normalize
    W_{t+1} = W_t - lr_t * extra_scale_factor * D_t    # step

Scalar params (embeddings, norms, biases, output proj) are routed to
AdamW via the same predicate Muon / Aurora use.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size

try:
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


@torch.no_grad()
def row_normalize(grad: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Row-wise L2 normalization of a 2D momentum tensor.

    For a tensor of shape (m, n), divides each row by its L2 norm.
    Output has the same shape as input; each row has L2 norm 1 (modulo eps).
    Equivalent to diag(V V^T)^(-1/2) @ V, the diagonal approximation of
    Muon's polar inverse-square-root preconditioner.
    """
    if grad.ndim != 2:
        raise ValueError(f"RMNP requires 2D grad, got shape {tuple(grad.shape)}")
    row_norms = grad.norm(dim=-1, keepdim=True).clamp_min(eps)
    return grad / row_norms


class TensorParallelRMNP(OrthogonalizedOptimizer):
    """RMNP optimizer wired through emerging_optimizers' OrthogonalizedOptimizer parent.

    The parent handles SGD-momentum + Nesterov + decoupled weight decay; this
    class supplies the orthogonalize step (row L2 normalization plus a
    constant extra scale factor). v1 supports tp_mode='duplicated' /
    'blockwise' only; each rank row-normalizes its full local shard.
    With TP > 1 and partition_dim along the row axis the local row L2
    norm is exact. With partition_dim along the column axis (rare in
    Megatron's TP layout for 2D weights), a cross-rank sum-reduce on
    the row L2 would be required and is not implemented in v1.

    QKV split follows the Muon / Aurora convention: when split_qkv is on
    and is_qkv_fn(p) is True, the (Q, K, V) sub-blocks are row-normalized
    independently to preserve their relative shape semantics.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        extra_scale_factor: float = 1.0,
        eps: float = 1e-7,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["duplicated", "blockwise"] = "duplicated",
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
    ) -> None:
        if not HAVE_EMERGING_OPTIMIZERS:
            raise ImportError(
                "emerging_optimizers is required for RMNP's parent class "
                "(OrthogonalizedOptimizer handles momentum + WD)."
            )
        if tp_mode not in ("duplicated", "blockwise"):
            raise ValueError(
                f"RMNP v1 supports tp_mode in ('duplicated', 'blockwise'), got {tp_mode!r}."
            )

        def scaled_orthogonalize_fn(
            g: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup | None,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            # Row L2 normalize the momentum tensor. partition_dim is
            # irrelevant in tp_mode='duplicated' / 'blockwise' (each rank
            # processes its full local matrix; v1 does not coordinate
            # across TP ranks).
            update = row_normalize(g, eps=eps)
            return update * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self._rmnp_eps = eps
        self._rmnp_extra_scale_factor = extra_scale_factor

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply RMNP's row normalization to the momentum tensor."""
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, "expert_tp", False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            grad_shape = grad.shape
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)  # type: ignore[arg-type]
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),  # type: ignore[arg-type]
                self.qkv_split_shapes,  # type: ignore[arg-type]
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]
            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad
