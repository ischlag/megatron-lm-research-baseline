# Copyright (c) 2026, ETH Zurich.

"""Muown / NorMuown optimizer (Lion, Hubler, Li, Orvieto, He; 2026).

Muown applies internal weight normalization to 2D weight matrices: each
W is implicitly parameterized as (g, v) where g is per-row magnitude and
W = g * v / ||v|| is the composed weight used in the forward pass. The
optimizer then applies:

- Muon (momentum + Newton-Schulz orthogonalization) to the direction v
- AdamW to the magnitude g and all 1D / 0D parameters (norms, biases)

NorMuown extends Muown with NorMuon-style per-row second-moment
rescaling on the orthogonalized direction update (use_normuon=True).

Reference paper: https://arxiv.org/abs/2605.10797
Reference impl:  https://github.com/kcc-lion/muown

This Megatron port mirrors the reference's per-param math while
deferring scalar-group routing (embeddings / norms / biases) and QKV
splitting to the existing emerging_optimizers infrastructure. State per
2D param is dominated by m_v (momentum, same shape as W) and a thin
[d_out, 1] vector of magnitude state (g, v_norm, m_g, v_g, optionally
v_normuon) -- the same kind of shape that breaks torch_dist
checkpointing for NorMuon, so chains with Muown / NorMuown should use
--ckpt-format torch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


logger = logging.getLogger(__name__)


@torch.no_grad()
def newton_schulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Quintic iteration with the canonical Muon coefficients (3.4445, -4.7750, 2.0315)
    in bf16. Caller is responsible for casting the result back to the param's
    dtype if needed.
    """
    assert G.ndim == 2, f"newton_schulz5 expects 2D, got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def _wn_pre_ns(W: Tensor, g: Tensor, v_norm: Tensor, grad_W: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Reconstruct direction v from (W, g, v_norm) and compute weight-norm Jacobian.

    Numerically stable: W_i / g_i is O(1) per element, while v_norm / g or
    1 / g can overflow for tiny |g|.
    """
    u = W / g
    v = u * v_norm
    grad_g = (grad_W * u).sum(dim=1, keepdim=True)
    grad_v = (g / v_norm) * (grad_W - u * grad_g)
    return v, grad_g, grad_v


@torch.no_grad()
def _wn_recompose(W: Tensor, g: Tensor, v_new: Tensor) -> Tensor:
    """Recompose W[:] = g * v_new / ||v_new|| in-place; returns the new v_norm."""
    v_norm_new = v_new.norm(dim=1, keepdim=True)
    W.copy_(g * (v_new / v_norm_new))
    return v_norm_new


@torch.no_grad()
def _normuon_rescale(update: Tensor, second_momentum: Tensor, beta2: float) -> Tensor:
    """NorMuon-style per-row second-moment rescaling, norm-preserving.

    Divides the orthogonalized direction update by sqrt of an EMA of per-row
    mean-square magnitudes, then re-scales so the Frobenius norm of the
    update is unchanged. Mutates second_momentum in place.
    """
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = (update * update).mean(dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add(1e-10)
    update = update * step_size
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update = update * (vnorm / vnorm_new.add(1e-10))
    return update


class Muown(Optimizer):
    """Muown / NorMuown: Muon with internal Weight Normalization.

    For 2D matrix parameters, maintains the implicit (g, v) parameterization
    described in the module docstring. For non-2D parameters (scalars,
    embeddings, biases), use AdamW separately via the scalar-group routing
    in the Megatron emerging_optimizers registry.

    Args:
        params: iterable of 2D parameters to optimize.
        lr: shared base LR (direction picks up an extra 0.2 * sqrt(max(m,n))
            scale via the Moonlight convention so the per-coord direction
            update has RMS approx 0.2 * lr).
        momentum: SGD momentum for the direction component.
        nesterov: Nesterov-style momentum lookahead.
        betas: (beta1, beta2) for the AdamW step on magnitude g.
        weight_decay: decoupled weight decay coefficient (applied to W, then
            g is resynced).
        adam_eps: stability epsilon in the AdamW denominator on g.
        ns_steps: number of Newton-Schulz iterations for orthogonalization.
        use_normuon: if True, apply NorMuon-style per-row 2nd-moment rescaling
            to the orthogonalized direction update before applying. Default False
            (= plain Muown).
        normuon_beta2: EMA coefficient for the per-row 2nd moment when
            use_normuon=True.
        split_qkv: if True, treat fused QKV params (detected via is_qkv_fn)
            as three separate 2D matrices for the Newton-Schulz step. The
            (g, v_norm, m_g, v_g) magnitude state stays per-row of the
            concatenated tensor.
        is_qkv_fn: callable(param) -> bool; defaults to checking p.is_qkv.
        qkv_split_shapes: (q, k, v) row counts for the QKV split.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.8e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
        use_normuon: bool = False,
        normuon_beta2: float = 0.95,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[Tensor], bool]] = None,
        qkv_split_shapes: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError(f"Invalid normuon_beta2: {normuon_beta2}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            ns_steps=ns_steps,
            use_normuon=use_normuon,
            normuon_beta2=normuon_beta2,
        )
        super().__init__(params, defaults)
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn or (lambda p: getattr(p, "is_qkv", False))
        self.qkv_split_shapes = qkv_split_shapes

    def _init_state_2d(self, p: Tensor, state: dict, use_normuon: bool) -> None:
        w_norm = p.data.norm(dim=1, keepdim=True)
        state["g"] = w_norm.clone()
        state["v_norm"] = w_norm.clone()
        state["m_v"] = torch.zeros_like(p.data)
        state["m_g"] = torch.zeros_like(w_norm)
        state["v_g"] = torch.zeros_like(w_norm)
        if use_normuon:
            state["v_normuon"] = torch.zeros_like(w_norm)
        state["step"] = 0

    def _muon_direction_update(
        self,
        update: Tensor,
        ns_steps: int,
        param: Tensor,
        state: dict,
        use_normuon: bool,
        normuon_beta2: float,
    ) -> Tensor:
        """Orthogonalize (optionally NorMuon-rescale) the direction-update tensor.

        Returns the update scaled by Moonlight's 0.2 * sqrt(max(m, n)). Handles
        QKV split if applicable.
        """
        is_qkv = self.split_qkv and self.is_qkv_fn(param) and self.qkv_split_shapes is not None
        if is_qkv:
            # update has shape (sum(qkv_split_shapes) * num_query_groups, d_in)
            grad_shape = update.shape
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)  # type: ignore[arg-type]
            qkv = torch.split(
                update.view(num_query_groups, sum(self.qkv_split_shapes), -1),  # type: ignore[arg-type]
                self.qkv_split_shapes,  # type: ignore[arg-type]
                dim=1,
            )
            qkv = [chunk.reshape(-1, grad_shape[-1]) for chunk in qkv]
            ortho = [newton_schulz5(chunk, steps=ns_steps) for chunk in qkv]
            if use_normuon:
                # NorMuon rescaling per-chunk before recombining
                rescaled = []
                for i, o in enumerate(ortho):
                    # second_momentum needs a per-chunk view of state["v_normuon"]
                    # but for simplicity we apply rescale on the recombined matrix
                    # below; here we just collect.
                    rescaled.append(o)
                ortho = rescaled
            chunks_out = []
            for chunk in ortho:
                scale = 0.2 * max(chunk.size(0), chunk.size(1)) ** 0.5
                chunks_out.append(chunk * scale)
            # Recombine in the original layout
            recomb = []
            for chunk in chunks_out:
                recomb.append(chunk.view(num_query_groups, -1, grad_shape[-1]))
            update = torch.cat(recomb, dim=1).view(grad_shape)
            if use_normuon:
                update = _normuon_rescale(update, state["v_normuon"], normuon_beta2)
            return update

        update = newton_schulz5(update, steps=ns_steps)
        if use_normuon:
            update = _normuon_rescale(update, state["v_normuon"], normuon_beta2)
        scale = 0.2 * max(update.size(0), update.size(1)) ** 0.5
        return update * scale

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            adam_eps = group["adam_eps"]
            ns_steps = group["ns_steps"]
            use_normuon = group["use_normuon"]
            normuon_beta2 = group["normuon_beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.data.ndim != 2:
                    raise ValueError(
                        f"Muown expects 2D params; got shape {tuple(p.data.shape)}. "
                        "Route scalar / embedding params to AdamW via the scalar group."
                    )

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    self._init_state_2d(p, state, use_normuon)
                state["step"] += 1
                step = state["step"]

                g = state["g"]
                v_norm = state["v_norm"]
                m_v = state["m_v"]
                m_g = state["m_g"]
                v_g = state["v_g"]
                if weight_decay != 0.0:
                    W_old = p.data.clone()

                # Reconstruct direction v from W, compute grad_g + grad_v
                v, grad_g, grad_v = _wn_pre_ns(p.data, g, v_norm, grad)

                # Muon momentum on v
                m_v.mul_(momentum).add_(grad_v)
                direction_update = grad_v.add(m_v, alpha=momentum) if nesterov else m_v.clone()
                direction_update = self._muon_direction_update(
                    direction_update,
                    ns_steps=ns_steps,
                    param=p,
                    state=state,
                    use_normuon=use_normuon,
                    normuon_beta2=normuon_beta2,
                )
                v_new = v.add(direction_update, alpha=-lr)

                # AdamW step on g (per-row magnitudes)
                m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                g.addcdiv_(m_g / bc1, (v_g / bc2).sqrt().add_(adam_eps), value=-lr)

                # Recompose W = g * v_new / ||v_new||
                state["v_norm"] = _wn_recompose(p.data, g, v_new)

                if weight_decay != 0.0:
                    p.data.add_(W_old, alpha=-lr * weight_decay)
                    # Decoupled WD perturbed p.data off the v_new direction;
                    # resync g so the invariant ||p.data[i]|| == g[i] holds.
                    g.copy_(p.data.norm(dim=1, keepdim=True))

        return loss
