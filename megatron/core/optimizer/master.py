# Adapted (slim) from alex's master.py.
#
# Slim cut:
#   - Hypersphere weight clipping (post-step only, no update normalization).
#   - L2 only 
#   - Modes: row, col, flat, embed
#   - GainsMasterOptimizer also handles the no-gains case (mode=None), so a
#     single class can be registered in _EMERGING_OPTIMIZERS for both cases.

import logging
import math
from typing import Callable, Literal, Optional

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler

try:
    import emerging_optimizers
    from emerging_optimizers.orthogonalized_optimizers import (
        get_muon_scale_factor as _emerging_get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
except ImportError:
    emerging_optimizers = None


logger = logging.getLogger(__name__)


def _get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Muon orthogonalization scale factor.

    `shape_up` is master-specific (= max(d_out/d_in, d_in/d_out)**0.5); other
    modes delegate to the emerging_optimizers implementation.
    """
    if mode == "shape_up":
        return max(size_out / size_in, size_in / size_out) ** 0.5
    if mode == "none":
        return 1.0
    return _emerging_get_muon_scale_factor(size_out, size_in, mode=mode)


class MasterOptimizer(torch.optim.Optimizer):
    """AdamW / AdEMAMix + optional Muon orthogonalized updates + L2 hypersphere
    post-step weight clipping. See module docstring for the slim cut."""

    def __init__(
        self,
        params,
        # Common.
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # Adam / AdEMAMix.
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 0.0,
        beta3_warmup: Optional[int] = None,
        alpha_warmup: Optional[int] = None,
        eps: float = 1e-8,
        # Hypersphere (L2, post-step weight projection only).
        hypersphere_mode: Optional[Literal["row", "col", "flat", "embed"]] = None,
        hypersphere_embedding_mode: Optional[Literal["row", "col", "flat", "embed", "none"]] = None,
        hypersphere_router_mode: Optional[Literal["row", "col", "flat", "embed", "none"]] = None,
        hypersphere_eps: float = 1e-8,
        # Muown-style options (off by default).
        hypersphere_tangential_grad: bool = False,
        hypersphere_preserve_init: bool = False,
        # When True, scale the hypersphere target radius for is_out_proj
        # params (linear_proj, linear_fc2) by 1/sqrt(2 * num_layers) — matches
        # scaled_init_method_normal so the constraint preserves Megatron's
        # depth-aware init for residual-out projections.
        hypersphere_scale_out_proj_init: bool = False,
        num_layers: Optional[int] = None,
        # Muon (orthogonalized updates).
        use_orthogonal_updates: bool = False,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        split_qkv: bool = True,
        qkv_split_shapes: Optional[tuple[int, int, int]] = None,
        qkv_dim: Optional[int] = None,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        # NorMuon (per-row or per-col 2nd-moment rescale on the orthogonalized
        # update; arXiv 2510.05491). Reduces along the longer axis, so the
        # buffer lives along the shorter axis. No norm-preservation step.
        use_normuon: bool = False,
        normuon_beta2: float = 0.95,
        normuon_eps: float = 1e-8,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ):
        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_nesterov = use_nesterov

        self.use_normuon = use_normuon
        self.normuon_beta2 = normuon_beta2
        self.normuon_eps = normuon_eps

        self.hypersphere_mode = hypersphere_mode
        self.hypersphere_embedding_mode = hypersphere_embedding_mode
        self.hypersphere_router_mode = hypersphere_router_mode
        self.hypersphere_eps = hypersphere_eps
        self.hypersphere_tangential_grad = hypersphere_tangential_grad
        self.hypersphere_preserve_init = hypersphere_preserve_init
        if hypersphere_scale_out_proj_init:
            assert num_layers is not None and num_layers > 0, (
                "hypersphere_scale_out_proj_init=True requires num_layers"
            )
            self.out_proj_radius_scale = 1.0 / math.sqrt(2 * num_layers)
        else:
            self.out_proj_radius_scale = 1.0

        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn if is_qkv_fn is not None else (lambda p: False)
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_dim = qkv_dim

        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta1=betas[0],
            beta2=betas[1],
            beta3=betas[2],
            momentum_beta=momentum_beta,
            alpha=alpha,
            step=0,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            eps=eps,
            use_orthogonal_updates=use_orthogonal_updates,
        )
        super().__init__(params, defaults)

        # Ckpt-resume workaround: Megatron's _filter_and_reorder_param_groups
        # (optimizer.py) keys saved groups by (wd_mult, lr_mult,
        # is_expert_parallel, is_decoupled_lr). Master's overrides differ only
        # in max_lr / min_lr / use_orthogonal_updates / optimizer, so multiple
        # groups (matrix, embedding, LM-head, router, ...) collide on that
        # tuple. The loader's dict-based map then silently overwrites colliding
        # entries and reassigns the surviving group's `params` repeatedly,
        # which trips torch.optim.Optimizer.load_state_dict's per-group size
        # check ("loaded state dict contains a parameter group that doesn't
        # match the size of optimizer's group"). lr_mult is unused at runtime
        # (OptimizerParamScheduler reads max_lr/min_lr), so a unique tag per
        # group disambiguates the loader without touching training math.
        # Save/load round-trip works because both jobs run this same code in
        # the same param-group order.
        for _i, _g in enumerate(self.param_groups):
            _g['lr_mult'] = float(_i + 1)

        # Normalize parameters at init so the first forward sees on-sphere weights.
        # When hypersphere_preserve_init=True we skip this so the model's init
        # magnitude survives into training (Muown-style — gains absorb the
        # magnitude downstream in GainsMasterOptimizer._setup_gains).
        if (not self.hypersphere_preserve_init
                and (self.hypersphere_mode is not None
                     or self.hypersphere_embedding_mode is not None
                     or self.hypersphere_router_mode is not None)):
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.ndim != 2:
                            continue
                        is_qkv = self.is_qkv_fn(p)
                        is_out_proj = getattr(p, "is_out_proj", False)
                        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                        is_router = getattr(p, "is_router", False)
                        self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                        is_embedding=is_embedding, is_router=is_router)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            if "momentum_beta" not in group:  # Old checkpoint compat.
                group["momentum_beta"] = group["beta1"]
            for p in group["params"]:
                if p.grad is not None:
                    self._param_step(p, group)

        return loss

    def _param_step(self, p, group):
        grad = p.grad
        state = self.state[p]

        if "exp_avg" not in state:  # row_gain may already be present from gains.
            state["exp_avg"] = torch.zeros_like(grad)
            if not group["use_orthogonal_updates"]:
                if group["beta2"] != 0:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if group["alpha"] != 0:
                    state["exp_avg_slow"] = torch.zeros_like(grad)

        exp_avg = state["exp_avg"]
        beta1 = group["beta1"]
        momentum_beta = group["momentum_beta"]
        is_qkv = self.is_qkv_fn(p)
        is_out_proj = getattr(p, "is_out_proj", False)
        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
        is_router = getattr(p, "is_router", False)

        # Strip the radial component of grad before it feeds any momentum
        # buffer or 2nd-moment estimate (applies to both Muon and AdamW).
        if self.hypersphere_tangential_grad:
            self._project_tangent_inplace(
                p, grad, is_qkv=is_qkv, is_out_proj=is_out_proj,
                is_embedding=is_embedding, is_router=is_router,
            )

        if group["use_orthogonal_updates"]:  # Muon branch.
            assert emerging_optimizers is not None, (
                "emerging_optimizers package required for --use-orthogonal-updates"
            )
            self._apply_weight_decay_inplace(p, group)
            exp_avg.lerp_(grad, 1 - momentum_beta)
            if self.use_nesterov:
                grad = grad.lerp(exp_avg, momentum_beta)
            else:
                grad = exp_avg
            with emerging_optimizers.utils.fp32_matmul_precision(self.fp32_matmul_prec):
                update = self._orthogonalize_param(p, grad, is_qkv=is_qkv)
            if self.use_normuon:
                update = self._normuon_rescale(update, state)
            # Shrink Muon update for is_out_proj params to match the smaller
            # target sphere. Muon's shape_up (and spectral) scale targets the
            # natural RMS of a unit-row/col matrix; with target radius
            # 1/sqrt(2L) the bare update needs the same shrink factor.
            radius_scale = self._resolve_radius_scale(is_out_proj)
            if radius_scale != 1.0:
                update = update * radius_scale
        else:  # AdamW / AdEMAMix branch.
            beta2 = group["beta2"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            bias_correction1 = 1.0 - (beta1 ** group["step"])

            if beta2 == 0:
                if group["alpha"] == 0:  # plain SGD with momentum (no exp_avg_sq).
                    update = exp_avg / bias_correction1
                else:
                    alpha = (linear_warmup_scheduler(
                        group["step"], group["alpha"], 0, group["alpha_warmup"])
                        if group["alpha_warmup"] is not None else group["alpha"])
                    beta3 = (linear_hl_warmup_scheduler(
                        group["step"], group["beta3"], beta1, group["beta3_warmup"])
                        if group["beta3_warmup"] is not None else group["beta3"])
                    exp_avg_slow = state["exp_avg_slow"]
                    exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                    update = exp_avg / bias_correction1 + alpha * exp_avg_slow
            else:
                exp_avg_sq = state["exp_avg_sq"]
                bias_correction2 = 1.0 - (beta2 ** group["step"])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                if group["alpha"] == 0:  # Adam.
                    update = exp_avg.div(bias_correction1) / denom
                else:  # AdEMAMix.
                    alpha = (linear_warmup_scheduler(
                        group["step"], group["alpha"], 0, group["alpha_warmup"])
                        if group["alpha_warmup"] is not None else group["alpha"])
                    beta3 = (linear_hl_warmup_scheduler(
                        group["step"], group["beta3"], beta1, group["beta3_warmup"])
                        if group["beta3_warmup"] is not None else group["beta3"])
                    exp_avg_slow = state["exp_avg_slow"]
                    exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                    update = (exp_avg.div(bias_correction1) + alpha * exp_avg_slow) / denom

            self._apply_weight_decay_inplace(p, group)

        # Apply update.
        p.add_(update, alpha=-group["lr"])

        # Post-step hypersphere normalization (matrix clipping).
        if p.ndim == 2 and self._resolve_mode(is_embedding, is_router) is not None:
            self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj,
                            is_embedding=is_embedding, is_router=is_router)

    def _normuon_rescale(self, update, state):
        """Pure NorMuon (arXiv 2510.05491): 2nd-moment rescale of the
        orthogonalized update along the shorter axis. Reduces (averages
        squared values) along the longer axis; rescales by 1/sqrt(max(v, eps)).
        """
        avg_dim = -1 if update.shape[-2] >= update.shape[-1] else -2
        if "normuon_v" not in state:
            buf_shape = list(update.shape)
            buf_shape[avg_dim] = 1
            state["normuon_v"] = update.new_zeros(buf_shape)
        moment2 = state["normuon_v"]
        v_mean = update.square().mean(dim=avg_dim, keepdim=True)
        moment2.lerp_(v_mean, 1 - self.normuon_beta2)
        res = update * moment2.clamp_min(self.normuon_eps).rsqrt()

        # to reuse the same shape factors as muon, we do the following:
        # * we now that a semi-ortho matrix has frob norm of min(d_out, d_in) ** 0.5
        # * we divide by frob norm of the scaled matrix of normuon rescaled matrix to get 1
        # * we scale by min(d_out, d_in) ** 0.5 to get the same frob norm as the original matrix
        # * we can then apply other shape scaling factors as in muon

        # new norm
        vnorm_new = res.norm(dim=(-2, -1), keepdim=True).clamp_min(self.normuon_eps)
        shape_scaling = min(update.size(-2), update.size(-1)) ** 0.5
        # now we have a matrix with frob norm 1, we can apply other shape scaling factors as in muon
        res = res * shape_scaling / vnorm_new

        # apply other shape scaling factors as in muon
        scaling_factor = _get_muon_scale_factor(update.size(-2), update.size(-1), mode=self.scale_mode)
        return res * scaling_factor
 

    def _apply_weight_decay_inplace(self, p, group):
        weight_decay = group["weight_decay"]
        if weight_decay != 0:
            p.add_(p, alpha=-weight_decay * group["lr"])

    def _orthogonalize_param(self, p, grad, is_qkv: bool = False):
        """Newton-Schulz orthogonalization, with optional QKV split."""
        if self.pg_collection is not None:
            tp_group = (self.pg_collection.expt_tp
                        if getattr(p, "expert_tp", False)
                        else self.pg_collection.tp)
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and is_qkv:
            qs, ks, vs = _split_qkv(grad, self.qkv_split_shapes)
            qs = self._orthogonalize_tensor(qs, tp_group, partition_dim)
            ks = self._orthogonalize_tensor(ks, tp_group, partition_dim)
            vs = self._orthogonalize_tensor(vs, tp_group, partition_dim)
            return _merge_qkv((qs, ks, vs), grad.shape, self.qkv_split_shapes)
        return self._orthogonalize_tensor(grad, tp_group, partition_dim)

    def _orthogonalize_tensor(self, grad, tp_group, partition_dim):
        assert grad.ndim == 2
        size = [grad.size(-2), grad.size(-1)]
        if partition_dim is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        orth = newton_schulz_tp(
            grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=("duplicated" if self.tp_mode == "blockwise" else self.tp_mode),
        )
        scale = _get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        return orth * scale * self.extra_scale_factor

    def _resolve_mode(self, is_embedding: bool, is_router: bool = False):
        if is_router and self.hypersphere_router_mode is not None:
            mode = self.hypersphere_router_mode
        elif is_embedding and self.hypersphere_embedding_mode is not None:
            mode = self.hypersphere_embedding_mode
        else:
            mode = self.hypersphere_mode
        return None if mode == "none" else mode

    def _resolve_radius_scale(self, is_out_proj: bool) -> float:
        """Effective hypersphere radius scale for this param. Returns 1.0
        unless --hypersphere-scale-out-proj-init is on AND the param is
        is_out_proj. When gains+preserve_init are both active the gain absorbs
        the init magnitude (_maybe_init_gain_state), so we keep bare scale=1
        to avoid double-counting."""
        if not is_out_proj or self.out_proj_radius_scale == 1.0:
            return 1.0
        if (getattr(self, "hypersphere_gains_mode", None) is not None
                and self.hypersphere_preserve_init):
            return 1.0
        return self.out_proj_radius_scale

    def _project_tangent_inplace(self, p, grad, is_qkv: bool = False, is_out_proj: bool = False,
                                  is_embedding: bool = False, is_router: bool = False):
        """In-place: remove the radial component of `grad` w.r.t. the hypersphere
        mode at `p`. Mirrors _normalize's QKV-split layout so the constraint
        matches the post-step projection. Muown's grad_v construction."""
        mode = self._resolve_mode(is_embedding, is_router)
        if mode is None:
            return
        if is_qkv and self.split_qkv:
            ps = _split_qkv(p, self.qkv_split_shapes)
            gs = _split_qkv(grad, self.qkv_split_shapes)
            for pi, gi in zip(ps, gs):
                self._project_tangent_single_(pi, gi, is_out_proj, mode)
            grad.copy_(_merge_qkv(gs, grad.size(), self.qkv_split_shapes))
            return
        self._project_tangent_single_(p, grad, is_out_proj, mode)

    def _project_tangent_single_(self, p, grad, is_out_proj: bool, mode: str):
        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "embed":
            dim = 0 if is_out_proj else 1
        elif mode == "flat":
            dim = None
        else:
            return
        if dim is None:
            p_norm_sq = (p * p).sum().clamp_min(self.hypersphere_eps)
            radial = (p * grad).sum() / p_norm_sq
        else:
            p_norm_sq = (p * p).sum(dim=dim, keepdim=True).clamp_min(self.hypersphere_eps)
            radial = (p * grad).sum(dim=dim, keepdim=True) / p_norm_sq
        grad.sub_(p * radial)

    def _normalize(self, p, x, is_qkv: bool = False, is_out_proj: bool = False,
                   is_embedding: bool = False, is_router: bool = False):
        """In-place L2-sphere projection of a 2D tensor `x` (sized like `p`).

        For QKV-merged weights, normalize each of Q/K/V separately. Modes:
            row    → unit-norm each output row     (dim=1)
            col    → unit-norm each input column   (dim=0)
            flat   → unit Frobenius then scale by sqrt(max(d_out, d_in))
            embed  → row for non-out_proj, col for out_proj
        """
        mode = self._resolve_mode(is_embedding, is_router)
        if mode is None:
            return

        radius_scale = self._resolve_radius_scale(is_out_proj)

        if is_qkv and self.split_qkv:
            qs, ks, vs = _split_qkv(x, self.qkv_split_shapes)
            self._normalize_single(qs, is_out_proj, mode, radius_scale)
            self._normalize_single(ks, is_out_proj, mode, radius_scale)
            self._normalize_single(vs, is_out_proj, mode, radius_scale)
            x.copy_(_merge_qkv((qs, ks, vs), x.size(), self.qkv_split_shapes))
            return

        self._normalize_single(x, is_out_proj, mode, radius_scale)

    def _normalize_single(self, x, is_out_proj: bool, mode: str, radius_scale: float = 1.0):
        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "flat":
            dim = None
        elif mode == "embed":
            dim = 0 if is_out_proj else 1
        else:
            raise ValueError(f"Unsupported hypersphere mode: {mode}")
        norm = torch.norm(x, dim=dim, keepdim=True).clamp_min(self.hypersphere_eps)
        x.div_(norm)
        if mode == "flat":
            x.mul_(max(x.size(-2), x.size(-1)) ** 0.5)
        if radius_scale != 1.0:
            x.mul_(radius_scale)
        # row_l2_after = x.norm(dim=-1).mean()
        # col_l2_after = x.norm(dim=-2).mean()
        # frob_norm = x.norm()
        # print(f"Hypersphere normalization: {x.shape} | row_norm avg: {row_l2_after.item():.5f}, col_norm avg: {col_l2_after.item():.5f}, frob_norm: {frob_norm.item():.5f}")


class GainsMasterOptimizer(MasterOptimizer):
    """MasterOptimizer with learnable per-axis gains as a fused
    reparameterization (p = normalized_w * gains).

    Setting `hypersphere_gains_mode=None` makes this class behave identically
    to plain `MasterOptimizer` — no gain tensors are created and the gain hooks
    in step() are no-ops. This lets a single class cover both cases for
    registry registration.

    Gain optimizer state (1st/2nd moments) is tracked manually as plain tensors
    in `self.state[p]`, like Muown's magnitude state. This avoids registering
    gain `nn.Parameter`s in an inner `AdamW` that lives outside `self.param_groups`
    — torch.optim's state_dict id-mapping doesn't survive that, which is what
    breaks `torch.save(super().state_dict(), ...)`. Gain tensors have a
    different shape than `p`, so use `--ckpt-format torch` (same caveat as Muown).
    """

    def __init__(
        self,
        params,
        hypersphere_gains_mode: Optional[Literal["row", "col", "rowcol", "flat", "embed"]] = None,
        hypersphere_gains_mode_output: Optional[Literal["row", "col", "rowcol", "flat", "none"]] = None,
        hypersphere_gains_mode_embedding: Optional[Literal["row", "col", "rowcol", "flat", "none"]] = None,
        gains_lr: Optional[float] = None,
        gains_betas: tuple[float, float] = (0.9, 0.999),
        gains_eps: float = 1e-8,
        gains_weight_decay: float = 0.0,
        # Reparametrize the per-axis gain: the stored state tensor is `g`, the
        # effective multiplier applied to `p` is `phi(g)`. "direct" reproduces
        # the original behavior (phi(g)=g). Applied uniformly to row/col/flat.
        gain_parametrization: Literal["direct", "offset", "softplus"] = "direct",
        **kwargs,
    ):
        self.hypersphere_gains_mode = hypersphere_gains_mode
        self.hypersphere_gains_mode_output = hypersphere_gains_mode_output
        self.hypersphere_gains_mode_embedding = hypersphere_gains_mode_embedding
        self.gains_lr = gains_lr  # None → follow group["lr"] verbatim
        self.gains_betas = gains_betas
        self.gains_eps = gains_eps
        self.gains_weight_decay = gains_weight_decay
        self.gain_parametrization = gain_parametrization
        super().__init__(params, **kwargs)
        # Gain state is initialized lazily at first step (see step() →
        # _maybe_init_gain_state). Eager init here would write entries into
        # self.state keyed by the bf16 model param, but with
        # --use-distributed-optimizer the layer-wise wrapper later (1) shards
        # param_groups (dropping params owned by other ranks) and then (2)
        # clones bf16 → fp32 main_params via Float16OptimizerWithFloat16Params,
        # which migrates self.state[bf16_p] → self.state[main_p] only for
        # params still in param_groups on this rank. Entries for params
        # sharded onto OTHER ranks become orphan tensor keys in self.state,
        # which then blows up torch.optim.Optimizer.state_dict() with
        # `KeyError: <id>` at save time (param_mappings is built from
        # self.param_groups, so the orphans don't map). Deferring to step()
        # mirrors Muown's approach: by then param_groups holds the final
        # post-shard main_params and the keys stay consistent.

    def _phi(self, g: torch.Tensor) -> torch.Tensor:
        """Forward map from raw gain g to effective multiplier phi(g)."""
        mode = self.gain_parametrization
        if mode == "direct":
            return g
        if mode == "offset":
            return 1.0 + g
        if mode == "softplus":
            return torch.nn.functional.softplus(g)
        raise ValueError(f"Unknown gain_parametrization {mode}")

    def _phi_prime(self, g: torch.Tensor) -> torch.Tensor | float:
        """Derivative phi'(g). Returns scalar 1.0 for the linear modes so the
        caller can skip a multiply."""
        mode = self.gain_parametrization
        if mode in ("direct", "offset"):
            return 1.0
        if mode == "softplus":
            return torch.sigmoid(g)
        raise ValueError(f"Unknown gain_parametrization {mode}")

    def _phi_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse map: given a target effective multiplier x>0, return g s.t.
        phi(g) = x. Used at init to seed the raw gain so the first step's
        effective multiplier matches the desired target (1.0 in the identity
        case, ||p[i]|| / cur_frob in the preserve_init absorb case)."""
        mode = self.gain_parametrization
        if mode == "direct":
            return x
        if mode == "offset":
            return x - 1.0
        if mode == "softplus":
            # Stable softplus_inv for x > 0: g = x + log1p(-exp(-x)).
            # As x → ∞, g → x. As x → 0+, g → −∞.
            return x + torch.log1p(-torch.exp(-x))
        raise ValueError(f"Unknown gain_parametrization {mode}")

    @torch.no_grad()
    def _maybe_init_gain_state(self, p):
        if self.hypersphere_gains_mode is None or p.ndim < 2:
            return
        mode = self._resolve_gains_mode(p)
        if mode is None or mode == "none":
            return
        state = self.state[p]
        if any(k in state for k in ("row_gain", "col_gain", "flat_gain")):
            return
        is_out_proj = getattr(p, "is_out_proj", False)
        preserve = self.hypersphere_preserve_init

        wants_row = ("row" in mode) or (mode == "embed" and not is_out_proj)
        wants_col = ("col" in mode) or (mode == "embed" and is_out_proj)
        wants_flat = mode == "flat"

        # For preserve_init, absorb the original magnitude into exactly one
        # axis (row wins when both are configured — the rowcol canonical
        # decomposition leaves col_gain at 1). Other gain axes start at 1.
        # We do NOT touch p.data: like Muown's `g = ||W[i]||`, the
        # decomposition is implicit at init — p still holds the original
        # weight, and the next step's _preprocess_gains will divide by
        # gain_init to recover bare_p before updating. This way the first
        # forward sees the model's original init.
        absorb_axis = None
        if preserve:
            if wants_row:
                absorb_axis = "row"
            elif wants_col:
                absorb_axis = "col"
            elif wants_flat:
                absorb_axis = "flat"

        # The stored tensor is the raw gain `g`; the effective multiplier is
        # phi(g). For the identity branch we want phi(g)=1 → seed phi_inv(1).
        # For the preserve_init absorb branch we want phi(g)=||p[i]|| (or
        # cur_frob/target_frob) → seed phi_inv of that.
        if wants_row:
            if absorb_axis == "row":
                target = (p.detach().norm(dim=1).to(torch.float32)
                          .clamp_min(self.hypersphere_eps))
            else:
                target = torch.ones(p.size(0), dtype=torch.float32, device=p.device)
            state["row_gain"] = self._phi_inv(target)
            state["row_gain_m"] = torch.zeros_like(state["row_gain"])
            state["row_gain_v"] = torch.zeros_like(state["row_gain"])

        if wants_col:
            if absorb_axis == "col":
                target = (p.detach().norm(dim=0).to(torch.float32)
                          .clamp_min(self.hypersphere_eps))
            else:
                target = torch.ones(p.size(1), dtype=torch.float32, device=p.device)
            state["col_gain"] = self._phi_inv(target)
            state["col_gain_m"] = torch.zeros_like(state["col_gain"])
            state["col_gain_v"] = torch.zeros_like(state["col_gain"])

        if wants_flat:
            if absorb_axis == "flat":
                target_frob = max(p.size(-2), p.size(-1)) ** 0.5
                cur_frob = (p.detach().norm().to(torch.float32)
                            .clamp_min(self.hypersphere_eps))
                target = (cur_frob / target_frob).reshape(())
            else:
                target = torch.ones((), dtype=torch.float32, device=p.device)
            state["flat_gain"] = self._phi_inv(target)
            state["flat_gain_m"] = torch.zeros_like(state["flat_gain"])
            state["flat_gain_v"] = torch.zeros_like(state["flat_gain"])

    @torch.no_grad()
    def step(self, closure=None):
        if self.hypersphere_gains_mode is None:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            if "momentum_beta" not in group:
                group["momentum_beta"] = group["beta1"]
            for p in group["params"]:
                if p.grad is not None:
                    self._maybe_init_gain_state(p)
                    gain_grads = self._preprocess_gains(p)
                    self._param_step(p, group)
                    self._gains_step(p, group, gain_grads)
                    self._apply_gains(p)

        return loss

    @torch.no_grad()
    def _preprocess_gains(self, p):
        state = self.state[p]
        eps = 1e-8
        flat = state.get("flat_gain")
        row = state.get("row_gain")
        col = state.get("col_gain")
        if flat is None and row is None and col is None:
            return None

        # Effective per-axis multipliers phi(g). For "direct" these are
        # identical to the stored raw gains (no-op overhead).
        flat_eff = self._phi(flat) if flat is not None else None
        row_eff = self._phi(row) if row is not None else None
        col_eff = self._phi(col) if col is not None else None

        # Undo gains to recover bare normalized weight.
        if flat_eff is not None:
            p.div_(flat_eff.clamp_min(eps))
        if row_eff is not None:
            p.div_(row_eff[:, None].clamp_min(eps))
        if col_eff is not None:
            p.div_(col_eff[None, :].clamp_min(eps))

        # Compute ∂L/∂phi(g) from bare p and the gain-baked grad still on p.grad.
        p_times_pgrad = p * p.grad
        gain_grads = {}
        if flat is not None:
            assert row is None and col is None
            gain_grads["flat_gain"] = torch.sum(p_times_pgrad)
        elif row is not None and col is not None:
            gain_grads["row_gain"] = torch.sum(p_times_pgrad * col_eff[None, :], dim=1)
            gain_grads["col_gain"] = torch.sum(p_times_pgrad * row_eff[:, None], dim=0)
        elif row is not None:
            gain_grads["row_gain"] = torch.sum(p_times_pgrad, dim=1)
        else:
            gain_grads["col_gain"] = torch.sum(p_times_pgrad, dim=0)

        # Chain rule: ∂L/∂g = phi'(g) · ∂L/∂phi(g). For "direct"/"offset"
        # phi'≡1 and _phi_prime returns the scalar 1.0; skip the multiply.
        if self.gain_parametrization not in ("direct", "offset"):
            for name in gain_grads:
                gain_grads[name] = gain_grads[name] * self._phi_prime(state[name])

        # Rescale p.grad so MasterOptimizer's step sees ∂L/∂(bare p).
        if flat_eff is not None:
            p.grad.mul_(flat_eff)
        if row_eff is not None:
            p.grad.mul_(row_eff[:, None])
        if col_eff is not None:
            p.grad.mul_(col_eff[None, :])

        return gain_grads

    @torch.no_grad()
    def _gains_step(self, p, group, gain_grads):
        if not gain_grads:
            return
        state = self.state[p]
        step = group["step"]
        beta1, beta2 = self.gains_betas
        eps = self.gains_eps
        wd = self.gains_weight_decay

        # Gains follow the param-group LR. If gains_lr is unset, use group["lr"]
        # verbatim (so the param scheduler drives gains too). Otherwise treat
        # gains_lr as a base LR and apply the param schedule shape on top.
        if self.gains_lr is None:
            lr = group["lr"]
        else:
            max_lr = group.get("max_lr", group["lr"])
            schedule_mult = (group["lr"] / max_lr) if max_lr > 0 else 1.0
            lr = self.gains_lr * schedule_mult

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        for name, grad in gain_grads.items():
            gain = state[name]
            m = state[f"{name}_m"]
            v = state[f"{name}_v"]
            if wd != 0.0:
                gain.mul_(1.0 - lr * wd)
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            gain.addcdiv_(m, denom, value=-lr / bias_correction1)

    @torch.no_grad()
    def _apply_gains(self, p):
        state = self.state[p]
        flat = state.get("flat_gain")
        row = state.get("row_gain")
        col = state.get("col_gain")
        if flat is not None:
            p.mul_(self._phi(flat))
        if row is not None:
            p.mul_(self._phi(row)[:, None])
        if col is not None:
            p.mul_(self._phi(col)[None, :])

    def _resolve_gains_mode(self, p):
        is_output = getattr(p, "is_output_parameter", False)
        is_embedding = getattr(p, "is_embedding_parameter", False)
        if is_output and self.hypersphere_gains_mode_output is not None:
            return self.hypersphere_gains_mode_output
        if is_embedding and self.hypersphere_gains_mode_embedding is not None:
            return self.hypersphere_gains_mode_embedding
        return self.hypersphere_gains_mode


def _split_qkv(x, shapes: tuple[int, int, int]) -> list[torch.Tensor]:
    """Split grouped attention (Q, K, V / GQA) along the head-group dim."""
    shape = x.shape
    num_query_groups = shape[0] // sum(shapes)
    qkv = torch.split(
        x.view(num_query_groups, sum(shapes), -1),
        shapes,
        dim=1,
    )
    return [g.reshape(-1, shape[-1]) for g in qkv]


def _merge_qkv(qkv, xshape: tuple[int, int], shapes: tuple[int, int, int]) -> torch.Tensor:
    num_query_groups = xshape[0] // sum(shapes)
    qkv = [g.view(num_query_groups, -1, xshape[-1]) for g in qkv]
    return torch.cat(qkv, dim=1).view(xshape)
