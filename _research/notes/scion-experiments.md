# Scion optimizer — 350M-ablation experiment log

Scion ([Pethick et al., 2025](https://arxiv.org/abs/2502.07529)) is a
norm-constrained Frank-Wolfe / LMO optimizer: per 2-D parameter it does a
momentum EMA, passes it through a per-layer norm oracle (LMO), and takes a
constrained Frank-Wolfe step that keeps the weight inside a radius-ρ norm-ball.
Muon's Newton-Schulz orthogonalization is the *Spectral* special case of that
oracle; Scion generalizes it to per-layer norms and folds weight decay into the
constraint.

This is the experiment log behind the `scion-*` leaderboard entries (ranks 1, 3,
8). The full design + faithful-port rationale lives in the repo-local `SCION.md`;
this file is the digestible results summary that ships with the PR.

## The recipe (fixed across every run)

One `TensorParallelScion(OrthogonalizedOptimizer)` dispatches on a per-group
`norm` string over three oracles:

| layer | oracle | `lmo(g)` | radius ρ |
|---|---|---|---|
| hidden matrices (qkv, proj, fc1, fc2, **router, experts**) | Spectral | `NS5(g)·√(d_out/d_in)` | 50 |
| output head + token embedding (untied) | Sign | `(1/d_in)·sign(g)` | 3000 |
| RMSNorm gains (fused + final) | BiasRMS | `g/√(mean(g²))` | 50 |

- **Weight decay *is* the constraint.** Scion runs decoupled WD with μ=1 (the
  Frank-Wolfe radius), pinned as an instance attribute and protected from the
  scheduler — the `--weight-decay 0.1` CLI flag is ignored by Scion. (Megatron's
  scheduler overwrites `group['weight_decay']` every step; left unprotected that
  would silently rescale the radius.)
- Radii are **sourced**, not tuned: hidden ρ=50 and head ρ=3000 come from
  modded-nanogpt (`scale=50`, `last_scale=3000`); the untied embedding reuses the
  head's Sign@3000 (mirrors the tied recipe where the embedding *is* the head),
  norm gains reuse hidden@50. Row/ColNorm oracles are not needed and not ported.
- Newton-Schulz: 5-step **quintic** (Muon's tuple), bf16 — identical NS to Muon
  for an apples-to-apples comparison. μ=1, momentum 0.9.
- TP: Spectral is the only oracle needing cross-rank comm (it already exists via
  `newton_schulz_tp`); Sign/BiasRMS are elementwise/replicated. All runs here are
  TP=1, where everything is a no-op. (TP=2 `duplicated` is bitwise-exact vs
  single-device; `blockwise` is the comm-free per-shard approximation.)

Shared with the rest of the leaderboard: 350M-active Transformer++ (24L / 1024H /
2560F / 16h / 4kv GQA), SwiGLU, RMSNorm, RoPE, 1B tokens ClimbMix, GBS=128, seq
4096, WSD schedule, bf16, seed 42, 1 GH200 node.

## Lineage & headline result

| entry | change | optimizer | γ | final | min | rank |
|---|---|---|---:|---:|---:|---:|
| `scion-lr3e-4` | root: Scion (norm-LMO FW) | Scion | 3e-4 | 2.196 | 2.036 | 8 |
| `scion-qkn-lr3e-4` | + `--qk-layernorm` | Scion + QKN | 3e-4 | 2.167 | 2.005 | 3 |
| `scion-qkn-moe-32e-tk3-sh1-lr3e-4` | + 32e-tk3 + 1 shared MoE | Scion + QKN + MoE | 3e-4 | **2.078** | **1.913** | **1** |

- The **γ=3e-4 optimum holds across all three variants** — adding QKN then MoE
  does not move the LR sweet spot.
- **Each step is a clean monotone gain**; MoE is the single biggest jump
  (−0.089 final), matching the leaderboard's dense→MoE Aurora gain (−0.087).
- The MoE winner **beats the prior leader `aurora-qkn-moe`** (2.098 / 1.934) by
  **−0.020 final / −0.021 min**; the dense `scion-qkn` ties `aurora-qkn-xsa` on
  final (2.167) and wins on min (2.005 < 2.009) with no attention change.

## Ablation 1 — base LR sweep (dense Scion, no QKN)

12-point γ sweep; the table shows the final round. γ ≥ 1e-3 diverges (ρ=50 makes
the literal Muon/Aurora LRs ~50× too aggressive): the Frank-Wolfe constraint
bounds each layer's operator norm but does not, by itself, stop loss divergence
when the per-step move is too large. Divergence onset ≈ 5e-3.

| γ | 6e-5 | 1e-4 | 2e-4 | **3e-4** | 4e-4 | 6e-4 | 8e-4 | ≥1e-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| final | 2.305 | 2.242 | 2.199 | **2.196** | 2.204 | 2.225 | 2.266 | diverged |
| min | 2.140 | 2.079 | 2.037 | **2.036** | 2.044 | 2.067 | 2.108 | — |

Sweet spot γ ∈ {2e-4, 3e-4, 4e-4}, tied within noise; 3e-4 best.

## Ablation 2 — QK-norm (dense + `--qk-layernorm`)

| γ | final | min | ΔQKN final (vs no-QKN base) |
|---|---:|---:|---:|
| 2e-4 | 2.175 | 2.013 | −0.022 |
| **3e-4** | **2.167** | **2.005** | **−0.028** |
| 4e-4 | 2.168 | 2.010 | −0.036 |

QK-norm helps at every γ. The matched no-QKN column reproduces the base sweep at
the same rev/seed → also a determinism check.

## Ablation 3 — MoE (Scion + QKN + 32e-tk3 + 1 shared)

| γ | final | min |
|---|---:|---:|
| 2e-4 | 2.086 | 1.922 |
| **3e-4** | **2.078** | **1.913** |

Both LRs beat the prior leader; γ=3e-4 holds. Only 2 LRs run since the dense
sweep already localized the optimum. Clean to the end (0 NaN, load-balancing ≈
1.005, params_norm bounded).

### Why the MoE experts are orthogonalized, not Adam-routed

The `aurora-qkn-moe` leaderboard note says 3-D expert weights "bypass the 2-D
predicate and fall through to AdamW." **That is not true for this fork's TE
path.** Traced end-to-end:

- **Storage.** `--moe-grouped-gemm` + TE → `TEGroupedMLP` →
  `TEGroupedLinear(te.pytorch.GroupedLinear)`, which registers **one 2-D
  `nn.Parameter` per local expert**, named `…experts.linear_fc1.weight{i}` /
  `…linear_fc2.weight{i}`. There is no stacked 3-D `[E,m,n]` tensor anywhere in
  this fork.
- **Routing.** Scion's hidden-matrix rule is a *shape* predicate (`ndim==2`,
  not a name glob), so the numeric `weight{i}` suffix is irrelevant — each
  per-expert matrix, the router `[E,H]`, and the shared expert all → Spectral@50.
- **Fail-loud.** Scion's `default_param_overrides` is empty (no matrix→Adam
  escape, unlike Muon/Aurora). A hypothetical stacked 3-D expert tensor would
  match no rule, reach `orthogonalize()` with `norm=None`, and hard-`ValueError`
  on step 1 — it would never silently misroute to Adam.

So under this recipe the MoE expert matrices are **fully orthogonalized** by
Scion. (The Aurora row's iso-active numbers stand; only the *mechanism* it
describes is wrong for our path. We left the Aurora row untouched.)

## Throughput / cost

Steady-state median: Scion **1027 ms/iter, ~309 TFLOP/s/GPU** (dense) — fastest
of the matrix optimizers (AdamW 271, Muon 264, Aurora 248). Per-iter cost is flat
across all 7 LRs (~3% spread). Newton-Schulz on small 2-D weight matrices is
nearly free (~3.5 ms / extra step, ~0.3% of step time), so Scion has ~255 ms/iter
of headroom vs Aurora's 12-step NS. Iteration-time spikes are a front-loaded
startup artifact (kernel autotune / warmup), not optimizer-intrinsic: in the
steady tail Scion's p99/p50 = 1.24, tighter than Muon (1.36). The MoE run is
~124 TFLOP/s/GPU (small-model per-token all-to-all dispatch penalty, same as the
Aurora MoE; vanishes at hidden ≥ 4K).

## Open follow-ups (not in these PRs)

- **Orthogonality-quality ablation:** `--scion-coefficient-type polar_express
  --scion-num-ns-steps 8–12` (monotone σ→1, ≈Aurora-grade) vs the current
  quintic@5, at the winning γ — cheap (NS is ~free), isolates whether better
  orthogonalization improves loss. (Naively raising `--scion-num-ns-steps` under
  `quintic` is meaningless — the package *cycles* the aggressive early tuples.)
- **Radius sweep** if results plateau (start with ρ_head / the head:hidden ratio).
- **SODA mode**: the init-centered `1/(k+2)` averaging variant as an alternative
  to the fixed FW constraint.

## Reproduce

```
source .env.pgd
sbatch _research/leaderboards/350m-ablation/runs/18-scion-lr3e-4.sbatch                  # root
sbatch _research/leaderboards/350m-ablation/runs/19-scion-qkn-lr3e-4.sbatch              # + QKN
sbatch _research/leaderboards/350m-ablation/runs/20-scion-qkn-moe-32e-tk3-sh1-lr3e-4.sbatch  # + MoE (leader)
```

Each frozen sbatch pins the git sha and W&B URL in its header. Editable templates
live at `_research/launch/transformer-pp-350m-ablation-scion{,-qkn,-qkn-moe-32e-tk3-sh1}.sbatch`.
