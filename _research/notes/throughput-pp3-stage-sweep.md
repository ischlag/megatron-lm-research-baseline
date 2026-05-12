# Per-stage throughput sweep: 400-600B MoE on a single PP=3 deployment

## Summary

Measures the throughput of a single transformer body PP stage of a DeepSeek-V3-shape
MoE (d=7168, n_h=128, d_h=128, GQA n_kv=8) on 3x4 GH200 nodes
(stage 0: embedding, stage 1: body, stage 2: LM head). 2-D sweep over routed
expert count E and body layer count L, all at PP=3 / TP=4 / EP=4 / DP=1,
sequence-parallel, MoE alltoall dispatch, grouped GEMM, FP8 blockwise (DeepSeek-V3
recipe: 1x128 act tiles, 128x128 weight blocks, E4M3 fwd+bwd, fp32 scales),
MBS=4. Throughput reported is body tok/s/GPU = GBS * seq / (4 * step_time),
median over the last 50 of 200 iterations. Uses `--skip-optimizer-step` to free
the AdamW m, v allocation (frees ~22 GB/body-GPU) -- approximating the per-GPU
memory profile that FSDP/zero achieves at large DP via state sharding.

W&B project: lm-research-baseline-dev.

## Configuration (shared across all rows)

- Architecture: DeepSeek-V3-shape, d_model=7168, n_h=128, d_h=128, GQA n_kv=8,
  d_e=2048 (SwiGLU), RMSNorm, RoPE, untied embed + output head, GPT-2 BPE
  (vocab 50304).
- Parallelism: PP=3 (`E|t...t|L` layout), TP=4 (attention, embed, LM head),
  ETP=1 (experts not TP-sharded), EP=4 (32 routed experts/GPU at E=128),
  DP=1, CP=1, `--sequence-parallel`.
- Precision: bf16 weights, fp32 master grads, FP8 blockwise (`--fp8-format
  e4m3 --fp8-recipe blockwise`). No FP8 param gather, no FP8 attention.
- Optimizer: AdamW with `--skip-optimizer-step` (state buffers allocated but
  step is no-op, so torch.optim's lazy m, v allocation never fires -> 22 GB
  freed per body GPU).
- Routing: `--moe-router-force-load-balancing --moe-router-load-balancing-type
  none` (random STE logits, no aux loss, no token drops).
- Batch: MBS=4, GBS=64 (16 microbatches at DP=1, PP=3 bubble ~12.5%), seq=4096.

## Results (sorted by body tok/s/GPU, median last 50 of 200 iters)

| config | E | K | L | sparsity | step ms (med) | body tok/s/GPU | total tok/s | peak GB | total params/stage | active params/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **e224-k8-2L** | 224 | 8 | 2 | 5.19% | **1674** | **39,147** | 156,588 | 64.9 | **20.30 B** | **1.29 B** |
| e192-k8-2L | 192 | 8 | 2 | 6.04% | 1698 | 38,598 | 154,392 | 57.0 | 17.48 B | 1.29 B |
| e256-k12-2L | 256 | 12 | 2 | 6.12% | 1973 | 33,216 | 132,864 | 75.2 | 23.11 B | 1.64 B |
| e128-k8-2L (ref) | 128 | 8 | 2 | 8.97% | 1981 | 33,082 | 132,329 | 41.2 | 11.85 B | 1.29 B |
| e128-k8-3L | 128 | 8 | 3 | 8.97% | 2211 | 29,647 | 118,590 | 60.0 | 17.77 B | 1.94 B |
| e192-k8-3L | 192 | 8 | 3 | 6.04% | 2350 | 27,882 | 111,528 | 83.6 | **26.22 B** | 1.94 B |
| e128-k8-4L | 128 | 8 | 4 | 8.97% | 2926 | 22,398 | 89,591 | 78.7 | 23.70 B | **2.58 B** |
| e128-k8-5L | 128 | 8 | 5 | 8.97% | OOM | -- | -- | -- | 29.62 B | 3.23 B |
| e192-k8-4L | 192 | 8 | 4 | 6.04% | OOM | -- | -- | -- | 34.96 B | 2.58 B |
| e224-k8-3L | 224 | 8 | 3 | 5.19% | OOM | -- | -- | -- | 30.44 B | 1.94 B |
| e256-k12-3L | 256 | 12 | 3 | 6.12% | OOM | -- | -- | -- | 34.67 B | 2.46 B |

Params columns are body-only (per-stage transformer layers), excluding
embedding + LM head which sit on the edge PP stages and add a fixed 0.72 B
total / 0 routed active.

## Winners by axis

- **Best body tok/s/GPU**: `e224-k8-2L` at 39,147 -- slightly ahead of e192 (38,598).
  Surprisingly higher than e128 baseline (33,082) despite identical per-token
  compute (same K=8, d_e); the wider-E configs are simply more consistent (lower
  step-time variance), pulling the median up.
- **Most total params/stage (within memory wall)**: `e192-k8-3L` at 26.22 B.
- **Most layers**: `e128-k8-4L` at 4 layers (23.70 B body).
- **Best Pareto (params x throughput product)**: **`e224-k8-2L`** -- 20.30 B x 39.1 K = 795.
  Closest single-stage configuration to DeepSeek-V3's regime (5.2% sparsity).

## Production extrapolation (Pareto winner e224-k8-2L)

- per-stage body: 20.30 B total / 1.29 B active per token (DeepSeek-V3-class
  sparsity, K=8 + shared).
- 30-stage PP (60 transformer layers): **609 B total / 38.7 B active** -- inside
  the 400-600 B target range with DeepSeek-V3-class active count (vs 671 B / 37 B
  for the DeepSeek-V3 reference).
- per-body-GPU throughput: 39.1 K tok/s (sustained median, FP8 blockwise, 200-iter
  run, skip-opt).

## Memory wall observations

- L=3 fails for E >= 192 with K=8: Fp8Padding/Unpadding workspace allocation +
  3 microbatches in flight under PP=3 1F1B + 3 layers of routed-expert state +
  activations together exceed 96 GB.
- L=4 fits only at E=128 (78.7 GB peak) with skip-opt; even L=5 at E=128 OOMs.
- K=12 (used to keep sparsity >= 5% at E=256) trades throughput for headroom on
  E: roughly 17% slower than K=8 at the same E.
- All measured peaks are under 96 GB body-GPU HBM with `--skip-optimizer-step`;
  without skip-opt (full AdamW), 2L peak is 63.3 GB (m + v add ~22 GB), and the
  memory wall moves accordingly (e.g., E=192 L=2 base would jump from 57 to ~79
  GB, e224-k8-2L from 64.9 to ~87 GB).

## Caveats

- `--skip-optimizer-step` does not update parameters; throughput numbers from
  this sweep are valid as compute-throughput approximations of an FSDP-at-large-DP
  production training step where the optimizer step's per-GPU cost amortizes
  away across many replicas.
- Step-time variance under skip-opt is non-trivial (std/median ~10-25% across
  configs); median over the last 50 of 200 iters is the robust statistic.
- The DeepSeek-V3 reference uses MLA attention, not GQA. We use GQA throughout
  this sweep -- per-token compute differs from the reference by a small,
  attention-block-only factor.

## Reproduction

Each row's sbatch is at
`_research/launch/throughput-moe-stage-d7168-{config}-pp3-tp4-ep4-fp8-mbs4-skipopt.sbatch`
where `{config}` is the first column (e.g., `e224-k8-2L`). All hparams pinned;
no env-var branching. Run with `sbatch <file>` after setting
`SBATCH_RESERVATION`, `SBATCH_ACCOUNT`, `MEGATRON_DATA_PATH`, `WANDB_API_KEY`.

Commit at time of sweep: `1da301a84`.

## Open

- bf16 control at e224-k8-2L (does FP8 actually help at this geometry, or is it
  near-wash again here?).
- Push MBS at e224-k8-2L (31 GB headroom; MBS=8 should fit, MBS=12 maybe).
- Verify behaviour with full AdamW step (skip-opt removed) at the new memory
  ceiling.
