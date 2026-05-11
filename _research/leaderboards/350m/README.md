# 350M Leaderboard

Full-baseline 350M dense Transformer++ (24L / 1024H / 2560F / 16h / 4kv GQA),
**15B tokens** on ClimbMix, GBS=128 (524K tokens/step), seq_len=4096, 2 GH200
nodes (DP=8), ~5h/run, WSD schedule (3000-step warmup, last 20% decay).

Recipes carry over from the 1B
[`350m-ablation`](../350m-ablation/README.md) leaderboard (same slugs); only
schedule and node count are scaled. Same seed (42), same data, same
architecture; only the optimizer block differs. W&B project:
[megatron-lm-research-baseline](https://wandb.ai/ischlag/megatron-lm-research-baseline).

The `entry` slug is the stable identifier (sbatch filename minus the numeric
prefix). `parent` references the entry this row builds on; `change` is the
one-line delta. `rank` shifts as new entries land — slug + parent + change is
the durable record.

| rank | entry | parent | change | optimizer | matrix LR | final | min | sbatch | wandb | commit |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | normuon-lr3.6e-4 |  | root: NorMuon recipe (per-row 2nd moment), matrix LR 3.6e-4 | NorMuon | 3.6e-4 | **1.775** | **1.647** | [`02-normuon-lr3.6e-4.sbatch`](runs/02-normuon-lr3.6e-4.sbatch) | [coe7k0xi](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/coe7k0xi) | [`9e2134654`](https://github.com/ischlag/megatron-lm-research-baseline/tree/9e2134654) |
| 2 | adamw-lr1e-3 |  | root: AdamW (decoupled WD), LR 1e-3 | AdamW | 1e-3 | 1.792 | 1.668 | [`01-adamw-lr1e-3.sbatch`](runs/01-adamw-lr1e-3.sbatch) | [p8g05viv](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/p8g05viv) | [`9e2134654`](https://github.com/ischlag/megatron-lm-research-baseline/tree/9e2134654) |
| 3 | aurora-qkn | aurora-lr1e-2 | add `--qk-layernorm` (RMSNorm on Q, K) | Aurora + QK norm | 1e-2 | 1.802 | 1.693 | [`04-aurora-qkn.sbatch`](runs/04-aurora-qkn.sbatch) | [0zg55qes](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/0zg55qes) | [`9e2134654`](https://github.com/ischlag/megatron-lm-research-baseline/tree/9e2134654) |
| 4 | aurora-lr1e-2 | normuon-lr3.6e-4 | Aurora polar (Tilde): row-uniform Stiefel; matrix LR rescaled to 1e-2 | Aurora | 1e-2 | 1.805 | 1.697 | [`03-aurora-lr1e-2.sbatch`](runs/03-aurora-lr1e-2.sbatch) | [awmmzpur](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/awmmzpur) | [`9e2134654`](https://github.com/ischlag/megatron-lm-research-baseline/tree/9e2134654) |

## Notes on entries

- **Aurora ranking flips vs the 1B ablation.** At 1B tokens, `aurora-qkn`
  led at 2.185 final / 2.025 min; `normuon-lr3.6e-4` was at 2.224 / 2.061
  (Aurora-qkn wins by 0.039 final). At 15B the ordering reverses: NorMuon
  wins by 0.027 final. The Aurora LR=1e-2 that was robust at 1B (sweep
  5e-3 / 8e-3 / 1e-2 / 2e-2 / 3e-2 within 0.05 nats) does not transfer
  cleanly to the 15B schedule. An LR sweep at 15B is queued as follow-up.
- **`aurora-qkn` vs `aurora-lr1e-2`**: QK-norm gain is ~0.003 final at
  15B (1.802 vs 1.805) — much smaller than at 1B (~0.015). The
  architectural lift compresses with longer training.
- **Throughput**: per-iter time at 2 nodes is ~620ms (AdamW, ~256
  TFLOP/s/GPU), ~640ms (NorMuon, ~247), ~700ms (Aurora variants, ~225).
  Inter-node NCCL overhead drops aggregate TFLOP/s ~10-15% vs 1-node
  ablation rates. AdamW completes in ~5h, Aurora variants in ~5h 40m.
- `--overlap-param-gather` is on for AdamW and off for NorMuon / Aurora
  (it corrupts Newton-Schulz; see the main README).
