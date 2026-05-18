# WSDMC: warmup-stable-decay multi-cooldown

## What it is

A training protocol that produces multiple decayed-loss endpoints from a
single main run. Standard WSD (warmup-stable-decay) trains warmup +
stable phase at peak LR + a single decay at the end. **WSDMC saves
checkpoints during the stable phase**, then for each checkpoint launches
a separate cooldown job that loads it and runs the decay phase. Each
cooldown is a real WSD endpoint at total tokens D = ckpt_tokens / (1-p),
where p is the cooldown's fraction of D.

This is the intermittent-cooldown methodology of Hu et al. 2024 (MiniCPM,
https://arxiv.org/abs/2404.06395), adapted to multi-job SLURM with
`--dependency=afterok`. It is the default for ablation runs at SCALING.md
Stage 2+ (350M / 760M / 1.3B).

### Why bother

- **One main run, many endpoints.** Instead of running 5 separate WSD
  runs at 5 different token budgets to plot loss-vs-data, run one main
  through the longest stable phase and decay from intermediate
  checkpoints. The cluster cost is roughly the same as 1 long run plus
  ~20% for the cooldowns.
- **Stable-phase loss ranks correctly (MiniCPM result).** Inside a run,
  smoothed stable-phase loss tracks decayed loss for ranking. WSDMC adds
  the decayed measurements at chosen waypoints, so you get both an
  early-stop ranking signal *and* publishable absolute numbers.
- **Composable with continued training.** Main run can be resumed across
  multiple SLURM jobs (job chain) when wall time exceeds the partition's
  MaxTime; cooldowns are siblings, each branching from one save point.

### Mechanics

- The main sbatch sets `--lr-decay-style WSD` but with
  `--lr-decay-samples` pushed past `train_iters`, so the WSD scheduler
  stays in stable phase (coeff = 1) for all of training. No decay
  happens during main.
- `--save-interval` triggers checkpoint saves at known iters.
  `--save = --load = $CKPT_DIR` enables resume across job-chain links;
  Megatron auto-saves at `--exit-duration-in-mins`.
- Each cooldown sbatch loads its specific iter via `--ckpt-step K` and
  imposes a fresh WSD schedule via `--override-opt_param-scheduler`
  (`--lr-warmup-samples 0`, `--lr-decay-samples = (K + L) * GBS`,
  `--lr-wsd-decay-samples = L * GBS`). Cooldown loads optim state + RNG
  + data iterator (no `--finetune`); only the LR schedule diverges from
  main.

### Endpoint math

For cooldown fraction `p` (default 0.2), save iter `t`, and cooldown
length `l`: solve `l / (t + l) = p` for `l = t * p / (1 - p)`. With
`p=0.2` and 5 evenly-spaced saves at `t_i = i * (train_iters / 5)`, the
endpoints land at `1.25 * t_i`. Pick `train_iters` so this gives clean
round numbers; the production AdamW/16e-tk1-sh1 sbatch uses
`train_iters = 22890 = 5 * 4578`, producing endpoints
`{3, 6, 9, 12, 15}B` tokens.

Note that the main only trains to 12B (not 15B) under this design —
cooldown 5 from save 5 extends to 15B.

## Files

| file | purpose |
| --- | --- |
| `transformer-pp-350m-wsdmc-adamw-lr5e-4-moe-16e-tk1-sh1.sbatch` | production main: 12B AdamW lr=5e-4, MoE 16e-tk1-sh1, 5 saves |
| `wsdmc-cooldown-adamw-lr5e-4-moe-16e-tk1-sh1.sbatch` | paired cooldown (one sbatch invoked per save, parameterized by env) |
| `submit-wsdmc.sh` | submits 1 main (or N-job main chain) + 5 cooldowns with SLURM dependencies |
| `wsdmc-smoke-adamw-main.sbatch` | 30-iter tiny version of production for save/load smoke |
| `wsdmc-smoke-verify.py` | structural + log checks on smoke output |

## How to run a wsdmc run

### Prereqs

Cluster-side sidecar clone on the wsdmc branch (per AGENTS.md):
```bash
cd /iopsstor/scratch/cscs/$USER
git clone --branch feat/wsdmc https://github.com/ischlag/megatron-lm-research-baseline.git \
    megatron-lm-research-baseline-wsdmc
cd megatron-lm-research-baseline-wsdmc
```

Required env (in your `~/.bashrc`):
```bash
export SBATCH_ACCOUNT=<...>
export SBATCH_RESERVATION=<...>   # optional
export WANDB_API_KEY=<...>
export WANDB_PROJECT=<...>        # default: megatron-lm-research-baseline
export MEGATRON_DATA_PATH=<...>
```

### 1. Smoke test (5-10 min cluster wall)

Always run before launching a full production chain — Megatron has had
silent MoE checkpoint bugs (cf. upstream `0d4e12e99 fix moe torch save
when TP > EP * ETP`) and optimizer-specific save failures (cf. our
`project_normuon_torch_dist_save_bug.md`).

```bash
sbatch _research/launch/wsdmc-smoke-adamw-main.sbatch
# wait for COMPLETED, then run the paired cooldown smoke via env override:
WORKDIR=$(pwd)
sbatch --time=00:20:00 --job-name=wsdmc-smoke-adamw-cd \
    --export=ALL,WSDMC_CKPT_DIR=$WORKDIR/_research/results/ckpts/wsdmc-smoke-adamw,WSDMC_CKPT_ITER=10,WSDMC_COOLDOWN_ITERS=3,WSDMC_EXP_NAME=wsdmc-smoke-adamw-cd \
    _research/launch/wsdmc-cooldown-adamw-lr5e-4-moe-16e-tk1-sh1.sbatch
# verify:
python3 _research/launch/wsdmc-smoke-verify.py
```

Expect `ALL CHECKS PASSED`.

### 2. Production chain

Once smoke passes:

```bash
WSDMC_N_MAIN_JOBS=2 ./_research/launch/submit-wsdmc.sh \
    _research/launch/transformer-pp-350m-wsdmc-adamw-lr5e-4-moe-16e-tk1-sh1.sbatch
```

This submits:
- 2 main job links (chained via `--dependency=afterok`; each runs to
  `--exit-duration-in-mins=600` then Megatron auto-saves and exits
  cleanly; the next link resumes from the auto-save). Use
  `WSDMC_N_MAIN_JOBS` to span the partition's 12h MaxTime.
- 5 cooldowns, each `--dependency=afterok` on the LAST main link, with
  per-cooldown SLURM time scaled to its iter count.

Track progress:
```bash
sacct -X -j <main_2_jid> --format=JobID,JobName,State,Elapsed,End
```

### 3. Outputs

Under `_research/results/ckpts/`:
```
<exp-name>/                       # main run ckpts (5 saves + auto-save)
  iter_0004578/  iter_0009156/  iter_0013734/  iter_0018312/  iter_0022890/  iter_<auto>/
<exp-name>-cd-3.0B/               # cooldown 1 final ckpt
<exp-name>-cd-6.0B/
<exp-name>-cd-9.0B/
<exp-name>-cd-12.0B/
<exp-name>-cd-15.0B/
```

W&B run names: `<exp-name>-<jobid>` for main, `<exp-name>-cd-<endpoint>B-<jobid>`
for each cooldown.

## Tunables

Defaults are encoded in `submit-wsdmc.sh`; override per submission via env:

| env var | default | meaning |
| --- | --- | --- |
| `WSDMC_N_MAIN_JOBS` | 1 | chained main-job submissions |
| `WSDMC_N_CKPTS` | 5 | number of cooldown endpoints |
| `WSDMC_COOLDOWN_FRAC` | 0.2 | `l / (t + l)`; cooldown's share of endpoint length |
| `WSDMC_GBS` | 128 | global batch size (samples) for sample/iter conversion |

The submit script enforces `train_iters == N_CKPTS * save_interval` so
the last canonical save coincides with end of main training.

## Adding a new ablation

Each ablation needs a paired (main, cooldown) sbatch with matching
architecture and optimizer flags (the checkpoint format encodes both).
Cloning the AdamW pair is the cleanest start:

```bash
cp transformer-pp-350m-wsdmc-adamw-lr5e-4-moe-16e-tk1-sh1.sbatch \
   transformer-pp-350m-wsdmc-<ablation>.sbatch
cp wsdmc-cooldown-adamw-lr5e-4-moe-16e-tk1-sh1.sbatch \
   wsdmc-cooldown-<ablation>.sbatch
# edit the optimizer / architecture block in both, keep the schedule /
# save / load section as-is.
```

`submit-wsdmc.sh` infers the paired cooldown by suffix:
`transformer-pp-...-wsdmc-<suffix>.sbatch` -> `wsdmc-cooldown-<suffix>.sbatch`.

## Known issues

- **NorMuon + `--ckpt-format torch_dist`** fails the dist-checkpoint
  shape assertion (per-row second-moment shape (1, cols) vs param
  (rows, cols)). Use `--ckpt-format torch` for NorMuon. AdamW and Aurora
  work fine with `torch_dist`.
- **Main saves are not free.** Each save is ~25-30 GB at 1.85B-param
  MoE; 5 saves + auto-save = ~180 GB on scratch. Plan disk before
  scaling N_CKPTS up.
