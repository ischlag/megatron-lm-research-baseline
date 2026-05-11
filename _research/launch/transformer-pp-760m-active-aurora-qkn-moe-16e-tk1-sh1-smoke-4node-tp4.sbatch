#!/bin/bash
#
# 760M-active MoE for 15B-token Stage-3 scaling experiments.
# 16 routed experts top-1 + 1 shared expert, all width 2048 (the K=1
# Switch-routing + shared-backstop recipe that won at 350M-active).
#
# Active params match dense 760M (~760M); total ~4.31B.
#   Per-layer FFN active = (topk_routed + 1_shared) * 3 * hidden * 2048
#                        = 2 * 3 * 1536 * 2048 = 18.87M  (matches dense
#                        FFN: 3 * 1536 * 4096 = 18.87M)
#   Per-layer FFN total  = (16 + 1) * 3 * 1536 * 2048 = 160.4M
# Routed sparsity 6.25% (K/E = 1/16). Optimizer is AdamW @ LR 5e-4 for
# this rung (not Aurora) — fits-in-one-node scaling experiment, not an
# optimizer ablation. AdamW lets us turn on --overlap-param-gather
# (Aurora forbids it; Newton-Schulz corruption).
# Architecture inherits the 760M dense recipe: 24 layers, hidden 1536,
# d_ffn 4096, 24/8 GQA, RMSNorm, SwiGLU, RoPE, qk-layernorm.
#
# Memory budget (per GPU, MBS=8, EP=1 DP=4):
#   bf16 weights + grads (replicated): ~17 GB
#   AdamW expert opt state (DP=4 sharded): ~12 GB
#   activations (incl. MoE permute/dispatch): ~45 GB
#   total ~74 GB / 96 GB. Drop to MBS=4 if OOM at startup.
#
# Run length: 15B tokens (D/N_active ~ 20, Chinchilla-like). Per-iter
# extrapolation from 350M tk1-sh1 (~120-180 TFLOP/s/GPU at 760M) puts
# the 28,610-iter run at ~24h wall — over the 12h "normal"-partition
# cap, so --save / --load every 1000 iters and resume via resubmit (2-3 times).
#
# Reference: Mixtral 8x7B (Jiang et al., 2024); DeepSeek-V3 shared expert.
#
#SBATCH --job-name=760m-active-aurora-qkn-moe-16e-tk1-sh1-smoke-4node-tp4
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

WORKDIR=${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline}
REPO_DIR=$WORKDIR
cd $WORKDIR

PACKAGE_DIR=$REPO_DIR/_research/packages
DATASET_CACHE_DIR=$REPO_DIR/cache/dataset
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance _research/results/runs/ckpt-$SLURM_JOB_NAME

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=760m-active-aurora-qkn-moe-16e-tk1-sh1-smoke-4node-tp4-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-aurora-moe-16e-tk1-sh1-smoke-4node-tp4
export APERTUS_TRACK=760m-aurora-qkn-moe-16e-tk1-sh1-smoke-4node-tp4
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=0
export APERTUS_LOG_TOP1_ACC=0
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=0
export APERTUS_LOG_NEURON_INTERVAL=50

MEGATRON_ARGS=(
    # Architecture: 760M-active Transformer++ MoE (24L, 1536H, 24h, 8kv GQA)
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
    --num-layers 24
    --hidden-size 1536
    --ffn-hidden-size 4096
    --num-attention-heads 24
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length 4096
    # MoE: 16 experts, top-1 routed + 1 shared (Switch + backstop), d_e=2048
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    # Training: 15B tokens, GBS=128 (524K tokens/step), MBS=4 (DP=4, 8 grad-accum
    # steps). 760M-active is bigger than the 350M MoE, so MBS=8 is the upper
    # CE step (logits buffer alone is 6.14 GB at MBS=8); dropping to MBS=4.
    --micro-batch-size 16
    --global-batch-size 128
    --train-samples 3662109
    --exit-duration-in-mins 10
    --log-interval 1
    --eval-interval 100000
    --eval-iters 0
    --optimizer adam
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
    # Regularization
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    # WSD schedule: 25600-sample warmup (~200 iters) + last 20% decay (~5722 iters)
    --lr 5e-4
    --min-lr 5e-5
    --lr-decay-style WSD
    --lr-warmup-samples 25600
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 732421
    # Init / precision
    --seed 42
    --init-method-std 0.02
    --bf16
    --fp8-format e4m3
    --fp8-recipe blockwise
    # Distributed: TP=1, PP=1, EP=1, DP=4. AdamW path so --overlap-param-gather
    # is safe to enable (Aurora forbids it for Newton-Schulz). Distributed
    # optimizer shards opt state across DP=4.
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    # Logging
    --log-throughput
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    # Checkpointing: save every 1000 iters (~1h at projected throughput);
    # auto-resume from latest ckpt on resubmit. Run can survive a 24h
    # 12h cap and resume with a fresh resubmit (the --load auto-picks latest).
    # Tokenizer / data
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/_research/data/gpt2-vocab.json
    --merge-file $WORKDIR/_research/data/gpt2-merges.txt
    --data-path $MEGATRON_DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 4
)

MEGATRON_ARGS_STR="${MEGATRON_ARGS[@]}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS_STR="$MEGATRON_ARGS_STR --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $WORKDIR/_research/results/runs"
else
    export WANDB_MODE=disabled
fi

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS/SLURM_NNODES)); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
