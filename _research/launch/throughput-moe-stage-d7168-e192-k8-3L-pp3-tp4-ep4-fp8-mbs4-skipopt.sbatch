#!/bin/bash
#
# Throughput probe for a single PP stage of a ~400-600B-class MoE,
# DeepSeek-V3-shaped (d=7168, n_h=128, d_h=128) with GQA and AdamW.
#
# Geometry:
#   d_model           = 7168
#   n_heads / n_kv    = 128 / 8 (GQA 16:1)
#   d_head            = 128
#   ffn (shared)      = 2048 (SwiGLU)
#   moe_ffn_hidden    = 2048 (SwiGLU)
#   num_experts       = 128 routed + 1 shared
#   top_k             = 8
#   n_layers          = 2  (fits 96GB/GPU with EP=4, no offload)
#
# Per-layer params (model view):
#   attention  ~  249M   (Q 7168x16384 + KV 7168x1024x2 + O 16384x7168)
#   experts    ~ 5.63B   (128 routed x 3 x 7168 x 2048)
#   shared     ~  44M
#   = 5.92B / layer; 2 layers = 11.8B
#
# Per-GPU memory at EP=4 / TP=1 / PP=1 / DP=4 (effective) / EDP=1, AdamW state in fp32:
#   experts per layer       ~ 23.2 GB   (1.45B params x 16B; EDP=1, no sharding)
#   attention per layer     ~  1.4 GB   (replicated 4x but distributed-optimizer
#                                       shards master+m+v over DP=4)
#   embeddings + output head~  3.6 GB   (distributed-optimizer sharded; untied)
#   = ~50 GB params/optstate ; activations + FA workspace + MoE dispatch fit
#   comfortably in the remaining ~46 GB.
#
# Parallelism note: with 4 GPUs, EP=4 forces TP=1 (Megatron requires
# DP * CP >= EP and DP * CP = world / TP / PP). TP=4 + EP=1 won't fit
# (256 experts replicated * AdamW state ~180 GB / GPU). EP=2 / TP=2 fits
# the same expert mass but adds a TP all-reduce inside every expert and
# halves alltoall partners -- almost certainly slower.
#
# Routing: --moe-router-force-load-balancing replaces router logits with
# random uniform values (Straight-Through Estimator); --moe-router-load-
# balancing-type none disables aux loss on those random logits. No
# capacity factor, no drops -- forced-uniform gives statistical balance
# by construction at sufficiently long sequences.
#
# Production-equivalent scale: 70-100 layers of this geometry = 410-590B
# total params, ~37-50B active per token (DeepSeek-V3 / Kimi-K2 class).
# Measured tput here is the per-stage forward+backward cost; P2P comm to
# the next PP stage is not modeled.
#
# Usage:
#   sbatch throughput-moe-stage-d7168-e128-tk8.sbatch        # MBS=1
#   sbatch throughput-moe-stage-d7168-e128-tk8.sbatch 2      # MBS=2
#
# Required env:
#   MEGATRON_DATA_PATH  : Megatron-binary prefix (content irrelevant for tput)
#   SBATCH_RESERVATION  : current reservation, e.g. SD-69241-apertus-1-5-0
#   SBATCH_ACCOUNT      : SLURM account
#
#SBATCH --job-name=tput-moe-stage-d7168-e192-k8-3L-pp3-tp4-ep4-fp8-mbs4-skipopt
#SBATCH --time=00:45:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

MBS=4                  # frozen variant: e192-k8-3L-pp3-tp4-ep4-fp8-mbs4-skipopt (E=192, K=8, L=3, 200 iters) (PP=3 with embed/LM-head on edge stages)
GBS=64                       # 16 microbatches (PP=3, MBS=4, DP=1), bubble ~12.5%

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

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=tput-moe-stage-d7168-e192-k8-3L-pp3-tp4-ep4-fp8-mbs4-skipopt-mbs${MBS}-$(git rev-parse --short HEAD)

# Logging-patch: turn off all per-step instrumentation that would
# contaminate throughput numbers (row_cv, act_stats, top1_acc, neuron_stats).
export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=throughput-moe-stage
export APERTUS_TRACK=throughput
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=0
export APERTUS_LOG_TOP1_ACC=0
export APERTUS_LOG_ROW_CV=0
export APERTUS_LOG_NEURON_STATS=0

MEGATRON_ARGS=(
    # --- Backend / precision ---
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
    --bf16
    --fp8-format e4m3
    --fp8-recipe blockwise
    --attention-backend auto
    # --- Architecture: DeepSeek-V3-shaped, GQA, 2 layers ---
    --num-layers 3
    --hidden-size 7168
    --ffn-hidden-size 2048
    --num-attention-heads 128
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --seq-length 4096
    # --- MoE: 128 routed + 1 shared, top-8, expert FFN 2048 ---
    --num-experts 192
    --moe-router-topk 8
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-router-force-load-balancing
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    # --- Distributed: TP=1, PP=1, EP=4 (DP=4 effective, EDP=1) ---
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 3
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --sequence-parallel
    "--pipeline-model-parallel-layout=E\\|ttt\\|L"
    --context-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    # --- Training: 200 iters, measure tail (e.g., iter 150-200) to debug variance ---
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters 200
    --log-interval 1
    --eval-interval 200
    --eval-iters 0
    --optimizer adam
    --skip-optimizer-step
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --no-check-for-nan-in-loss-and-grad
    # --- Regularization (irrelevant to throughput; standard values) ---
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    # --- LR: constant 1e-4 with short warmup; not training to convergence ---
    --lr 1e-4
    --min-lr 1e-4
    --lr-decay-style constant
    --lr-warmup-iters 10
    --seed 42
    --init-method-std 0.02
    # --- Logging ---
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    # --- Tokenizer / data (content irrelevant for tput) ---
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
echo "MBS=$MBS GBS=$GBS"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
