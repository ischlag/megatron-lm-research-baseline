#!/bin/bash
#
# 350M-active ablation: Scion + QK-norm + MoE. 32 routed experts top-3 + 1
# shared expert (all width 640), iso-active to dense scion-qkn (~355M active,
# ~1.73B total). Scion (arXiv 2502.07529) gamma=3e-4. Under --moe-grouped-gemm
# + TE the experts are per-expert 2-D weight{i} params -> all Spectral@50
# (fully orthogonalized, NOT routed to AdamW). See runs/20 header + the note.
#
# Parent: transformer-pp-350m-ablation-scion-qkn.sbatch (19-scion-qkn-lr3e-4)
# Change: replace dense FFN with 32 routed experts top-3 + 1 shared (width 640);
#         + aux/z losses (1e-2/1e-3); MBS 16 -> 8; time 1h -> 2.5h.
#
# Frozen leaderboard entry: runs/20-scion-qkn-moe-32e-tk3-sh1-lr3e-4.sbatch
# To try a variant, copy this file and edit; do not re-introduce env-var branching.
#SBATCH --job-name=350m-abl-scion-qkn-moe-32e-tk3-sh1
#SBATCH --time=02:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

# REPO_DIR follows WORKDIR so a cluster-side sidecar clone (one per local
# worktree) keeps caches, packages, TB, and wandb out of the canonical path.
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
EXP_NAME=350m-abl-scion-qkn-moe-32e-tk3-sh1-lr3e-4-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-scion-moe-32e-tk3-sh1
export APERTUS_TRACK=350m-ablation-leaderboard
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

MEGATRON_ARGS=(
    # Architecture: 350M-active Transformer++ MoE (24L, 1024H, 16h, 4kv GQA)
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 2560
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
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
    # MoE: 32 experts, top-3, expert FFN 640 + 1 shared 640 (fine-grained,
    # iso-active to dense). Active FFN width = (3 + 1) * 640 = 2560 = dense.
    --num-experts 32
    --moe-router-topk 3
    --moe-ffn-hidden-size 640
    --moe-shared-expert-intermediate-size 640
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    # Training: 1B tokens, GBS=128 (524K tokens/step), MBS=8 (DP=4, 4 grad-accum
    # steps). Dense baseline runs at MBS=16; MoE permuted activations + grouped
    # GEMM workspace double the per-microbatch activation footprint, so we halve.
    --micro-batch-size 8
    --global-batch-size 128
    --train-samples 244141
    --exit-duration-in-mins 140
    --log-interval 1
    --eval-interval 244141
    --eval-iters 0
    --optimizer scion
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
    # Regularization. NOTE: --weight-decay 0.1 is the scheduler's WD field; it
    # is IGNORED by Scion, which pins its own decoupled mu=1 (= the Frank-Wolfe
    # constraint) inside the optimizer, protected from the scheduler.
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    # Scion recipe (norm-LMO Frank-Wolfe). Every 2-D matrix -- attention, FFN,
    # embeds, output, router AND each per-expert weight{i} -- gets Spectral@50;
    # 1-D norm gains get BiasRMS@50. mu=1, momentum 0.9, quintic NS x5.
    --scion-momentum 0.9
    --scion-constraint-coeff 1.0
    --scion-coefficient-type quintic
    --scion-num-ns-steps 5
    --scion-hidden-radius 50
    --scion-head-radius 3000
    --scion-embed-radius 3000
    --scion-norm-radius 50
    # WSD schedule, 200-step warmup, last 20% decay
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style WSD
    --lr-warmup-samples 25600
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 48828
    # Init / precision
    --seed 42
    --init-method-std 0.02
    --bf16
    # Distributed: TP=1, PP=1, EP=1, DP=4. EP=1 still fits (32 experts
    # * 31.5M = 1GB/layer, 24L = ~24GB on a 96GB GH200). --overlap-param-gather
    # omitted (corrupts Newton-Schulz; same caveat as Muon).
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    # Logging
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
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
