#!/bin/bash
#
# Cooldown sbatch paired with
# transformer-pp-350m-wsdmc-aurora-lr5e-4-fp8-moe-16e-tk1-sh1.sbatch.
# Architecture and optimizer block must match the main run exactly.
# Schedule, save, load, exit-duration, and warmup are overridden via
# --override-opt_param-scheduler.
#
# Parameters via env (set by submit-wsdmc.sh):
#   WSDMC_CKPT_DIR / WSDMC_CKPT_ITER / WSDMC_COOLDOWN_ITERS / WSDMC_EXP_NAME
#
#SBATCH --job-name=wsdmc-cooldown-aurora-lr5e-4-fp8-moe-16e-tk1-sh1
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline-wsdmc/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline-wsdmc/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH}"
: "${WSDMC_CKPT_DIR:?missing WSDMC_CKPT_DIR}"
: "${WSDMC_CKPT_ITER:?missing WSDMC_CKPT_ITER}"
: "${WSDMC_COOLDOWN_ITERS:?missing WSDMC_COOLDOWN_ITERS}"
: "${WSDMC_EXP_NAME:?missing WSDMC_EXP_NAME}"

echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

WORKDIR=${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline-wsdmc}
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

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance _research/results/ckpts

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}

ENDPOINT_TOKENS=$(( (WSDMC_CKPT_ITER + WSDMC_COOLDOWN_ITERS) * 524288 ))
ENDPOINT_GB=$(awk "BEGIN {printf \"%.1f\", $ENDPOINT_TOKENS / 1e9}")
COOLDOWN_NAME="$WSDMC_EXP_NAME-cd-${ENDPOINT_GB}B"
COOLDOWN_CKPT_DIR=$WORKDIR/_research/results/ckpts/$COOLDOWN_NAME
mkdir -p $COOLDOWN_CKPT_DIR

TRAIN_ITERS_TOTAL=$(( WSDMC_CKPT_ITER + WSDMC_COOLDOWN_ITERS ))
TRAIN_SAMPLES=$(( TRAIN_ITERS_TOTAL * 128 ))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WSD_DECAY_SAMPLES=$(( WSDMC_COOLDOWN_ITERS * 128 ))

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-adamw-fp8-moe-16e-tk1-sh1-wsdmc
export APERTUS_TRACK=350m-wsdmc-cooldowns
export APERTUS_RUN_NAME=$COOLDOWN_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

MEGATRON_ARGS=(
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
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 1280
    --moe-shared-expert-intermediate-size 1280
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    --micro-batch-size 8
    --global-batch-size 128
    --train-samples $TRAIN_SAMPLES
    --log-interval 1
    --eval-interval $TRAIN_SAMPLES
    --eval-iters 0
    --optimizer aurora
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr 1.5e-3
    --muon-scalar-weight-decay 0.0
    --polar-precision bf16
    --aurora-pp-iterations 2
    --aurora-pp-beta 0.5
    --aurora-num-ns-steps 12
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 5e-4
    --min-lr 5e-5
    --lr-decay-style WSD
    --lr-warmup-samples 0
    --lr-wsd-decay-style minus_sqrt
    --lr-decay-samples $LR_DECAY_SAMPLES
    --lr-wsd-decay-samples $LR_WSD_DECAY_SAMPLES
    --override-opt_param-scheduler
    --seed 42
    --init-method-std 0.02
    --bf16
    --fp8-format e4m3
    --fp8-recipe blockwise
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --load $WSDMC_CKPT_DIR
    --ckpt-step $WSDMC_CKPT_ITER
    --save $COOLDOWN_CKPT_DIR
    --save-interval $WSDMC_COOLDOWN_ITERS
    --ckpt-format torch_dist
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
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
    export WANDB_NAME=$COOLDOWN_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS_STR="$MEGATRON_ARGS_STR --wandb-project $PROJECT --wandb-exp-name $COOLDOWN_NAME-$SLURM_JOB_ID --wandb-save-dir $WORKDIR/_research/results/runs"
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
echo "COOLDOWN_NAME: $COOLDOWN_NAME"
echo "  load:                 $WSDMC_CKPT_DIR (iter $WSDMC_CKPT_ITER)"
echo "  cooldown_iters:       $WSDMC_COOLDOWN_ITERS"
echo "  endpoint_tokens:      $ENDPOINT_TOKENS (~${ENDPOINT_GB}B)"
echo "  train_samples:        $TRAIN_SAMPLES"
echo "  lr_wsd_decay_samples: $LR_WSD_DECAY_SAMPLES"
echo "  save:                 $COOLDOWN_CKPT_DIR"

export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
