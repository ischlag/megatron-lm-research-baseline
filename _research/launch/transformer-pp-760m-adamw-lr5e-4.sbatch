#!/bin/bash
#
# Dense 760M Transformer++ AdamW baseline: 15B tokens, plain WSD with
# 20% cooldown (minus_sqrt). Mirrors the old 350M run
# 350m-adamw-lr5e-4-0e26a01dd-2087424 (2 nodes, DP=8, MBS=16, 1 microbatch/step,
# 15B tokens, 3B cooldown, 1.57B warmup) but at 760M on 4 nodes
# (DP=16, MBS=8 -> still 1 microbatch/step, no grad accumulation).
#
# Architecture: 24L / 1536H / 4096F / 24h GQA-8kv, SwiGLU, RMSNorm,
# RoPE-500k, untied embed. Same shape as transformer-pp-760m-adamw.sbatch.
#
# Schedule: warmup 384000 samples (1.57B, ~10%), peak lr=5e-4, WSD
# stable + minus_sqrt cooldown over last 732420 samples (3B = 20% of 15B).
# train-samples=3662109 -> 15.0B tokens at seq=4096; train-iters=28610.
#
# Save every 5000 iters (~47 min at ~570ms/iter) for resume safety.
# Prior run (2226101) crashed at iter 24287/28610 from a NCCL watchdog
# timeout on the DP+CP group with no recent save (save-interval was
# 100k); ~3.1B tokens of compute lost. No WSDMC, no mid-run cooldowns.
#
# Expected wall: ~4-5 h on 4 nodes (16 GH200) at ~60 k tok/s/GPU dense.
#
#SBATCH --job-name=transformer-pp-760m-adamw-lr5e-4
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline-wsdmc/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline-wsdmc/_research/results/runs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH}"
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
EXP_NAME=transformer-pp-760m-adamw-lr5e-4-$(git rev-parse --short HEAD)
CKPT_DIR=$WORKDIR/_research/results/ckpts/$EXP_NAME
mkdir -p $CKPT_DIR

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-adamw-dense-wsd
export APERTUS_TRACK=transformer-pp-760m
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
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
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length 4096
    --micro-batch-size 8
    --global-batch-size 128
    --train-samples 3662109
    --exit-duration-in-mins 700
    --log-interval 1
    --eval-interval 3662109
    --eval-iters 0
    --optimizer adam
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
    --lr-warmup-samples 384000
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 732420
    --seed 42
    --init-method-std 0.02
    --bf16
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --save $CKPT_DIR
    --load $CKPT_DIR
    --save-interval 5000
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
echo "CKPT_DIR: $CKPT_DIR"
echo "EXP_NAME: $EXP_NAME"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
