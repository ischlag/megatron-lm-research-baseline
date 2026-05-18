#!/bin/bash
#
# 175M-active MoE wsdmc production run (smaller N companion to 350M/760M
# for D/N estimation).
#   Architecture: 16 routed experts top-1 + 1 shared, all width 960
#                 (Switch-routing with shared backstop, iso-active to dense).
#                 16 layers, hidden 768, GQA 4 kv heads (head_dim=64 same
#                 as 350M). Aspect ratio 16/768 ~ 24/1024 of the 350M.
#                 Per-layer FFN active = (1 routed + 1 shared) * 3 * 768 * 960 = 4.42M
#                                        (matches dense reference 3 * 768 * 1920)
#                 Per-layer FFN total  = (16 + 1) * 3 * 768 * 960 = 37.6M
#   Active params: ~173M.  Total params: ~700M.
#   Sparsity K_routed/E_routed = 1/16 = 6.25% (matches 350M/760M).
#
#   Optimizer: AdamW at lr=5e-4 (same lr as 350M/760M AdamW lr=5e-4
#   chains; same baseline anchor for D/N curve).
#
#   WSDMC schedule (round endpoints {3, 6, 9, 12, 15}B):
#     save-interval = 4578 iters  =>  saves at {4578, 9156, 13734, 18312, 22890}
#     train-iters = 22890 (= 5 * save-interval; main stops at 12B tokens)
#   Each cooldown extends its save into a real WSD endpoint:
#     ckpt @  4578 (2.4B)  + cd 1144 (0.6B)  ->  3.0B
#     ckpt @  9156 (4.8B)  + cd 2289 (1.2B)  ->  6.0B
#     ckpt @ 13734 (7.2B)  + cd 3434 (1.8B)  ->  9.0B
#     ckpt @ 18312 (9.6B)  + cd 4578 (2.4B)  -> 12.0B
#     ckpt @ 22890 (12.0B) + cd 5722 (3.0B)  -> 15.0B
#
#   1-node DP=4 with MBS=16 (memory headroom over 350M). Expected
#   ~1.2-1.5 s/iter at FP8; main fits a single 11h SLURM link, but
#   submitted with WSDMC_N_MAIN_JOBS=2 for resume safety.
#
#SBATCH --job-name=175m-wsdmc-adamw-lr5e-4-fp8-moe-16e-tk1-sh1
#SBATCH --time=11:00:00
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
EXP_NAME=${WSDMC_EXP_NAME:-175m-wsdmc-adamw-lr5e-4-fp8-moe-16e-tk1-sh1-$(git rev-parse --short HEAD)}
CKPT_DIR=${WSDMC_CKPT_DIR:-$WORKDIR/_research/results/ckpts/$EXP_NAME}
mkdir -p $CKPT_DIR

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-adamw-fp8-moe-16e-tk1-sh1-wsdmc
export APERTUS_TRACK=175m-wsdmc
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
    --num-layers 16
    --hidden-size 768
    --ffn-hidden-size 1920
    --num-attention-heads 12
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
    --moe-ffn-hidden-size 960
    --moe-shared-expert-intermediate-size 960
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    --micro-batch-size 16
    --global-batch-size 128
    --train-samples 2929920
    --exit-duration-in-mins 600
    --log-interval 1
    --eval-interval 2929920
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
    --lr-warmup-samples 25600
    --lr-wsd-decay-style minus_sqrt
    --lr-decay-samples 5859840
    --lr-wsd-decay-samples 1024
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
    --save $CKPT_DIR
    --load $CKPT_DIR
    --save-interval 4578
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
