#!/bin/bash
#
# Debug helper. Run this INSIDE an interactive shell that you already got
# via `srun --account=... --environment=.../alps3.toml --pty bash`. It sets
# the same env vars the sbatches export, installs the python deps into
# $PACKAGE_DIR, then either execs back into bash (default) or runs the
# command you passed.
#
# Typical flow on clariden:
#
#   srun --account=infra01 --time=01:00:00 --nodes=1 --gpus-per-node=4 \
#        --cpus-per-task=72 --mem=460000 --mpi=pmix \
#        --environment=_research/launch/alps3.toml --pty bash
#   # ... now inside the alps3 container ...
#   bash _research/launch/interactive.sh
#   # ... env is set, deps installed, prompt back ...
#
# Or to run a single command directly:
#
#   bash _research/launch/interactive.sh python3 -c "from megatron.core.optimizer.master import MasterOptimizer; print('ok')"
#
# Once you have the env set, useful debug commands:
#
#   # CLI smoke test
#   python3 pretrain_gpt.py --help | grep -E \
#       'hypersphere|matrix-lr|orthogonal-updates|ademamix|master-min-lr'
#
#   # Import smoke test
#   python3 -c "from megatron.core.optimizer.master import \
#       MasterOptimizer, GainsMasterOptimizer; print('ok')"
#
#   # Short end-to-end on the master ablation recipe (20 iters, 4 GPUs).
#   #   MEGATRON_DATA_PATH must be set first.
#   export MEGATRON_DATA_PATH=...
#   torchrun --nproc-per-node=4 --nnodes=1 pretrain_gpt.py \
#       $(grep -E '^\s+--' _research/launch/transformer-pp-350m-ablation-master-fp8.sbatch \
#           | sed 's/#.*//' | tr -d '\\') \
#       --train-iters 20 --exit-duration-in-mins 10 --log-interval 1
#
#   # pdb-friendly single-rank run (drop a breakpoint() in master.py first):
#   torchrun --nproc-per-node=1 --nnodes=1 pretrain_gpt.py ... --train-iters 5

set -euo pipefail

REPO_DIR=${REPO_DIR:-/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline}
WORKDIR=${WORKDIR:-$REPO_DIR}
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
export TMPDIR=$TMP_CACHE_DIR

# torchrun rendezvous (single-node interactive debug). MASTER_ADDR is just
# the local hostname — torchrun reuses it for all local ranks.
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-25678}

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR \
         $TMP_CACHE_DIR $PACKAGE_DIR

cd $WORKDIR

echo ">>> installing python deps into $PACKAGE_DIR (idempotent — fast on repeat)"
bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR

cat <<EOF

>>> env ready. WORKDIR=$WORKDIR
>>> MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
>>> PYTHONPATH set, caches under $REPO_DIR/cache, deps in $PACKAGE_DIR.

EOF

if [ $# -eq 0 ]; then
    exec bash
else
    exec "$@"
fi
