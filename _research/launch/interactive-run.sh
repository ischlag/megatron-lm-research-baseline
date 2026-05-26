#!/bin/bash
#
# Run an sbatch's training command INLINE in your current interactive shell
# (skips the wrapping srun). Pre-req: you're already inside an interactive
# allocation on a GH200 node, inside the alps3 container.
#
# Typical flow on clariden:
#
#   srun --account=<acc> --time=01:00:00 --nodes=1 --gpus-per-node=4 \
#        --cpus-per-task=72 --mem=460000 --mpi=pmix \
#        --environment=_research/launch/alps3.toml --pty bash
#   # ... now inside the alps3 container ...
#   export MEGATRON_DATA_PATH=/path/to/climbmix_prefix
#   bash _research/launch/interactive-run.sh
#
# Usage:
#   bash _research/launch/interactive-run.sh
#     → runs the master ablation sbatch (1B tokens, ~30 min on 1 node).
#
#   bash _research/launch/interactive-run.sh _research/launch/<other.sbatch>
#     → runs any other sbatch interactively.
#
#   bash _research/launch/interactive-run.sh <sbatch> --train-iters 20 --exit-duration-in-mins 5
#     → extra flags are appended AFTER the sbatch args, so they override
#       (last value wins for argparse).
#
# Env overrides:
#   NPROC_PER_NODE=2 bash ...   → fewer ranks (default 4)
#   SKIP_DEPS=1     bash ...    → skip install_python_deps.sh
#
# Implementation: stubs `srun` + `scontrol`, sources the sbatch (which
# defines MEGATRON_ARGS as a bash array but never calls the real srun
# because we've shadowed it), then launches torchrun with those args.

set -euo pipefail

# Repo root = parent of _research/. Computed from this script's own path so
# the wrapper works regardless of where the user invoked `srun --pty bash`
# from (otherwise SLURM_SUBMIT_DIR points there and the sbatch's cd lands in
# a non-repo dir, causing `git rev-parse` to fail).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export SLURM_SUBMIT_DIR=$REPO_ROOT
cd $REPO_ROOT

DEFAULT_SBATCH=_research/launch/transformer-pp-350m-ablation-master-fp8.sbatch
SBATCH=${1:-$DEFAULT_SBATCH}
[ $# -gt 0 ] && shift

if [ ! -f "$SBATCH" ]; then
    echo "Sbatch not found: $SBATCH" >&2
    exit 1
fi

# Stubs so sourcing the sbatch doesn't try to launch / call SLURM commands.
srun() { :; }
scontrol() { hostname; }
export -f srun scontrol

# Fallback defaults for SLURM vars the sbatch references directly. Inside an
# interactive allocation these are normally set; the defaults let the wrapper
# also work from a plain login shell (e.g. for arg-extraction dry runs).
: ${SLURM_CPUS_PER_TASK:=72}
: ${SLURM_NTASKS:=4}
: ${SLURM_NNODES:=1}
: ${SLURM_JOB_NODELIST:=$(hostname)}
: ${SLURM_JOB_ID:=interactive}
export SLURM_CPUS_PER_TASK SLURM_NTASKS SLURM_NNODES SLURM_JOB_NODELIST SLURM_JOB_ID

# Source the sbatch — defines MEGATRON_ARGS, WORKDIR, PACKAGE_DIR, exports
# APERTUS_* env vars, runs idempotent mkdirs, then echos "END TIME" after the
# stubbed srun call returns. We end up with everything set up.
echo ">>> sourcing $SBATCH (srun stubbed)"
source "$SBATCH"
echo

# Install deps (the sbatch's TRAINING_CMD does this; we replicate it).
if [ -z "${SKIP_DEPS:-}" ]; then
    echo ">>> installing python deps into $PACKAGE_DIR"
    bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR
    echo
fi

# Single-node torchrun rendezvous (overrides the sbatch's SLURM-based addr).
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-25678}

NPROC=${NPROC_PER_NODE:-4}
cd $WORKDIR

echo ">>> WORKDIR=$WORKDIR"
echo ">>> torchrun --nproc-per-node=$NPROC --nnodes=1 pretrain_gpt.py [MEGATRON_ARGS] $*"
echo
exec torchrun --nproc-per-node=$NPROC --nnodes=1 \
    pretrain_gpt.py "${MEGATRON_ARGS[@]}" "$@"
