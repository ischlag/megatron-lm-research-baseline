#!/usr/bin/env bash
# Submit a matrix-LR sweep for transformer-pp-350m-master.sbatch.
# MLR grid: 2^(k/2) * BASE for k in [K_MIN, K_MAX]  (sqrt-2 steps).
#
# Usage:
#   bash submit-master-mlr-sweep.sh                  # default: k=0..10, base=1e-3
#   bash submit-master-mlr-sweep.sh 2 8              # k in [2, 8]
#   BASE=3e-3 bash submit-master-mlr-sweep.sh 0 6    # custom base
#   LR=3e-4 bash submit-master-mlr-sweep.sh          # custom embedding LR

set -euo pipefail

SBATCH=${SBATCH:-_research/launch/transformer-pp-350m-master.sbatch}
LR=${LR:-1e-3}
BASE=${BASE:-1e-3}
K_MIN=${1:-2}
K_MAX=${2:-8}

[ -f "$SBATCH" ] || { echo "ERROR: sbatch not found: $SBATCH" >&2; exit 1; }

# Generate grid: 2^(k/2) * BASE, formatted to 4 significant figures
mapfile -t MLR_VALUES < <(awk -v base="$BASE" -v k0="$K_MIN" -v k1="$K_MAX" \
    'BEGIN { for (k=k0; k<=k1; k++) printf "%.4g\n", 2^(k/2) * base }')

GIT_SHA=$(git rev-parse --short HEAD)

echo "Sweep: LR=$LR  BASE=$BASE  k=[${K_MIN},${K_MAX}]  sbatch=$SBATCH"
printf "  MLR grid: %s\n" "${MLR_VALUES[*]}"
echo

for mlr in "${MLR_VALUES[@]}"; do
    JOB_NAME="master-350m-lr${LR}-mlr${mlr}-${GIT_SHA}"
    JID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --export=ALL,LR="$LR",MLR="$mlr" \
        "$SBATCH")
    printf "  MLR=%-10s  jobid=%s\n" "$mlr" "$JID"
done
