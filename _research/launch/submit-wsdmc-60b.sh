#!/bin/bash
# Submit a 60B-target WSDMC chain: N chained mains + 7 cooldowns at
# endpoints {3, 6, 9, 12, 15, 30, 60}B. Use this instead of submit-wsdmc.sh
# when the main sbatch is configured with train-samples=11719680 (= 91560
# iters * 128, 48B tokens). The cooldowns at 30B and 60B are not on the
# uniform i*save_interval grid, so submit-wsdmc.sh would not emit them.
#
# Usage:
#   ./submit-wsdmc-60b.sh <main_sbatch_path>
# Env:
#   WSDMC_N_MAIN_JOBS  (default 1) - chained 11h main links
#   WSDMC_SEC_PER_ITER (default 2.5) - per-iter estimate, used to budget cooldown --time
#   WSDMC_EXP_NAME     (default <main-stem>-<git-sha>)
#   WSDMC_CKPT_DIR     (default _research/results/ckpts/$WSDMC_EXP_NAME)
#
set -euo pipefail

MAIN_SBATCH="${1:?usage: $0 <main_sbatch>}"
[ -f "$MAIN_SBATCH" ] || { echo "ERROR: main sbatch not found: $MAIN_SBATCH" >&2; exit 1; }

N_MAIN_JOBS="${WSDMC_N_MAIN_JOBS:-1}"
SEC_PER_ITER="${WSDMC_SEC_PER_ITER:-2.5}"

MAIN_DIR=$(dirname "$MAIN_SBATCH")
MAIN_BASE=$(basename "$MAIN_SBATCH")
DEFAULT_SUFFIX=$(echo "$MAIN_BASE" | sed -E 's/^.*-wsdmc-//; s/\.sbatch$//')
SIZE=$(echo "$MAIN_BASE" | sed -nE 's/^transformer-pp-([0-9a-z.]+)-wsdmc-.*/\1/p')
COOLDOWN_SBATCH="$MAIN_DIR/wsdmc-cooldown-$SIZE-$DEFAULT_SUFFIX.sbatch"
[ -f "$COOLDOWN_SBATCH" ] || { echo "ERROR: cooldown sbatch not found: $COOLDOWN_SBATCH" >&2; exit 1; }

EXP_STEM=$(echo "$MAIN_BASE" | sed 's/^transformer-pp-//; s/\.sbatch$//')
EXP_NAME="${WSDMC_EXP_NAME:-${EXP_STEM}-$(git rev-parse --short HEAD)}"
CKPT_DIR="${WSDMC_CKPT_DIR:-$(pwd)/_research/results/ckpts/$EXP_NAME}"
mkdir -p "$CKPT_DIR"

echo "WSDMC-60B submission plan"
echo "  main sbatch:     $MAIN_SBATCH"
echo "  cooldown sbatch: $COOLDOWN_SBATCH"
echo "  exp name:        $EXP_NAME"
echo "  ckpt dir:        $CKPT_DIR"
echo "  N main jobs:     $N_MAIN_JOBS"
echo "  sec/iter:        $SEC_PER_ITER"
echo

# Submit N chained main links.
LAST_MAIN_JID=""
for ((i=1; i<=N_MAIN_JOBS; i++)); do
    if [ -z "$LAST_MAIN_JID" ]; then
        JID=$(sbatch --parsable --export=ALL,WSDMC_EXP_NAME=$EXP_NAME,WSDMC_CKPT_DIR=$CKPT_DIR "$MAIN_SBATCH")
    else
        # afterany: continue chain even if previous link crashed (NCCL hang,
        # node fault). Each link uses --load=--save so it auto-resumes from
        # latest_checkpointed_iteration.txt. Cooldowns still use afterok the
        # final main link, which exits clean only when consumed >= train_samples.
        JID=$(sbatch --parsable --dependency=afterany:$LAST_MAIN_JID --export=ALL,WSDMC_EXP_NAME=$EXP_NAME,WSDMC_CKPT_DIR=$CKPT_DIR "$MAIN_SBATCH")
    fi
    echo "Submitted MAIN $i/$N_MAIN_JOBS: jobid=$JID${LAST_MAIN_JID:+ (afterany:$LAST_MAIN_JID)}"
    LAST_MAIN_JID="$JID"
done

# 7 cooldowns at endpoints {3, 6, 9, 12, 15, 30, 60}B.
# Each tuple: ckpt_iter cooldown_iters endpoint_label
COOLDOWNS=(
    "4578 1145 3.0B"
    "9156 2289 6.0B"
    "13734 3434 9.0B"
    "18312 4578 12.0B"
    "22890 5723 15.0B"
    "45780 11445 30.0B"
    "91560 22890 60.0B"
)

idx=1
for entry in "${COOLDOWNS[@]}"; do
    read -r CI CDI EP <<<"$entry"
    SECS=$(python3 -c "import math; print(int(math.ceil($CDI * $SEC_PER_ITER * 1.3 + 600)))")
    [ "$SECS" -lt 1800 ] && SECS=1800
    [ "$SECS" -gt 42900 ] && SECS=42900   # 11h55m, just under partition 12h MaxTime
    HH=$((SECS / 3600))
    MM=$(((SECS % 3600) / 60))
    TIME=$(printf "%02d:%02d:00" "$HH" "$MM")

    CD_NAME="${EXP_NAME}-cd-${EP}"
    JID=$(sbatch --parsable --time="$TIME" --dependency=afterok:"$LAST_MAIN_JID" \
        --job-name="$CD_NAME" \
        --export=ALL,WSDMC_CKPT_DIR=$CKPT_DIR,WSDMC_CKPT_ITER=$CI,WSDMC_COOLDOWN_ITERS=$CDI,WSDMC_EXP_NAME=$EXP_NAME \
        "$COOLDOWN_SBATCH")
    echo "Submitted COOLDOWN $idx/7: jobid=$JID ckpt_iter=$CI cooldown_iters=$CDI endpoint=$EP time=$TIME"
    idx=$((idx + 1))
done

echo
echo "All jobs submitted. Track with:"
echo "  sacct -X -j $LAST_MAIN_JID --format=JobID,JobName,State,Elapsed,End"
