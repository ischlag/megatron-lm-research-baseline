#!/usr/bin/env bash
# Submit a WSDMC run: 1 main sbatch + N cooldown sbatches each conditioned on
# the main job via SLURM --dependency=afterok.
#
# Usage:
#   ./submit-wsdmc.sh <main-sbatch> [cooldown-sbatch]
#   ./submit-wsdmc.sh _research/launch/transformer-pp-350m-wsdmc-aurora-qkn-moe-32e-tk3-sh1.sbatch
#
# Tunables (env vars):
#   WSDMC_N_CKPTS         number of checkpoints (default 5)
#   WSDMC_COOLDOWN_FRAC   cooldown length as fraction of accumulated tokens
#                         at each checkpoint (default 0.2)
#   WSDMC_GBS             global batch size (default 128). Used to convert
#                         --train-samples / --save-interval if main sbatch
#                         specifies samples; for iter-based, set 1.
#
# Invariant enforced: --train-samples must equal WSDMC_N_CKPTS * --save-interval
# (i.e. final save coincides with end of training, no orphan partial cycle).

set -euo pipefail

MAIN_SBATCH="${1:?usage: submit-wsdmc.sh <main-sbatch> [cooldown-sbatch]}"
# Default cooldown: same dir, same suffix-after-'wsdmc-' as main.
# transformer-pp-350m-wsdmc-FOO.sbatch -> wsdmc-cooldown-FOO.sbatch
MAIN_DIR=$(dirname "$MAIN_SBATCH")
MAIN_BASE=$(basename "$MAIN_SBATCH")
DEFAULT_COOLDOWN_SUFFIX=$(echo "$MAIN_BASE" | sed -E 's/^.*-wsdmc-//; s/\.sbatch$//')
# Extract size (e.g. "350m", "760m", "1.3b") from "transformer-pp-<size>-wsdmc-..."
# and prefer a size-prefixed cooldown if present (avoids architecture-mismatch
# when two different-size mains share the same optimizer/MoE suffix).
MAIN_SIZE=$(echo "$MAIN_BASE" | sed -nE 's/^transformer-pp-([0-9a-z.]+)-wsdmc-.*/\1/p')
if [ -n "${2:-}" ]; then
    COOLDOWN_SBATCH="$2"
elif [ -n "$MAIN_SIZE" ] && [ -f "$MAIN_DIR/wsdmc-cooldown-${MAIN_SIZE}-${DEFAULT_COOLDOWN_SUFFIX}.sbatch" ]; then
    COOLDOWN_SBATCH="$MAIN_DIR/wsdmc-cooldown-${MAIN_SIZE}-${DEFAULT_COOLDOWN_SUFFIX}.sbatch"
else
    COOLDOWN_SBATCH="$MAIN_DIR/wsdmc-cooldown-${DEFAULT_COOLDOWN_SUFFIX}.sbatch"
fi

[ -f "$MAIN_SBATCH" ]     || { echo "ERROR: main sbatch not found: $MAIN_SBATCH" >&2; exit 1; }
[ -f "$COOLDOWN_SBATCH" ] || { echo "ERROR: cooldown sbatch not found: $COOLDOWN_SBATCH" >&2; exit 1; }

N_CKPTS="${WSDMC_N_CKPTS:-5}"
# Fraction of the WSD endpoint's total length that the cooldown spans:
#   l / (t + l) = p, so l = t * p / (1 - p)
# Each cooldown is then a real WSD run of total length (t + l) where the
# last p fraction is decay. With p=0.2 the endpoints land at 1.25 * t.
# Design the main run so the saved iters t_i = (i/N) * train_iters give
# round-number endpoints; e.g. p=0.2 and train_iters = 5 * save_interval
# = 22890 give endpoints {3,6,9,12,15}B at GBS=128 / seq=4096.
COOLDOWN_FRAC="${WSDMC_COOLDOWN_FRAC:-0.2}"
GBS="${WSDMC_GBS:-128}"
# Number of chained main-job submissions. Each main job runs until its
# --exit-duration-in-mins triggers a save+exit; the next link in the chain
# auto-resumes via --load=--save. Use >1 when full training exceeds the
# partition's MaxTime. Cooldowns depend on the LAST main-link's afterok.
N_MAIN_JOBS="${WSDMC_N_MAIN_JOBS:-1}"

# Parse main sbatch for --train-samples and --save-interval (as iters).
# These appear inside the MEGATRON_ARGS=( ... ) array in the sbatch body.
get_arg() {
    local flag="$1"
    grep -E "^[[:space:]]*$flag[[:space:]]+[0-9]+" "$MAIN_SBATCH" | head -n1 | awk '{print $2}'
}

TRAIN_SAMPLES=$(get_arg "--train-samples")
SAVE_INTERVAL=$(get_arg "--save-interval")

[ -n "$TRAIN_SAMPLES" ] || { echo "ERROR: --train-samples not found in $MAIN_SBATCH" >&2; exit 1; }
[ -n "$SAVE_INTERVAL" ] || { echo "ERROR: --save-interval not found in $MAIN_SBATCH" >&2; exit 1; }

TRAIN_ITERS=$(( TRAIN_SAMPLES / GBS ))

# Invariant: train-iters must equal N_CKPTS * save-interval. Relaxed when
# WSDMC_SKIP_COOLDOWNS=1 (custom non-uniform cooldowns submitted out-of-band).
EXPECTED_INTERVAL=$(( TRAIN_ITERS / N_CKPTS ))
if [ "${WSDMC_SKIP_COOLDOWNS:-0}" = "0" ]; then
    if [ "$SAVE_INTERVAL" -ne "$EXPECTED_INTERVAL" ] || [ $(( SAVE_INTERVAL * N_CKPTS )) -ne "$TRAIN_ITERS" ]; then
        echo "ERROR: invariant violated: train-iters ($TRAIN_ITERS) must equal N_CKPTS ($N_CKPTS) * save-interval ($SAVE_INTERVAL); got expected interval $EXPECTED_INTERVAL" >&2
        exit 1
    fi
fi

# Build experiment name. The main sbatch's EXP_NAME default is
# "<jobname>-<git-sha>"; we recompute it here so submit-time and run-time
# agree, and pass via env so cooldowns can target the same ckpt dir.
GIT_SHA=$(git rev-parse --short HEAD)
JOB_NAME=$(grep -E '^#SBATCH --job-name=' "$MAIN_SBATCH" | head -n1 | sed 's/^#SBATCH --job-name=//')
WSDMC_EXP_NAME="${WSDMC_EXP_NAME:-${JOB_NAME}-${GIT_SHA}}"
WORKDIR="$(pwd)"
WSDMC_CKPT_DIR="${WSDMC_CKPT_DIR:-$WORKDIR/_research/results/ckpts/$WSDMC_EXP_NAME}"

echo "WSDMC submission plan"
echo "  main sbatch:     $MAIN_SBATCH"
echo "  cooldown sbatch: $COOLDOWN_SBATCH"
echo "  exp name:        $WSDMC_EXP_NAME"
echo "  ckpt dir:        $WSDMC_CKPT_DIR"
echo "  train iters:     $TRAIN_ITERS"
echo "  save interval:   $SAVE_INTERVAL"
echo "  N checkpoints:   $N_CKPTS"
echo "  cooldown frac:   $COOLDOWN_FRAC"
echo "  N main jobs:     $N_MAIN_JOBS"
echo

# Submit chain of main jobs. Each link auto-resumes from the previous via
# --load=--save (set to the same dir in the main sbatch). The cooldowns
# depend on the LAST link's afterok; intermediate links' afterok keeps the
# chain alive only if the previous link exits cleanly (which Megatron does
# at exit-duration-in-mins).
# Set WSDMC_SKIP_MAIN=1 to submit cooldowns only (mains already done).
LAST_MAIN_JID=""
if [ "${WSDMC_SKIP_MAIN:-0}" = "0" ]; then
    for j in $(seq 1 "$N_MAIN_JOBS"); do
        DEP_ARG=()
        if [ -n "$LAST_MAIN_JID" ]; then
            DEP_ARG=(--dependency=afterok:"$LAST_MAIN_JID")
        fi
        MAIN_OUT=$(sbatch --parsable "${DEP_ARG[@]}" --export=ALL,WSDMC_EXP_NAME="$WSDMC_EXP_NAME",WSDMC_CKPT_DIR="$WSDMC_CKPT_DIR" "$MAIN_SBATCH")
        MAIN_JID=$(echo "$MAIN_OUT" | tr -d '\n' | awk '{print $NF}')
        echo "Submitted MAIN $j/$N_MAIN_JOBS: jobid=$MAIN_JID${LAST_MAIN_JID:+ (afterok:$LAST_MAIN_JID)}"
        LAST_MAIN_JID="$MAIN_JID"
    done
else
    echo "WSDMC_SKIP_MAIN=1 - skipping main submission; cooldowns will run immediately"
fi
MAIN_JID="$LAST_MAIN_JID"

# Submit one cooldown per checkpoint, unless WSDMC_SKIP_COOLDOWNS=1
# (e.g. when emitting non-uniform endpoints out-of-band).
if [ "${WSDMC_SKIP_COOLDOWNS:-0}" = "1" ]; then
    echo "WSDMC_SKIP_COOLDOWNS=1 - skipping cooldown submission${MAIN_JID:+; chain ends at MAIN_JID=$MAIN_JID}"
    exit 0
fi

for i in $(seq 1 "$N_CKPTS"); do
    CKPT_ITER=$(( SAVE_INTERVAL * i ))
    CKPT_TOKENS=$(( CKPT_ITER * GBS * 4096 ))                          # GBS * seq=4096
    # COOLDOWN_ITERS = round(CKPT_ITER * p / (1 - p)), at least 1.
    # See COOLDOWN_FRAC comment above for derivation.
    COOLDOWN_ITERS=$(awk -v ci="$CKPT_ITER" -v p="$COOLDOWN_FRAC" 'BEGIN{ v=ci*p/(1-p); printf "%d", (v<1?1:v+0.5) }')
    ENDPOINT_TOKENS=$(( (CKPT_ITER + COOLDOWN_ITERS) * GBS * 4096 ))
    ENDPOINT_GB=$(awk -v t="$ENDPOINT_TOKENS" 'BEGIN{ printf "%.1f", t/1e9 }')

    # SLURM time budget: ~1.0 sec/iter on the MoE-32E-tk3-sh1 config + 30%
    # margin + 5 min container/init overhead. Min 30 min.
    # Per-iter sec estimate × 1.3 margin + 600s container/init overhead, min 30 min.
    # Default 2.5 covers 350m bf16 (~2s), 350m fp8 2-node (~2.1s), 350mx2 PP=2 (~2.6s).
    # Override via WSDMC_SEC_PER_ITER for faster configs (e.g. 4-node 760m fp8 at ~1.3).
    SEC_PER_ITER="${WSDMC_SEC_PER_ITER:-2.5}"
    SLURM_SECS=$(awk -v c="$COOLDOWN_ITERS" -v s="$SEC_PER_ITER" 'BEGIN{ x=c*s*1.3+600; if(x<1800)x=1800; printf "%d", x }')
    SLURM_HHMM=$(printf '%02d:%02d:00' $((SLURM_SECS/3600)) $(( (SLURM_SECS%3600)/60 )))

    JNAME="${WSDMC_EXP_NAME}-cd-${ENDPOINT_GB}B"

    DEP_ARGS=()
    if [ -n "$MAIN_JID" ]; then
        DEP_ARGS=(--dependency=afterok:"$MAIN_JID")
    fi
    CD_OUT=$(sbatch --parsable \
        "${DEP_ARGS[@]}" \
        --time="$SLURM_HHMM" \
        --job-name="$JNAME" \
        --export=ALL,WSDMC_EXP_NAME="$WSDMC_EXP_NAME",WSDMC_CKPT_DIR="$WSDMC_CKPT_DIR",WSDMC_CKPT_ITER="$CKPT_ITER",WSDMC_COOLDOWN_ITERS="$COOLDOWN_ITERS" \
        "$COOLDOWN_SBATCH")
    CD_JID=$(echo "$CD_OUT" | tr -d '\n' | awk '{print $NF}')

    printf "Submitted COOLDOWN %d/%s: jobid=%s ckpt_iter=%d cooldown_iters=%d endpoint=%sB time=%s\n" \
        "$i" "$N_CKPTS" "$CD_JID" "$CKPT_ITER" "$COOLDOWN_ITERS" "$ENDPOINT_GB" "$SLURM_HHMM"
done

echo
echo "All jobs submitted. Track with:"
echo "  sacct -X -j $MAIN_JID --format=JobID,JobName,State,Elapsed,End"
