#!/usr/bin/env bash
# Run the Stage 2 eval suite at every step=N/ checkpoint of a Stage 3b run, so
# we can plot how Stage 2 capabilities (LLM4Mat-Bench MAD:MAE, MatterChat
# accuracy, language retention, leak rate, etc.) drift as the LLM is trained
# to pool chemical info into the [atoms_i] tokens.
#
# Stage 3b checkpoints carry both `lora_adapter/` and `projector_and_state.pt`,
# so the existing Stage 2 eval pipeline (run_all_evals.sh + load_alm) treats
# them transparently — no code changes in the eval scripts needed.
#
# Stage 3a step= dirs (which only contain `atoms_mapper.pt`) have no LLM-side
# state changes during training; this script skips them with a notice — eval
# the Stage 2 init checkpoint that started the 3a run instead, those numbers
# are constant.
#
# Usage:
#   bash helper_scripts/eval_stage2_across_stage3b.sh STAGE3B_RUN_DIR [GPU_LIST]
#
# Example:
#   bash helper_scripts/eval_stage2_across_stage3b.sh \
#       /home/sathyae/orcd/pool/stage3a/ckpts/run10_stage3b_K4_2node \
#       0,1,2,3
#
# Env overrides:
#   CKPT_GLOB=...     pattern for step dirs (default: step=*)
#   STEP_STRIDE=1     evaluate every Nth checkpoint (default: every one)
#   BATCH_SIZE=32     forwarded to run_all_evals.sh
#   SKIP_COMPLETED=1  set to 0 to force re-run benchmarks at completed steps
#
# Eval results land at:
#   $POOL/eval_results/{bench}/<run_name>__step=N/
# (RUN_ID is auto-derived by run_all_evals.sh from the ckpt's parent + step
#  basename, so two Stage 3b runs with overlapping step numbers don't collide.)
#
# Optional: pipe through evals/aggregate_results.py afterward to produce a
# trajectory plot — once enough step=N have completed, you'll have one row per
# (bench, step) in the long-format CSV.

set -euo pipefail

CKPT_PARENT="${1:?usage: $0 STAGE3B_RUN_DIR [GPU_LIST]}"
GPU_LIST="${2:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
GLOB_PAT="${CKPT_GLOB:-step=*}"
STEP_STRIDE="${STEP_STRIDE:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$CKPT_PARENT" ]]; then
    echo "error: $CKPT_PARENT does not exist" >&2
    exit 1
fi

# Collect step dirs sorted by numeric step (so step=11000 sorts after step=2000)
mapfile -t STEP_DIRS < <(
    ls -d "$CKPT_PARENT"/$GLOB_PAT 2>/dev/null \
        | awk -F'step=' '{print $2"\t"$0}' \
        | sort -n -k1 \
        | cut -f2-
)

if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
    echo "error: no step dirs matching $CKPT_PARENT/$GLOB_PAT" >&2
    exit 1
fi

echo "[main] sweeping ${#STEP_DIRS[@]} checkpoints in $CKPT_PARENT (stride=$STEP_STRIDE, GPUs=$GPU_LIST)"

i=0
for STEP_DIR in "${STEP_DIRS[@]}"; do
    if (( i % STEP_STRIDE != 0 )); then
        i=$((i+1))
        continue
    fi
    i=$((i+1))

    if [[ ! -d "$STEP_DIR/lora_adapter" ]]; then
        echo "[skip] $(basename "$STEP_DIR") — no lora_adapter (Stage 3a or stale)"
        continue
    fi
    if [[ ! -f "$STEP_DIR/projector_and_state.pt" ]]; then
        echo "[skip] $(basename "$STEP_DIR") — no projector_and_state.pt"
        continue
    fi

    STEP_NAME="$(basename "$STEP_DIR")"
    RUN_NAME="$(basename "$CKPT_PARENT")"
    echo ""
    echo "================================================================"
    echo "[main] $RUN_NAME / $STEP_NAME → Stage 2 eval suite"
    echo "================================================================"

    CKPT="$STEP_DIR" \
    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        bash "$SCRIPT_DIR/run_all_evals.sh"
done

echo ""
echo "[main] all Stage 3b checkpoints evaluated. Results under:"
echo "  /home/sathyae/orcd/pool/eval_results/{bench}/${RUN_NAME}__step=N/"
echo ""
echo "Aggregate trajectories with:"
echo "  python evals/aggregate_results.py --root /home/sathyae/orcd/pool/eval_results"
