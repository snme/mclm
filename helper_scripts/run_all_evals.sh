#!/bin/bash

# Run all 8 benchmarks in PARALLEL across N GPUs (round-robin assignment, longest
# bench first so the heavyweights spread out before any GPU doubles up).
#
# GPU resolution (precedence, highest first):
#   1. CUDA_VISIBLE_DEVICES from the calling shell — pinned set of GPUs.
#      e.g.  CUDA_VISIBLE_DEVICES=7 ./run_all_evals.sh        → 1 GPU (id 7)
#            CUDA_VISIBLE_DEVICES=2,5,7 ./run_all_evals.sh    → 3 GPUs
#   2. NUM_GPUS=K env var — clamps the list (or auto-detected count) to the
#      first K. Useful for sharing a node.
#   3. nvidia-smi -L auto-detect — falls back to all physical GPUs.
#
# Other override-able env vars:
#   CKPT=/path/to/step=N      — checkpoint dir (REQUIRED to update before run)
#   BATCH_SIZE=32             — bigger batches → faster on H200; default 32
#   BLOCK_LEAK_FLAG=""        — see note below
#   SKIP_COMPLETED=1          — skip benchmarks with existing metrics.json
#
# Layout examples (after GPU resolution gives N usable GPUs):
#   N=8 → llm4mat alone on GPU[0]; every other bench gets its own GPU; one idle.
#   N=4 → llm4mat on GPU[0], others queued sequentially across GPUs[1..3].
#   N=1 → everything sequential on the single GPU.
#
# Logs land in ${POOL}/eval_results/{bench}/{step=N}/run.log so you can
# tail individual jobs.

set -e

CKPT="${CKPT:-/tmp/step=6500}"   # honors $CKPT env var; default = /tmp staging path
CKPT="/tmp/stage2_r128_arxivIT_stage2p5/step=18000"

MC_VAL_CSV=/home/sathyae/orcd/pool/eval_data/Dataset_MatterChat/dataset/Material_data_postprocess1_out_correct_val.csv
MC_GNOME_CSV=/home/sathyae/orcd/pool/eval_data/Dataset_MatterChat/dataset/Material_data_GnoME_processed_mace.csv

POOL=/home/sathyae/orcd/pool
# Namespace eval results by the checkpoint's parent dir + step basename so
# different experiment branches with `step=N` don't collide. Eval scripts pick
# this up via runs.py::run_dir reading ALM_EVAL_RUN_ID.
PARENT_TAG=$(basename "$(dirname "$CKPT")")
STEP_BASE=$(basename "$CKPT")
RUN_ID="${RUN_ID:-${PARENT_TAG}__${STEP_BASE}}"
export ALM_EVAL_RUN_ID="$RUN_ID"
STEP="$RUN_ID"   # used by should_skip / output_dir_for so the new layout is consistent
BATCH_SIZE=${BATCH_SIZE:-32}

# Resolve which GPUs to use. Honor CUDA_VISIBLE_DEVICES from the calling shell
# (e.g. CUDA_VISIBLE_DEVICES=7 to pin to a single GPU). Fall back to all GPUs on
# the node via nvidia-smi. NUM_GPUS clamps the resulting list to the first N.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    GPU_SOURCE="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    AVAIL=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ -z "$AVAIL" ] || [ "$AVAIL" -lt 1 ] && AVAIL=1
    GPU_LIST=()
    for ((i=0; i<AVAIL; i++)); do GPU_LIST+=("$i"); done
    GPU_SOURCE="nvidia-smi (${AVAIL} GPUs detected)"
fi
if [ -n "${NUM_GPUS:-}" ] && [ "$NUM_GPUS" -lt "${#GPU_LIST[@]}" ]; then
    GPU_LIST=("${GPU_LIST[@]:0:$NUM_GPUS}")
fi
NUM_GPUS=${#GPU_LIST[@]}
# Unset the inherited CUDA_VISIBLE_DEVICES so that each child process gets a
# fresh value set explicitly per-job below. (We pass the absolute GPU id from
# GPU_LIST; CUDA in the child sees that one GPU as cuda:0.)
unset CUDA_VISIBLE_DEVICES
echo "[main] GPUs=${GPU_LIST[*]} (NUM_GPUS=$NUM_GPUS, source: $GPU_SOURCE)"
echo "[main] BATCH_SIZE=$BATCH_SIZE  CKPT=$CKPT"
echo "[main] RUN_ID=$RUN_ID  (eval results land at $POOL/eval_results/{bench}/$RUN_ID/)"

# Decode-time leak guard. Off by default — empirically, `--block_leak_tokens`
# DEGRADES MAE because the imgur/URL fallback was acting as a calibration
# safety valve: when the model was uncertain, it would emit the imgur path
# (which the parser drops as a leak); blocking that path forced the model to
# commit wrong-magnitude numbers instead. Confirmed on r128_arxivIT/step=4500:
# mp/formation MAE 0.11 → 14.5, mp/eh MAE 0.06 → 177, mp/density 0.29 → 2.9.
# Set BLOCK_LEAK_FLAG="--block_leak_tokens" to ablate.
BLOCK_LEAK_FLAG=${BLOCK_LEAK_FLAG:-""}

SKIP_COMPLETED=${SKIP_COMPLETED:-1}

# Map a logical bench name to its actual output dir. eval_language_retention.py
# prefixes the run dir with the model name ("alm" or "base"), so its layout is
# language_retention/alm_<RUN_ID>/ and language_retention/base_<RUN_ID>/.
# (The base run gets the same RUN_ID since the namespaced id still uniquely
# identifies the experiment branch the comparison was made within.)
output_dir_for() {
    local bench=$1
    case "$bench" in
        language_retention/alm)  echo "$POOL/eval_results/language_retention/alm_$STEP" ;;
        language_retention/base) echo "$POOL/eval_results/language_retention/base_$STEP" ;;
        *)                       echo "$POOL/eval_results/$bench/$STEP" ;;
    esac
}

should_skip() {
    local bench=$1
    [ "$SKIP_COMPLETED" = 1 ] && [ -f "$(output_dir_for "$bench")/metrics.json" ]
}

# Benchmarks in LONGEST-FIRST order. Round-robin across N GPUs spreads the
# heavyweights out before doubling up. Format: "bench_name<TAB>python_args".
TAB=$'\t'
BENCHES=(
    "llm4mat${TAB}alm/eval/eval_llm4mat.py --checkpoint $CKPT --configs all --split validation --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
    "mattext${TAB}alm/eval/eval_mattext.py --checkpoint $CKPT --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
    "mat2props${TAB}alm/eval/eval_mat2props.py --checkpoint $CKPT --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
    "matterchat${TAB}alm/eval/eval_matterchat.py --checkpoint $CKPT --config matterchat_mp --split validation --data_csv $MC_VAL_CSV --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
    "gnome_fe${TAB}alm/eval/eval_gnome_fe.py --checkpoint $CKPT --split validation --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
    "mat2mcq${TAB}alm/eval/eval_mat2mcq.py --checkpoint $CKPT --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
    "language_retention/alm${TAB}alm/eval/eval_language_retention.py --model alm --checkpoint $CKPT $BLOCK_LEAK_FLAG"
    "language_retention/base${TAB}alm/eval/eval_language_retention.py --model base $BLOCK_LEAK_FLAG"
    "mascqa${TAB}alm/eval/eval_mascqa.py --checkpoint $CKPT --batch_size $BATCH_SIZE $BLOCK_LEAK_FLAG"
)

# Per-GPU job queues. queue[g] = newline-separated "bench<TAB>args" lines.
declare -a queue
for ((g=0; g<NUM_GPUS; g++)); do queue[$g]=""; done

for ((i=0; i<${#BENCHES[@]}; i++)); do
    g=$((i % NUM_GPUS))
    queue[$g]+="${BENCHES[$i]}"$'\n'
done

# Show the planned assignment so the user sees what's happening up-front.
echo "[main] assignment:"
for ((g=0; g<NUM_GPUS; g++)); do
    actual_gpu=${GPU_LIST[$g]}
    benches_on_gpu=$(printf "%s" "${queue[$g]}" | awk -F'\t' 'NF>0 {print $1}' | tr '\n' ' ')
    n=$(printf "%s" "${queue[$g]}" | grep -c .)
    printf "  GPU %s (%d jobs): %s\n" "$actual_gpu" "$n" "$benches_on_gpu"
done
echo ""

# Spawn one background worker per GPU; each runs its queue sequentially.
for ((g=0; g<NUM_GPUS; g++)); do
    actual_gpu=${GPU_LIST[$g]}
    q="${queue[$g]}"
    [ -z "$q" ] && continue
    (
        while IFS=$'\t' read -r bench args; do
            [ -z "$bench" ] && continue
            if should_skip "$bench"; then
                echo "[gpu=$actual_gpu] skip $bench (metrics.json exists)"
                continue
            fi
            out_dir="$(output_dir_for "$bench")"
            mkdir -p "$out_dir"
            log="$out_dir/run.log"
            echo "[gpu=$actual_gpu] start $bench → $log"
            # $args is intentionally unquoted so word-splitting passes flags through.
            CUDA_VISIBLE_DEVICES=$actual_gpu python $args > "$log" 2>&1 || \
                echo "[gpu=$actual_gpu] FAIL $bench (see $log)"
            echo "[gpu=$actual_gpu] done $bench"
        done <<< "$q"
    ) &
done

echo "[main] spawned $(jobs -r | wc -l) GPU workers; waiting..."
wait
echo "[main] all workers done"
echo ""
echo "=== summary ==="
for entry in "${BENCHES[@]}"; do
    bench="${entry%%$TAB*}"
    f="$(output_dir_for "$bench")/metrics.json"
    if [ -f "$f" ]; then
        printf "  ✓ %-30s\n" "$bench"
    else
        printf "  ✗ %-30s  (no metrics.json — see %s/run.log)\n" \
            "$bench" "$(output_dir_for "$bench")"
    fi
done
