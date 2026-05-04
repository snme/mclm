#!/usr/bin/env bash
# Run the full Stage 3b *quantitative* eval suite (CSP + DNG + text-conditional)
# on a step= ckpt. Tees all output (stdout + stderr) to one log file. Designed
# to be re-runnable; previously-completed sub-runs land in distinct dirs under
# $ALM_EVAL_RESULTS_ROOT and aren't re-executed unless you delete them.
#
# Usage:
#   bash helper_scripts/eval_stage3b_metrics.sh CKPT_DIR [DEST_DIR] [GPU] [MODE]
#
# MODE:
#   smoke   (default) — small subsets; ~16 min on 1 GPU (DNG + text-cond)
#   full              — full DNG (1024 samples + relax) + 200-row text-cond; ~5h overnight
#
# OPTIONAL ENV FLAGS:
#   RUN_CSP=true      — also run CSP eval (MP-20 n=1, MP-20 n=20, MPTS-52 n=20).
#                       Adds ~25 min to smoke, ~12-72h to full. mclm CSP
#                       match-rate is empirically ~0% (resample N=100 hits
#                       0 lattice matches even at composition_match=2%), so
#                       this is OFF by default. Use for ablations / honesty
#                       table in writeup.
#
# Examples:
#   # Default: DNG + text-cond only
#   bash helper_scripts/eval_stage3b_metrics.sh \
#       /home/sathyae/orcd/pool/stage3_outputs/stage3a/ckpts/run9_stage3b_2node/step=4500
#
#   # With CSP for completeness
#   RUN_CSP=true bash helper_scripts/eval_stage3b_metrics.sh \
#       /home/sathyae/orcd/pool/stage3_outputs/stage3a/ckpts/run10_stage3b_K4_2node/step=5000 \
#       /home/sathyae/orcd/pool/stage3_outputs/stage3b/run10_K4_step5000 7 full

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CKPT_DIR [DEST_DIR] [GPU] [smoke|full]" >&2
  exit 1
fi

CKPT_DIR="${1%/}"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "error: $CKPT_DIR does not exist" >&2
  exit 1
fi
if [[ ! -f "$CKPT_DIR/atoms_mapper.pt" ]]; then
  echo "error: $CKPT_DIR/atoms_mapper.pt missing — pass a step=N dir." >&2
  exit 1
fi

RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
STEP_NAME="$(basename "$CKPT_DIR")"
RUN_TAG="${RUN_NAME}_${STEP_NAME}"

DEST="${2:-/home/sathyae/orcd/pool/stage3a/eval_metrics/${RUN_TAG}}"
GPU="${3:-7}"
MODE="${4:-smoke}"
mkdir -p "$DEST"

case "$MODE" in
  smoke)
    # Smoke = pipeline validation. Goal: confirm each script reads inputs,
    # generates, scores, and writes metrics.json end-to-end. Statistical
    # significance is NOT a goal here; use `full` for that.
    DNG_NUM_SAMPLES=32                 # ~5 min generation
    DNG_BATCH_SIZE=8
    TEXT_N_TEST_ROWS=10                # ~10 min
    TEXT_SAMPLES_PER_PROMPT=4
    DNG_SCORE_RELAX="--skip_relax"     # our eval_dng.py relax is ~15s/structure; skip for smoke
    MG_EVAL_SKIP_RELAX="true"          # mattergen-evaluate also skips relax in smoke
                                       # (would add another 32 × 15s = 8 min). Stability/SUN
                                       # in mattergen_metrics.json will be NaN; rerun in full
                                       # mode for the publishable number.
    TEXT_SCORE_ENERGY=""
    # CSP knobs (only used when RUN_CSP=true). Smoke uses 10 rows for plumbing.
    CSP_MAX_ROWS=10
    CSP_N=20
    CSP_PROMPT_TEMPLATE=rich_v1
    ;;
  full)
    DNG_NUM_SAMPLES=1024
    DNG_BATCH_SIZE=16
    TEXT_N_TEST_ROWS=200
    TEXT_SAMPLES_PER_PROMPT=8
    DNG_SCORE_RELAX=""
    MG_EVAL_SKIP_RELAX="false"         # mattergen-evaluate relaxes 1024 × ~15s ≈ 4.5 hr;
                                       # produces the headline S.U.N. number for the writeup.
    TEXT_SCORE_ENERGY="--score_energy"
    # CSP knobs (only used when RUN_CSP=true). Full benchmark = all rows.
    CSP_MAX_ROWS=-1
    CSP_N=20
    CSP_PROMPT_TEMPLATE=rich_v1
    ;;
  *)
    echo "unknown MODE '$MODE' — use 'smoke' or 'full'" >&2
    exit 1
    ;;
esac

# ─── Env ──────────────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export ALM_EVAL_RESULTS_ROOT="${ALM_EVAL_RESULTS_ROOT:-/home/sathyae/orcd/pool/eval_results}"

# Resolve repo root so we can call helpers regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LOGFILE="$DEST/eval_metrics.log"
ALM_CKPT="$CKPT_DIR"
AM="$CKPT_DIR/atoms_mapper.pt"

{
  echo "=================================================================="
  echo "Stage 3b QUANTITATIVE eval — $(date)"
  echo "  CKPT_DIR             $ALM_CKPT"
  echo "  AM                   $AM"
  echo "  DEST                 $DEST"
  echo "  GPU                  $CUDA_VISIBLE_DEVICES"
  echo "  MODE                 $MODE"
  echo "  ALM_EVAL_RESULTS_ROOT $ALM_EVAL_RESULTS_ROOT"
  echo "  RUN_CSP              ${RUN_CSP:-false}"
  echo "=================================================================="

  echo
  echo "===== 0. Pre-flight checks ====="
  if [[ ! -d /home/sathyae/orcd/pool/eval_data/csp/mp_20 ]]; then
    echo "[pre-flight] CSP benchmarks missing — running download script."
    bash helper_scripts/download_csp_benchmarks.sh
  else
    echo "[pre-flight] CSP benchmarks present at /home/sathyae/orcd/pool/eval_data/csp/"
  fi
  if [[ ! -f /home/sathyae/orcd/pool/eval_data/mp_hull/preferred.txt ]]; then
    echo "[pre-flight] MP hull reference missing — running fetch."
    python helper_scripts/fetch_mp_hull.py
  else
    echo "[pre-flight] MP hull reference present at /home/sathyae/orcd/pool/eval_data/mp_hull/"
  fi
  # The bundled MatterGen hull is a Git LFS file. Detect a still-pointer state.
  HULL_GZ=/home/sathyae/orcd/pool/eval_data/mp_hull/reference_MP2020correction.gz
  if [[ -f "$HULL_GZ" ]] && head -c 64 "$HULL_GZ" | grep -q "git-lfs.github.com/spec"; then
    echo "[pre-flight] WARNING: $HULL_GZ is still a Git LFS pointer."
    echo "             Run: cd external/mattergen && git lfs install --local && git lfs pull"
    echo "             Continuing — eval_dng.py will skip stability/SUN scoring."
  fi

  # CSP (formula+sg → exact MP entry) is OFF by default. The N=100 resample
  # experiment on mp-1225695 confirmed mclm hits 2/100 composition_match and
  # 0/100 lattice_match — mclm is text-conditioned generative, not a CSP
  # solver. CSP is the wrong evaluation framing for this architecture.
  #
  # To opt in, set RUN_CSP=true in the env:
  #   RUN_CSP=true bash helper_scripts/eval_stage3b_metrics.sh CKPT_DIR ...
  # Adds ~25 min to smoke (3 benchmarks × 10 rows × ~50s each) and ~12-72 hr
  # to full mode (the 9000-MP-row sweep was the long pole previously).

  if [[ "${RUN_CSP:-false}" == "true" ]]; then
    echo
    echo "===== CSP (1/3): MP-20 (n=1) ====="
    python alm/eval/eval_csp.py \
        --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
        --benchmark mp_20 --n 1 \
        --max_rows "$CSP_MAX_ROWS" \
        --guidance_factor 1.0 \
        --prompt_template "$CSP_PROMPT_TEMPLATE" \
        --run_id "$RUN_TAG"

    echo
    echo "===== CSP (2/3): MP-20 (n=$CSP_N) ====="
    python alm/eval/eval_csp.py \
        --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
        --benchmark mp_20 --n "$CSP_N" \
        --max_rows "$CSP_MAX_ROWS" \
        --guidance_factor 1.0 \
        --prompt_template "$CSP_PROMPT_TEMPLATE" \
        --run_id "$RUN_TAG"

    echo
    echo "===== CSP (3/3): MPTS-52 (n=$CSP_N) ====="
    python alm/eval/eval_csp.py \
        --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
        --benchmark mpts_52 --n "$CSP_N" \
        --max_rows "$CSP_MAX_ROWS" \
        --guidance_factor 1.0 \
        --prompt_template "$CSP_PROMPT_TEMPLATE" \
        --run_id "$RUN_TAG"
  else
    echo
    echo "===== CSP — SKIPPED (set RUN_CSP=true to enable; mclm CSP match-rate is ~0%) ====="
  fi

  echo
  echo "===== 1. De-novo (DNG) eval ====="
  python alm/eval/eval_dng.py \
      --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
      --num_samples "$DNG_NUM_SAMPLES" \
      --batch_size "$DNG_BATCH_SIZE" \
      --guidance_factor 0.0 \
      --run_id "$RUN_TAG" \
      $DNG_SCORE_RELAX

  echo
  echo "===== 1b. MatterGen-format DNG eval (apples-to-apples vs MatterGen 38.57% S.U.N.) ====="
  # Run MatterGen's own `mattergen-evaluate` CLI on the same generations so we
  # get their headline metrics (% Stable, % Unique, % Novel, % S.U.N., RMSD)
  # under their convention (disordered matcher, MatterSim relaxation, MP2020
  # corrected hull). This is the comparison number the writeup references.
  DNG_RAW="$ALM_EVAL_RESULTS_ROOT/stage3b_dng_g00/$RUN_TAG/pre_relax.extxyz"
  if [[ -f "$DNG_RAW" ]]; then
    bash helper_scripts/run_mattergen_evaluate.sh \
        "$DNG_RAW" \
        "$DEST/mattergen_dng" \
        "$CUDA_VISIBLE_DEVICES" \
        "$MG_EVAL_SKIP_RELAX" \
        || echo "[mg-eval] non-zero exit — check $DEST/mattergen_dng/ for partial output"
  else
    echo "[mg-eval] skipped — $DNG_RAW not found (eval_dng.py may have failed earlier)"
  fi

  echo
  echo "===== 2. Text-conditional eval ====="
  python alm/eval/eval_text_conditional.py \
      --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
      --n_test_rows "$TEXT_N_TEST_ROWS" \
      --samples_per_prompt "$TEXT_SAMPLES_PER_PROMPT" \
      --guidance_factor 1.0 \
      --run_id "$RUN_TAG" \
      $TEXT_SCORE_ENERGY

  echo
  echo "===== 3. Aggregate headline table ====="
  AGG_BENCHMARKS=("stage3b_dng_g00" "stage3b_text_cond_g10")
  if [[ "${RUN_CSP:-false}" == "true" ]]; then
    # CSP run dirs are suffixed with the prompt-template tag (e.g. _rich_v1).
    # If you switch CSP_PROMPT_TEMPLATE, update these names accordingly.
    SUFFIX=""
    if [[ "$CSP_PROMPT_TEMPLATE" != "minimal" ]]; then
      SUFFIX="_${CSP_PROMPT_TEMPLATE}"
    fi
    AGG_BENCHMARKS+=(
      "stage3b_csp_mp_20_n1_g10${SUFFIX}"
      "stage3b_csp_mp_20_n${CSP_N}_g10${SUFFIX}"
      "stage3b_csp_mpts_52_n${CSP_N}_g10${SUFFIX}"
    )
  fi
  python evals/aggregate_results.py --run_id "$RUN_TAG" \
      --benchmarks "${AGG_BENCHMARKS[@]}" \
      || echo "[aggregate] aggregate_results.py exited non-zero — table may be partial"

  echo
  echo "=================================================================="
  echo "DONE — $(date)"
  echo "  log: $LOGFILE"
  echo "  per-bench results under $ALM_EVAL_RESULTS_ROOT/stage3b_*/$RUN_TAG/"
  echo "=================================================================="
} 2>&1 | tee "$LOGFILE"
