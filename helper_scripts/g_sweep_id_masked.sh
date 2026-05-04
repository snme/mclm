#!/usr/bin/env bash
# CFG sweep across the 4 in-distribution behavioral prompts (v2gafe, laclau,
# livre, limnpoh) with **inference-time element masking ON**, so each prompt's
# generations are constrained to the elements named in its narrative.
#
# Mirrors the L4300 inline-loop sweep that produced the 6-point CFG response
# table (g=0/0.5/1.0/1.3/1.5/2.0 × 4 prompts × 4 samples), but:
#   - paths updated to /stage3_outputs/stage3a/...
#   - generate_stage3a.py now takes --allowed_elements, so each prompt's mask is
#     plumbed through the score-model atomic-numbers head
#   - default tag set adds the suffix _masked so the generations sit alongside
#     the prior unmasked sweep without overwriting
#
# Read this against the prior sweep: same prompts, same g values, same sample
# counts. Anything that differs in the per-cell summary.tsv is attributable to
# masking, not to a different model state or prompt.
#
# Usage:
#   bash helper_scripts/g_sweep_id_masked.sh CKPT_DIR [DEST_DIR] [GPU] [G_VALUES]
#
#   CKPT_DIR    Stage 3b step= dir (must contain atoms_mapper.pt + lora_adapter/)
#   DEST_DIR    output root (default /home/sathyae/orcd/pool/stage3_outputs/stage3a/generations/<run-tag>_masked_sweep)
#   GPU         CUDA_VISIBLE_DEVICES (default 7)
#   G_VALUES    space-separated guidance factors (default "0.5 1.0 1.3 1.5 2.0")
#
# Wallclock: 4 prompts × |G_VALUES| × ~50s/cell ≈ 4 × 5 × 50s = ~17 min.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CKPT_DIR [DEST_DIR] [GPU] [G_VALUES]" >&2
  exit 1
fi

CKPT_DIR="${1%/}"
if [[ ! -f "$CKPT_DIR/atoms_mapper.pt" ]]; then
  echo "error: $CKPT_DIR/atoms_mapper.pt missing — pass a step=N dir." >&2
  exit 1
fi

RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
STEP_NAME="$(basename "$CKPT_DIR")"
RUN_TAG="${RUN_NAME}_${STEP_NAME}"

DEST="${2:-/home/sathyae/orcd/pool/stage3_outputs/stage3a/generations/${RUN_TAG}_masked_sweep}"
# GPU resolution order: explicit positional arg > already-exported CUDA_VISIBLE_DEVICES > 7.
# This makes `CUDA_VISIBLE_DEVICES=3 bash ...` work as expected.
GPU="${3:-${CUDA_VISIBLE_DEVICES:-7}}"
G_VALUES="${4:-0.5 1.0 1.3 1.5 2.0}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export EVAL="${EVAL:-/home/sathyae/orcd/pool/stage3_outputs/stage3a/eval_prompts}"
export ALM_CKPT="$CKPT_DIR"
export AM="$CKPT_DIR/atoms_mapper.pt"
export DEST
export PYTHONPATH="/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$DEST"
LOGFILE="$DEST/sweep.log"

# Resolve script root so we run python helpers from repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ─── ID prompts + their target element sets ──────────────────────────────
# (the same 4 used in eval_stage3b.sh and the original L4300 sweep)
declare -A ID_PROMPT_FILE=(
    [v2gafe]="$EVAL/mp_3d_2020-118833.narrative.txt"
    [laclau]="$EVAL/aflow2-113933.narrative.txt"
    [livre]="$EVAL/oqmd-35291.narrative.txt"
    [limnpoh]="$EVAL/dft_3d-630.narrative.txt"
)
# Allowed-element set per prompt (parsed from the narrative tag — see
# helper_scripts/eval_stage3b.sh comments). Masking forbids the score model
# from sampling outside this set at every denoising step.
declare -A ID_PROMPT_ELEMENTS=(
    [v2gafe]="V,Ga,Fe"
    [laclau]="La,Cl,Au"
    [livre]="Li,V,Re"
    [limnpoh]="Li,Mn,P,O,H"
)

run_gen () {
  local tag="$1" g="$2" prompt="$3" mask="$4"
  local cell_dir="$DEST/g${g/./_}_${tag}"
  # Skip if a previous run of the same cell completed (the visualizer's
  # final-output marker is summary.tsv + the copied extxyz). Set FORCE=1 to
  # re-run a cell even if outputs exist.
  if [[ -z "${FORCE:-}" \
        && -s "$cell_dir/summary.tsv" \
        && -s "$cell_dir/generated_crystals.extxyz" ]]; then
    echo "[skip] $cell_dir already has summary.tsv + generated_crystals.extxyz "
    echo "       (set FORCE=1 to re-run)"
    return 0
  fi
  local out=/tmp/g${g/./_}_${tag}_masked_$$
  python helper_scripts/generate_stage3a.py \
      --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
      --prompt "$prompt" --out_dir "$out" \
      --batch_size 4 --num_batches 1 \
      --diffusion_guidance_factor "$g" \
      --allowed_elements "$mask"
  python helper_scripts/visualize_stage3a_gen.py \
      --gen_dir "$out" --dest "$cell_dir"
  rm -rf "$out"
}

{
  echo "=================================================================="
  echo "Stage 3b CFG sweep — masked variant — $(date)"
  echo "  CKPT_DIR  $ALM_CKPT"
  echo "  AM        $AM"
  echo "  DEST      $DEST"
  echo "  GPU       $CUDA_VISIBLE_DEVICES"
  echo "  G_VALUES  $G_VALUES"
  echo "  prompts   ${!ID_PROMPT_FILE[@]}"
  for tag in "${!ID_PROMPT_FILE[@]}"; do
    echo "    $tag → mask={${ID_PROMPT_ELEMENTS[$tag]}}"
  done
  echo "=================================================================="

  for tag in v2gafe laclau livre limnpoh; do
    PROMPT="$(cat "${ID_PROMPT_FILE[$tag]}")"
    MASK="${ID_PROMPT_ELEMENTS[$tag]}"
    for g in $G_VALUES; do
      echo
      echo "===== $tag @ g=$g  (mask=$MASK) ====="
      run_gen "$tag" "$g" "$PROMPT" "$MASK"
    done
  done

  echo
  echo "===== SUMMARIES ====="
  for tag in v2gafe laclau livre limnpoh; do
    for g in $G_VALUES; do
      g_norm="${g/./_}"
      summary="$DEST/g${g_norm}_${tag}/summary.tsv"
      if [[ -f "$summary" ]]; then
        echo
        echo "--- g=${g}  prompt=${tag}  mask=${ID_PROMPT_ELEMENTS[$tag]} ---"
        cat "$summary"
      fi
    done
  done

  echo
  echo "=================================================================="
  echo "DONE — $(date)"
  echo "  log:           $LOGFILE"
  echo "  generations:   $DEST"
  echo "  Compare against the unmasked sweep at:"
  echo "    /home/sathyae/orcd/pool/stage3a/generations/${RUN_TAG}/g{$G_VALUES}_{v2gafe,laclau,livre,limnpoh}/"
  echo "=================================================================="
} 2>&1 | tee "$LOGFILE"
