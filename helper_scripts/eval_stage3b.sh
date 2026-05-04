#!/usr/bin/env bash
# Run the full Stage 3b eval suite against a given step checkpoint:
#   1. Cosine-sim diagnostic (5-prompt cond-vector spread)
#   2. Linear probes (presence + count/stoichiometry) on 800 held-out narratives
#   3. ID-prompt behavioral generation (4 narratives × G_VALUES) — multi-GPU when available
#   3b. OOD-prompt behavioral generation (4 prompts × G_VALUES) — opt-in via RUN_OOD
#   3c. Unconditional baseline g=0.0 — opt-in via RUN_UNCOND
#   4. Per-cell summaries
#   5. Wider conditional DNG → MSUN (N prompts from pairs.parquet at DNG_GUIDANCE,
#      MatterSim relax + disordered matcher + MP2020 hull) — opt-in via RUN_DNG_MSUN
#      DNG generation parallelized across CUDA_VISIBLE_DEVICES via prompt-slicing.
#
# Two output files:
#   $DEST/eval_run.log       full noise (tqdm, warnings, every step) — for debug
#   $DEST/eval_summary.txt   curated: section headers, key metrics extracted from
#                            JSON/TSV files, full paths to all artifacts. NO tqdm.
#
# Usage:
#   bash helper_scripts/eval_stage3b.sh CKPT_DIR [DEST_DIR] [GPU]
#
# Multi-GPU: pass `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (or whichever subset)
# and the script auto-parallelizes step 3a (ID prompts) and step 5a (DNG generation).
# Single-GPU also works (CUDA_VISIBLE_DEVICES=7) — same code path, no parallel branches.
#
# Env-var knobs (with defaults):
#   G_VALUES="1.0 2.0"     space-separated guidance factors for behavioral cells
#   RUN_OOD=true           include 4 OOD prompts at each G_VALUES cell
#   RUN_UNCOND=true        include the g=0.0 unconditional baseline
#   RUN_DNG_MSUN=true      run step 5 (conditional DNG + MSUN); ~5-12 min/g depending on N_GPUS
#   DNG_NUM_SAMPLES=32     # of conditional DNG prompts (1 sample per prompt)
#   DNG_GUIDANCE=1.0       guidance factor for the conditional DNG eval
#
# Examples:
#   # Apples-to-apples vs run9 — only g=1.0 ID prompts + DNG+MSUN at g=1.0, on 8 GPUs
#   G_VALUES="1.0" RUN_OOD=false RUN_UNCOND=false \
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash helper_scripts/eval_stage3b.sh \
#       /home/sathyae/orcd/pool/stage3_outputs/stage3a/ckpts/run13_stage3b_r128_presence/step=5000
#
#   # Wider sweep (default), full G_VALUES + OOD + uncond + DNG@1.0
#   bash helper_scripts/eval_stage3b.sh \
#       /home/sathyae/orcd/pool/stage3_outputs/stage3a/ckpts/run9_stage3b_2node/step=4500

set -euo pipefail

# ─── Args ─────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CKPT_DIR [DEST_DIR] [GPU]" >&2
  exit 1
fi

CKPT_DIR="${1%/}"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "error: CKPT_DIR does not exist: $CKPT_DIR" >&2
  exit 1
fi
if [[ ! -f "$CKPT_DIR/atoms_mapper.pt" ]]; then
  echo "error: $CKPT_DIR/atoms_mapper.pt missing — pass a step=N dir." >&2
  exit 1
fi

RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
STEP_NAME="$(basename "$CKPT_DIR")"
RUN_TAG="${RUN_NAME}_${STEP_NAME}"

DEST="${2:-/home/sathyae/orcd/pool/stage3_outputs/stage3a/generations/${RUN_TAG}}"
# GPU resolution order: positional arg > existing CUDA_VISIBLE_DEVICES > 7.
GPU="${3:-${CUDA_VISIBLE_DEVICES:-7}}"

# ─── Env ──────────────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EVAL="${EVAL:-/home/sathyae/orcd/pool/stage3_outputs/stage3a/eval_prompts}"
export PAIRS_PARQUET="${PAIRS_PARQUET:-/home/sathyae/orcd/pool/stage3_outputs/stage3a/pairs.parquet}"
export ALM_CKPT="$CKPT_DIR"
export AM="$CKPT_DIR/atoms_mapper.pt"
export DEST
export PYTHONPATH="/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

# Parametrized knobs
G_VALUES="${G_VALUES:-1.0 2.0}"
RUN_OOD="${RUN_OOD:-true}"
RUN_UNCOND="${RUN_UNCOND:-true}"
RUN_DNG_MSUN="${RUN_DNG_MSUN:-true}"
DNG_NUM_SAMPLES="${DNG_NUM_SAMPLES:-32}"
DNG_GUIDANCE="${DNG_GUIDANCE:-1.0}"

# Parse GPU list for multi-GPU parallelization. Single GPU → list of length 1.
IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS=${#GPU_LIST[@]}

mkdir -p "$DEST"
LOGFILE="$DEST/eval_run.log"
SUMMARYFILE="$DEST/eval_summary.txt"
DIAG_DIR="/home/sathyae/orcd/pool/stage3_outputs/stage3a/diagnostics/${RUN_TAG}"
mkdir -p "$DIAG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Helpers: g="1.0" → "1_0" filesystem-safe; g="1.0" → "10" eval_dng.py bench_int.
g_tag () { echo "${1//./_}"; }
g_int () { python3 -c "import sys; print(f'{int(float(sys.argv[1])*10):02d}')" "$1"; }

# Run a single (tag, g, prompt) generation cell. Caller must export
# CUDA_VISIBLE_DEVICES if pinning to a specific GPU (multi-GPU branch does this).
run_gen () {
  local tag="$1" g="$2" prompt="$3"
  local out=/tmp/g$(g_tag "$g")_${tag}_$$_$RANDOM
  python helper_scripts/generate_stage3a.py \
      --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
      --prompt "$prompt" --out_dir "$out" \
      --batch_size 4 --num_batches 1 --diffusion_guidance_factor "$g"
  python helper_scripts/visualize_stage3a_gen.py \
      --gen_dir "$out" --dest "$DEST/g$(g_tag "$g")_${tag}"
  rm -rf "$out"
}

declare -A OOD_PROMPTS=(
  [perovskite]="A stable cubic perovskite ABO3 where A is calcium and B is a transition metal, suitable for photovoltaic applications."
  [cathode]="An olivine-structured lithium-iron phosphate cathode material for lithium-ion batteries."
  [topological]="A bismuth-tellurium binary topological insulator with rhombohedral symmetry."
  [binary_alloy]="A binary intermetallic alloy of nickel and titanium with shape-memory properties."
)

ID_TAGS=(v2gafe laclau livre limnpoh)
id_prompt_for () {
  case "$1" in
    v2gafe)  cat "$EVAL/mp_3d_2020-118833.narrative.txt" ;;
    laclau)  cat "$EVAL/aflow2-113933.narrative.txt" ;;
    livre)   cat "$EVAL/oqmd-35291.narrative.txt" ;;
    limnpoh) cat "$EVAL/dft_3d-630.narrative.txt" ;;
    *) echo "unknown id tag: $1" >&2; return 1 ;;
  esac
}

# Run a list of "tag g" pairs in parallel, one per GPU, batched at N_GPUS at a time.
# Args: list of "tag|g|prompt" entries (pipe-separated since prompts contain spaces).
run_cells_parallel () {
  local i=0
  for entry in "$@"; do
    local tag="${entry%%|*}"
    local rest="${entry#*|}"
    local g="${rest%%|*}"
    local prompt="${rest#*|}"
    local gpu="${GPU_LIST[$((i % N_GPUS))]}"
    (CUDA_VISIBLE_DEVICES="$gpu" run_gen "$tag" "$g" "$prompt") &
    i=$((i + 1))
    if (( i % N_GPUS == 0 )); then wait; fi
  done
  wait
}

# ─── Run pipeline (full output → LOGFILE; summary built at end → SUMMARYFILE) ──
{
  echo "=================================================================="
  echo "Stage 3b eval — $(date)"
  echo "  CKPT_DIR        $ALM_CKPT"
  echo "  AM              $AM"
  echo "  DEST            $DEST"
  echo "  DIAG_DIR        $DIAG_DIR"
  echo "  CUDA devices    $CUDA_VISIBLE_DEVICES  (N_GPUS=$N_GPUS)"
  echo "  G_VALUES        $G_VALUES"
  echo "  RUN_OOD         $RUN_OOD"
  echo "  RUN_UNCOND      $RUN_UNCOND"
  echo "  RUN_DNG_MSUN    $RUN_DNG_MSUN"
  if [[ "$RUN_DNG_MSUN" == "true" ]]; then
    echo "  DNG_NUM_SAMPLES $DNG_NUM_SAMPLES"
    echo "  DNG_GUIDANCE    $DNG_GUIDANCE"
  fi
  echo "=================================================================="

  echo
  echo "===== 1. Diagnostic (cosine matrix + weight norms) ====="
  # Single GPU — ALM forward on 5 narratives is fast and not worth parallelizing.
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" \
  python helper_scripts/diagnose_conditioning.py \
      --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" --eval_dir "$EVAL"

  echo
  echo "===== 2. Linear probes (presence + count) on 800 held-out narratives ====="
  # Single GPU — probe is one Python process; not worth multi-GPU.
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" \
  python helper_scripts/probe_atoms_mapper_clusters.py \
      --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
      --pairs_parquet  "$PAIRS_PARQUET" \
      --out_dir        "$DIAG_DIR" \
      --n_samples      800

  echo
  echo "===== 3a. ID prompts × G_VALUES=($G_VALUES)  (parallel: $N_GPUS GPUs) ====="
  ID_CELLS=()
  for tag in "${ID_TAGS[@]}"; do
    PROMPT="$(id_prompt_for "$tag")"
    for g in $G_VALUES; do
      ID_CELLS+=("${tag}|${g}|${PROMPT}")
    done
  done
  run_cells_parallel "${ID_CELLS[@]}"

  if [[ "$RUN_OOD" == "true" ]]; then
    echo
    echo "===== 3b. OOD prompts × G_VALUES=($G_VALUES)  (parallel: $N_GPUS GPUs) ====="
    OOD_CELLS=()
    for tag in "${!OOD_PROMPTS[@]}"; do
      for g in $G_VALUES; do
        OOD_CELLS+=("ood_${tag}|${g}|${OOD_PROMPTS[$tag]}")
      done
    done
    run_cells_parallel "${OOD_CELLS[@]}"
  else
    echo
    echo "===== 3b. OOD prompts — SKIPPED (RUN_OOD=false) ====="
  fi

  if [[ "$RUN_UNCOND" == "true" ]]; then
    echo
    echo "===== 3c. Unconditional baseline (g=0.0) ====="
    out_uncond=/tmp/g0_uncond_$$
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" \
    python helper_scripts/generate_stage3a.py \
        --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
        --prompt "ignored" --out_dir "$out_uncond" \
        --batch_size 4 --num_batches 1 --diffusion_guidance_factor 0.0
    python helper_scripts/visualize_stage3a_gen.py \
        --gen_dir "$out_uncond" --dest "$DEST/g0_uncond"
    rm -rf "$out_uncond"
  else
    echo
    echo "===== 3c. Unconditional baseline — SKIPPED (RUN_UNCOND=false) ====="
  fi

  if [[ "$RUN_DNG_MSUN" == "true" ]]; then
    echo
    echo "===== 5. Wider conditional DNG → MSUN (parallel: $N_GPUS GPUs) ====="
    DNG_RUN_ID="${RUN_TAG}_n${DNG_NUM_SAMPLES}_cond_g$(g_int "$DNG_GUIDANCE")"
    EVAL_ROOT="${ALM_EVAL_RESULTS_ROOT:-/home/sathyae/orcd/pool/eval_results}"
    BENCH="stage3b_dng_g$(g_int "$DNG_GUIDANCE")"
    MSUN_DEST="$DEST/dng_msun_g$(g_int "$DNG_GUIDANCE")"

    echo "       N=$DNG_NUM_SAMPLES, g=$DNG_GUIDANCE, sliced across $N_GPUS GPUs"

    chunk=$(( (DNG_NUM_SAMPLES + N_GPUS - 1) / N_GPUS ))
    DNG_GEN_DIRS=()
    pids=()
    for gpu_i in "${!GPU_LIST[@]}"; do
      start=$((gpu_i * chunk))
      end=$(( (gpu_i + 1) * chunk ))
      (( end > DNG_NUM_SAMPLES )) && end=$DNG_NUM_SAMPLES
      if (( start >= end )); then continue; fi
      this_run_id="${DNG_RUN_ID}_gpu${gpu_i}"
      this_dir="$EVAL_ROOT/$BENCH/$this_run_id"
      DNG_GEN_DIRS+=("$this_dir")
      echo "  [gpu ${GPU_LIST[$gpu_i]}] slice [$start:$end] → $this_run_id"
      ( CUDA_VISIBLE_DEVICES="${GPU_LIST[$gpu_i]}" \
        python alm/eval/eval_dng.py \
            --alm_checkpoint "$ALM_CKPT" --atoms_mapper "$AM" \
            --num_samples "$DNG_NUM_SAMPLES" --batch_size 1 \
            --guidance_factor "$DNG_GUIDANCE" \
            --prompts_from_parquet "$PAIRS_PARQUET" \
            --prompts_seed 1337 \
            --prompt_slice_start "$start" --prompt_slice_end "$end" \
            --run_id "$this_run_id" \
            --skip_relax ) &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done

    echo
    echo "  --- 5b. concat per-GPU pre_relax.extxyz files + score_manual_gens.sh ---"
    MERGED_DIR="$DEST/dng_parallel_gens"
    mkdir -p "$MERGED_DIR"
    # Symlink each gpu's output dir under one parent so score_manual_gens.sh's
    # recursive glob picks them all up. This is more robust than concatenating
    # extxyz files (preserves per-prompt prov + handles missing files cleanly).
    for d in "${DNG_GEN_DIRS[@]}"; do
      if [[ -d "$d" ]]; then
        ln -sfn "$d" "$MERGED_DIR/$(basename "$d")"
      fi
    done
    bash helper_scripts/score_manual_gens.sh \
        "$MERGED_DIR" \
        "$MSUN_DEST" \
        "${GPU_LIST[0]}"
  else
    echo
    echo "===== 5. Wider conditional DNG → MSUN — SKIPPED (RUN_DNG_MSUN=false) ====="
  fi

  echo
  echo "=================================================================="
  echo "DONE — $(date)"
  echo "  raw log:        $LOGFILE"
  echo "  curated:        $SUMMARYFILE  (built next)"
  echo "=================================================================="
} 2>&1 | tee "$LOGFILE"

# ─── Build the curated summary file (no tqdm, no warnings; only key results) ──
{
  echo "============================================================"
  echo "Stage 3b Eval Summary — $(date)"
  echo "============================================================"
  echo "  CKPT_DIR        $ALM_CKPT"
  echo "  DEST            $DEST"
  echo "  CUDA devices    $CUDA_VISIBLE_DEVICES  (N_GPUS=$N_GPUS)"
  echo "  G_VALUES        $G_VALUES"
  if [[ "$RUN_DNG_MSUN" == "true" ]]; then
    echo "  DNG: N=$DNG_NUM_SAMPLES, g=$DNG_GUIDANCE"
  fi
  echo

  PROBE_JSON="$DIAG_DIR/probe_metrics.json"
  if [[ -f "$PROBE_JSON" ]]; then
    echo "── 2. Probe results (presence + count/stoichiometry) ─────────"
    /home/sathyae/.conda/envs/llm/bin/python3 - <<PYEOF
import json, sys
m = json.load(open("$PROBE_JSON"))
def f(x):
    if x is None: return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)
print(f"  presence precision     {f(m.get('final_val_precision'))}")
print(f"  presence recall        {f(m.get('final_val_recall'))}")
print(f"  count present_mae      {f(m.get('final_count_present_mae'))}")
print(f"  count present_exact    {f(m.get('final_count_present_exact_pct'))}")
print(f"  whole_composition_exact{'' if 'final_whole_composition_exact_pct' not in m else ''} {f(m.get('final_whole_composition_exact_pct'))}")
print(f"  reduced_formula_exact  {f(m.get('final_reduced_formula_exact_pct'))}")
print(f"  verdict                {m.get('verdict','')}")
PYEOF
    echo
  fi

  echo "── 3. ID prompt cells (formulas + densities + lattice) ──────"
  for tag in "${ID_TAGS[@]}"; do
    for g in $G_VALUES; do
      cell="$DEST/g$(g_tag "$g")_${tag}/summary.tsv"
      if [[ -f "$cell" ]]; then
        echo
        echo "── g=${g}  prompt=${tag}"
        cat "$cell"
      fi
    done
  done

  if [[ "$RUN_OOD" == "true" ]]; then
    echo
    echo "── 3b. OOD prompt cells ─────────────────────────────────────"
    for tag in "${!OOD_PROMPTS[@]}"; do
      for g in $G_VALUES; do
        cell="$DEST/g$(g_tag "$g")_ood_${tag}/summary.tsv"
        if [[ -f "$cell" ]]; then
          echo
          echo "── g=${g}  prompt=ood_${tag}"
          cat "$cell"
        fi
      done
    done
  fi

  if [[ "$RUN_UNCOND" == "true" && -f "$DEST/g0_uncond/summary.tsv" ]]; then
    echo
    echo "── 3c. g=0.0 unconditional baseline ─────────────────────────"
    cat "$DEST/g0_uncond/summary.tsv"
  fi

  if [[ "$RUN_DNG_MSUN" == "true" ]]; then
    MSUN_DEST="$DEST/dng_msun_g$(g_int "$DNG_GUIDANCE")"
    MGM_JSON="$MSUN_DEST/mattergen_metrics.json"
    echo
    echo "── 5. Wider conditional DNG → MSUN headline ─────────────────"
    if [[ -f "$MGM_JSON" ]]; then
      /home/sathyae/.conda/envs/llm/bin/python3 - <<PYEOF
import json
m = json.load(open("$MGM_JSON"))
def fmt(v):
    if isinstance(v, dict): v = v.get("value")
    return f"{v:.4f}" if isinstance(v, float) else str(v)
keys = [
    'frac_stable_structures',
    'frac_novel_unique_stable_structures',
    'avg_energy_above_hull_per_atom',
    'avg_rmsd_from_relaxation',
    'avg_comp_validity',
    'avg_structure_validity',
    'frac_novel_structures',
    'frac_novel_systems',
    'frac_unique_structures',
    'precision',
]
for k in keys:
    if k in m: print(f"  {k:42s} = {fmt(m[k])}")
PYEOF
    else
      echo "  (mattergen_metrics.json missing — see $LOGFILE)"
    fi
  fi

  echo
  echo "── Output files (full paths for reference) ──────────────────"
  echo "  raw log:                  $LOGFILE"
  echo "  this summary:             $SUMMARYFILE"
  if [[ -f "$PROBE_JSON" ]]; then
    echo "  probe metrics JSON:       $PROBE_JSON"
    echo "  probe arrays NPZ:         $DIAG_DIR/probe_arrays.npz"
    echo "  probe PCA/UMAP plots:     $DIAG_DIR/{pca2,umap}_by_*.png"
  fi
  echo "  ID generation summaries:  $DEST/g{g}_{prompt}/summary.tsv"
  echo "  ID PNG renders:           $DEST/g{g}_{prompt}/structure_*.png"
  if [[ "$RUN_DNG_MSUN" == "true" ]]; then
    MSUN_DEST="$DEST/dng_msun_g$(g_int "$DNG_GUIDANCE")"
    echo "  DNG per-GPU outputs:      $DEST/dng_parallel_gens/*/pre_relax.extxyz"
    echo "  MSUN headline JSON:       $MSUN_DEST/mattergen_metrics.json"
    echo "  MSUN per-structure JSON:  $MSUN_DEST/mattergen_metrics_detailed.json"
    echo "  MSUN merged extxyz:       $MSUN_DEST/merged.extxyz"
    echo "  MSUN dropped log:         $MSUN_DEST/merged_dropped.txt"
    echo "  MSUN evaluate log:        $MSUN_DEST/mattergen_evaluate.log"
  fi
} > "$SUMMARYFILE"

echo
echo "[done] full log:      $LOGFILE"
echo "[done] curated:       $SUMMARYFILE"
