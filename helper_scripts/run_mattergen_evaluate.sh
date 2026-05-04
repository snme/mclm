#!/usr/bin/env bash
# Run MatterGen's own `mattergen-evaluate` CLI on a directory of generated
# structures, producing the metrics.json schema used by their benchmark folder
# (https://github.com/microsoft/mattergen/tree/main/benchmark/metrics).
#
# This is the apples-to-apples way to compare against MatterGen's published
# numbers (S.U.N. 38.57%, RMSD 0.021 Å) — same matcher (disordered), same
# energy model (MatterSim), same reference dataset (Alex-MP-ICSD via the
# bundled MP2020 correction).
#
# Usage:
#   bash helper_scripts/run_mattergen_evaluate.sh <STRUCTURES_PATH> <DEST_DIR> [GPU]
#
# STRUCTURES_PATH: an .extxyz file (e.g. eval_dng.py's pre_relax.extxyz),
#                  a .zip of CIFs, or a directory of CIFs.
# DEST_DIR:        where to write mattergen_metrics.json + mattergen_metrics_detailed.json
# GPU:             CUDA_VISIBLE_DEVICES index (default 7).
#
# Examples:
#   bash helper_scripts/run_mattergen_evaluate.sh \
#       /home/sathyae/orcd/pool/eval_results/stage3b_dng_g00/run9_stage3b_2node_step=4500/pre_relax.extxyz \
#       /home/sathyae/orcd/pool/stage3a/eval_metrics/run9_step4500/mattergen_dng

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STRUCTURES_PATH DEST_DIR [GPU] [skip_relax]" >&2
  exit 1
fi

STRUCTURES="${1%/}"
DEST="${2%/}"
GPU="${3:-7}"
SKIP_RELAX="${4:-false}"

if [[ ! -e "$STRUCTURES" ]]; then
  echo "error: structures path missing: $STRUCTURES" >&2
  exit 1
fi
mkdir -p "$DEST"

# Resolve the MP2020-corrected hull reference (real bytes, not LFS pointer).
HULL=/home/sathyae/orcd/pool/eval_data/mp_hull/reference_MP2020correction.gz
if [[ ! -f "$HULL" ]]; then
  echo "error: hull reference missing at $HULL" >&2
  echo "       run: python helper_scripts/fetch_mp_hull.py" >&2
  exit 1
fi
if head -c 64 "$HULL" 2>/dev/null | grep -q "git-lfs.github.com/spec"; then
  echo "error: $HULL is still a Git LFS pointer (134 bytes), not the real ~833 MB gz." >&2
  echo "       run: cd external/mattergen && git lfs install --local && git lfs pull \\" >&2
  echo "              --include='data-release/alex-mp/reference_MP2020correction.gz' --exclude=''" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
# mattergen-evaluate uses the mattergen submodule; PYTHONPATH not strictly required
# because it's pip-installed as `mattergen` package via mattergen.egg-info, but we
# add it for safety.
export PYTHONPATH="/home/sathyae/mclm/external/mattergen:${PYTHONPATH:-}"

DEVICE="cuda"
if ! /home/sathyae/.conda/envs/llm/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[mg-eval] WARNING: CUDA not available, falling back to CPU (slow)."
  DEVICE="cpu"
fi

echo "[mg-eval] structures: $STRUCTURES"
echo "[mg-eval] hull ref:   $HULL"
echo "[mg-eval] device:     $DEVICE"
echo "[mg-eval] dest:       $DEST"
echo

# `--relax=True` runs MatterSim relaxation per structure (~15s/each).
# `--structure_matcher='disordered'` is MatterGen's published convention
# (more permissive than the CDVAE-style ordered matcher we use elsewhere).
RELAX_ARG="--relax=True"
if [[ "$SKIP_RELAX" == "true" ]]; then
  RELAX_ARG="--relax=False"
  echo "[mg-eval] WARNING: skipping relaxation — stability will be NaN (smoke-mode behavior)."
fi

mattergen-evaluate \
    "$STRUCTURES" \
    $RELAX_ARG \
    --structure_matcher='disordered' \
    --reference_dataset_path="$HULL" \
    --device="$DEVICE" \
    --save_as="$DEST/mattergen_metrics.json" \
    --save_detailed_as="$DEST/mattergen_metrics_detailed.json"

echo
echo "[mg-eval] DONE — metrics written to:"
echo "          $DEST/mattergen_metrics.json"
echo "          $DEST/mattergen_metrics_detailed.json"
echo
echo "[mg-eval] headline numbers:"
/home/sathyae/.conda/envs/llm/bin/python -c "
import json, sys
with open('$DEST/mattergen_metrics.json') as f:
    m = json.load(f)
def fmt(v):
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)
keys = sorted(m.keys())
for k in keys:
    print(f'    {k:40s} = {fmt(m[k])}')
"
