#!/usr/bin/env bash
# Set up the mattergen fork submodule for mclm Stage 3a.
#
# What this does:
#   1. Initializes the external/mattergen submodule (if not already).
#   2. Applies external/mattergen_alm_stage3a.patch onto the submodule. The patch
#      retargets pyproject.toml from cu118 → cu124 (H200 compat), bumps
#      pytorch-lightning from 2.0.6 → ≥2.4, adds the alm_embedding YAML
#      (registers AtomsMapper as the conditional_embedding_module of a new
#      adapter property), appends "alm_embedding" to PROPERTY_SOURCE_IDS,
#      and writes install_for_h200.sh.
#   3. Marks install_for_h200.sh executable (git diff doesn't preserve +x).
#   4. Reminds the operator how to install (in a fresh py3.10 env).
#
# Idempotent: re-running on an already-patched submodule will fail at step 2 and
# print a hint.
#
# Usage (from mclm root):
#   bash external/setup_mattergen.sh
#
# To re-baseline (e.g. after upstream microsoft/mattergen advances):
#   git submodule update --remote external/mattergen
#   git -C external/mattergen reset --hard <new commit>
#   bash external/setup_mattergen.sh

set -euo pipefail

MCLM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="${MCLM_ROOT}/external/mattergen"
PATCH="${MCLM_ROOT}/external/mattergen_alm_stage3a.patch"

if [[ ! -f "$PATCH" ]]; then
  echo "ERROR: patch not found at $PATCH"
  exit 1
fi

echo "[1/3] init submodule ${SUBMODULE} ..."
git -C "$MCLM_ROOT" submodule update --init external/mattergen

echo "[2/3] apply Stage 3a patch ..."
cd "$SUBMODULE"
if git diff --quiet HEAD; then
  if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "  patch applied."
  else
    echo "  ERROR: patch does not apply cleanly to current submodule HEAD."
    echo "  Likely the submodule is at a different commit than the patch was generated against."
    echo "  Run \`git -C $SUBMODULE reset --hard a245cf2\` and retry, or regenerate the patch."
    exit 1
  fi
else
  echo "  submodule already has uncommitted edits — skipping. (Run \`git -C $SUBMODULE reset --hard\` to start fresh.)"
fi

echo "[3/3] chmod +x install_for_h200.sh ..."
chmod +x "$SUBMODULE/install_for_h200.sh"

echo
echo "Submodule ready. To install MatterGen for H200:"
echo "  conda create -n mattergen python=3.10 -y"
echo "  conda activate mattergen"
echo "  cd $SUBMODULE"
echo "  bash install_for_h200.sh"
echo
echo "Then for Stage 3a fine-tune (with mattergen env active and AtomsMapper on PYTHONPATH):"
echo "  PYTHONPATH=$MCLM_ROOT/alm:\$PYTHONPATH mattergen-finetune \\"
echo "      adapter.pretrained_name=mattergen_base \\"
echo "      adapter.full_finetuning=False \\"
echo "      data_module=mp_20 \\"
echo "      data_module.root_dir=/home/sathyae/orcd/pool/stage3a/mattergen_dataset \\"
echo "      data_module.properties=[alm_embedding] \\"
echo "      +adapter/property_embeddings_adapt@adapter.property_embeddings_adapt.alm_embedding=alm_embedding"
