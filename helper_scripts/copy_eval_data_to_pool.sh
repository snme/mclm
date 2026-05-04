#!/bin/bash
# Stage all eval-only data into pool. Idempotent (rsync --update + skip-if-exists).
# Skips data that's already staged; LLM4Mat-Bench + MaScQA + GPT-Narratives are
# expected to be staged separately by copy_ds_to_tmp_stage2.sh and friends.
#
# Run on a node with internet access (login node usually works for HF; Zenodo
# downloads benefit from a compute node with more bandwidth).

set -euo pipefail
POOL=${POOL:-/home/sathyae/orcd/pool}
EVAL_DATA="$POOL/eval_data"
mkdir -p "$EVAL_DATA"

echo "[copy_eval_data] POOL=$POOL → $EVAL_DATA"

# ---- MatterChat 142,899 MP structures (Zenodo record 18735961) ----
MATTERCHAT_DIR="$EVAL_DATA/MatterChat"
MATTERCHAT_ZIP="https://zenodo.org/records/18735961/files/Dataset_MatterChat.zip?download=1"
if [[ ! -d "$MATTERCHAT_DIR" ]]; then
    echo "[copy_eval_data] MatterChat: downloading $MATTERCHAT_ZIP → $MATTERCHAT_DIR/"
    mkdir -p "$MATTERCHAT_DIR"
    tmp=$(mktemp --suffix=.zip)
    curl -L -o "$tmp" "$MATTERCHAT_ZIP"
    unzip -q -d "$MATTERCHAT_DIR" "$tmp"
    rm -f "$tmp"
fi

# ---- Park et al. GPT-Narratives-for-Materials code (Mat2Props / Mat2MCQ) ----
PARK_REPO="$POOL/GPT-Narratives-for-Materials/code"
if [[ ! -d "$PARK_REPO/.git" ]]; then
    echo "[copy_eval_data] git clone parkyjmit/GPT-Narratives-for-Materials → $PARK_REPO"
    git clone https://github.com/parkyjmit/GPT-Narratives-for-Materials.git "$PARK_REPO"
fi

# ---- HF datasets (MMLU + GSM8K + MatText + GPQA) ----
# Pull into a writable HF cache. GPQA is gated; you must have run
# `huggingface-cli login` with an account that's been approved.
export HF_HOME="${HF_HOME:-$EVAL_DATA/hf_cache}"
mkdir -p "$HF_HOME"
echo "[copy_eval_data] HF_HOME=$HF_HOME — pulling datasets…"
python - <<'PY'
import os
from datasets import load_dataset
for name, cfg, split in [
    ("cais/mmlu", "all", "test"),
    ("cais/mmlu", "all", "dev"),
    ("openai/gsm8k", "main", "test"),
    ("n0w0f/MatText", "perovskites-test-filtered", "test"),
    ("n0w0f/MatText", "kvrh-test-filtered", "test"),
    ("n0w0f/MatText", "gvrh-test-filtered", "test"),
]:
    print(f"  load_dataset({name!r}, {cfg!r}, split={split!r})")
    load_dataset(name, cfg, split=split)
# GPQA is gated. Requires `huggingface-cli login` w/ an approved account.
try:
    load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print("  GPQA Diamond cached.")
except Exception as e:
    print(f"  GPQA skipped: {type(e).__name__}: {e}")
PY

echo "[copy_eval_data] done."
