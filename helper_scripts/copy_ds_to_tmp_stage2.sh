#!/bin/bash
# Stage all Stage 2 data into /tmp for fast DataLoader access.
# Idempotent: rsync --update skips files already at /tmp with an equal-or-newer mtime.
set -e

POOL="/home/sathyae/orcd/pool"
DST_TMP="/tmp"
MODEL="orb_v3_direct_20_omat"

# --- 1. LLM4Mat-Bench CSVs + DBs + cached OrbV3 flats ---
for ds in cantor_hea gnome hmof jarvis_dft jarvis_qetb mp omdb oqmd qmof snumat; do
  mkdir -p "${DST_TMP}/LLM4Mat-Bench/${ds}"
  mkdir -p "${DST_TMP}/cached_embs/${ds}/embeddings"
  for split in train validation; do
    for ext in csv db id_index.json; do
      rsync -a --update \
        "${POOL}/LLM4Mat-Bench/${ds}/${split}.${ext}" \
        "${DST_TMP}/LLM4Mat-Bench/${ds}/${split}.${ext}"
    done
    for ext in flat.bin flat.idx.json; do
      rsync -a --update \
        "${POOL}/cached_embs/${ds}/embeddings/${MODEL}_${split}_atom.${ext}" \
        "${DST_TMP}/cached_embs/${ds}/embeddings/${MODEL}_${split}_atom.${ext}"
    done
  done
  echo "LLM4Mat/${ds}: done"
done

# --- 2. GPT-Narratives parquets + cached OrbV3 flats ---
mkdir -p "${DST_TMP}/GPT-Narratives-for-Materials"
rsync -a --update \
  "${POOL}/GPT-Narratives-for-Materials/"*.parquet \
  "${DST_TMP}/GPT-Narratives-for-Materials/"
for name in dft_3d mp_3d_2020 aflow2 oqmd; do
  emb_src="${POOL}/cached_embs_narratives/${name}/embeddings"
  emb_dst="${DST_TMP}/cached_embs_narratives/${name}/embeddings"
  if [ -d "${emb_src}" ]; then
    mkdir -p "${emb_dst}"
    rsync -a --update "${emb_src}/${MODEL}_atom.flat.bin"      "${emb_dst}/" 2>/dev/null || true
    rsync -a --update "${emb_src}/${MODEL}_atom.flat.idx.json" "${emb_dst}/" 2>/dev/null || true
    echo "GPT-Narratives/${name}: done"
  else
    echo "GPT-Narratives/${name}: SKIP (no cache at ${emb_src})"
  fi
done

# --- 3. MaScQA questions + scoresheet ---
mkdir -p "${DST_TMP}/MaScQA/scoresheets"
rsync -a --update "${POOL}/MaScQA/mascqa-eval.json" "${DST_TMP}/MaScQA/"
rsync -a --update "${POOL}/MaScQA/scoresheets/all_questions.xlsx" \
                  "${DST_TMP}/MaScQA/scoresheets/"
echo "MaScQA: done"

# --- 4. arXiv + CAMEL-AI: preprocess into lean formats (idempotent by mtime) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/preprocess_text_datasets.py" \
    --arxiv_json  "${POOL}/jarvis/arXivdataset.json" \
    --arxiv_out   "${DST_TMP}/jarvis_arxiv.parquet" \
    --camel_root  "${POOL}/camel-ai" \
    --camel_out   "${DST_TMP}/camel_ai.jsonl"
echo "arxiv + camel: done"
