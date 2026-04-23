#!/bin/bash

set -e

SRC_DS="/home/sathyae/orcd/pool/LLM4Mat-Bench"
DST_DS="/tmp/LLM4Mat-Bench"
SRC_EMB="/home/sathyae/orcd/pool/cached_embs"
DST_EMB="/tmp/cached_embs"
MODEL="orb_v3_direct_20_omat"

for ds in cantor_hea gnome hmof jarvis_dft jarvis_qetb mp omdb oqmd qmof snumat; do
  echo "=== ${ds} ==="
  mkdir -p "${DST_DS}/${ds}"
  mkdir -p "${DST_EMB}/${ds}/embeddings"
  for split in train validation; do
    for ext in csv db id_index.json; do
      src="${SRC_DS}/${ds}/${split}.${ext}"
      dst="${DST_DS}/${ds}/${split}.${ext}"
      echo "  ${src} -> ${dst}"
      cp "${src}" "${dst}"
    done
    # Only the flat mmap pair is needed at training time; the .pt dict and _keys.txt
    # are cache-build intermediates.
    for ext in flat.bin flat.idx.json; do
      src="${SRC_EMB}/${ds}/embeddings/${MODEL}_${split}_atom.${ext}"
      dst="${DST_EMB}/${ds}/embeddings/${MODEL}_${split}_atom.${ext}"
      echo "  ${src} -> ${dst}"
      cp "${src}" "${dst}"
    done
  done
done
