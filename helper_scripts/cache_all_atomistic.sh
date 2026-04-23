#!/bin/bash

set -e

DATA_ROOT="/home/sathyae/orcd/pool/LLM4Mat-Bench"
MODEL_NAME="orb_v3_direct_20_omat"
BATCH_SIZE=10

for ds in cantor_hea gnome hmof jarvis_dft jarvis_qetb mp omdb oqmd qmof snumat; do
  for split in train validation; do
    db_path="${DATA_ROOT}/${ds}/${split}.db"
    if [ ! -f "${db_path}" ]; then
      echo "Skipping missing ${db_path}"
      continue
    fi
    echo "=== Caching ${ds}/${split} ==="
    python helper_scripts/cache_embeddings_atomistic_orbv3.py \
      --model_name "${MODEL_NAME}" \
      --data_path "${db_path}" \
      --dataset_name "${ds}" \
      --batch_size "${BATCH_SIZE}" \
      --postfix "_${split}_atom" \
      --save_atom_embeddings
  done
done
