#!/bin/bash

set -e

for ds in cantor_hea gnome hmof jarvis_dft jarvis_qetb mp omdb oqmd qmof snumat; do
  echo "Copying ${ds} to /tmp/LLM4Mat-Bench/${ds}"
  mkdir -p "/tmp/LLM4Mat-Bench/${ds}"
  echo "Copying train.csv to /tmp/LLM4Mat-Bench/${ds}/train.csv"
  cp "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.csv" "/tmp/LLM4Mat-Bench/${ds}/train.csv"
  echo "Copying train.db to /tmp/LLM4Mat-Bench/${ds}/train.db"
  cp "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.db" "/tmp/LLM4Mat-Bench/${ds}/train.db"
  echo "Copying train.id_index.json to /tmp/LLM4Mat-Bench/${ds}/train.id_index.json"
  cp "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.id_index.json" "/tmp/LLM4Mat-Bench/${ds}/train.id_index.json"
  echo "Copying validation.csv to /tmp/LLM4Mat-Bench/${ds}/validation.csv"
  cp "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.csv" "/tmp/LLM4Mat-Bench/${ds}/validation.csv"
  echo "Copying validation.db to /tmp/LLM4Mat-Bench/${ds}/validation.db"
  cp "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.db" "/tmp/LLM4Mat-Bench/${ds}/validation.db"
  echo "Copying validation.id_index.json to /tmp/LLM4Mat-Bench/${ds}/validation.id_index.json"
  cp "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.id_index.json" "/tmp/LLM4Mat-Bench/${ds}/validation.id_index.json"
done
