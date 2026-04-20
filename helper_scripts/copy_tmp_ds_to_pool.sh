#!/bin/bash

set -e

for ds in cantor_hea gnome hmof jarvis_dft jarvis_qetb mp omdb oqmd qmof snumat; do
  # echo "Copying train.db to /home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.db"
  # cp "/tmp/LLM4Mat-Bench/${ds}/train.db" "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.db" 
  # echo "Copying validation.db to /home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.db"
  # cp "/tmp/LLM4Mat-Bench/${ds}/validation.db" "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.db"
  echo "Copying train.id_index.json to /home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.id_index.json"
  cp "/tmp/LLM4Mat-Bench/${ds}/train.id_index.json" "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/train.id_index.json" 
  echo "Copying validation.id_index.json to /home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.id_index.json"
  cp "/tmp/LLM4Mat-Bench/${ds}/validation.id_index.json" "/home/sathyae/orcd/pool/LLM4Mat-Bench/${ds}/validation.id_index.json"
done
