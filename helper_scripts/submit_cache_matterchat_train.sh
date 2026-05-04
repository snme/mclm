#!/bin/bash
#SBATCH --job-name=cache_mc_train
#SBATCH -o logs/cache_mc_train%j.o
#SBATCH -e logs/cache_mc_train%j.e
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=pg_tata
#SBATCH --gres=gpu:h200:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sathyae@mit.edu

source /etc/profile
module purge
ml miniforge
ml cuda/12
conda activate llm

set -e
cd /home/sathyae/mclm

DB=/home/sathyae/orcd/pool/eval_data/Dataset_MatterChat/dataset/Material_data_postprocess1_out_correct_train.db
POOL=/home/sathyae/orcd/pool/cached_embs/matterchat_mp/embeddings

echo "=== Caching OrbV3 embeddings for MatterChat train (128k rows) ==="
python helper_scripts/cache_embeddings_atomistic_orbv3.py \
    --model_name orb_v3_direct_20_omat \
    --data_path "$DB" \
    --dataset_name matterchat_mp \
    --id_key material_id \
    --postfix _train_atom \
    --save_atom_embeddings \
    --batch_size 32

echo "=== Flattening to flat.bin ==="
python helper_scripts/flatten_cached_embs.py \
    --parent /home/sathyae/orcd/pool/cached_embs \
    --splits train \
    --model_name orb_v3_direct_20_omat

echo "Done."
