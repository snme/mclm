#!/bin/bash
#SBATCH --job-name=train
#SBATCH -o logs/train%j.o         # Job name
#SBATCH -e logs/train%j.e         # Job name
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --partition=pg_tata
#SBATCH --gres=gpu:h200:8
#SBATCH --mail-type=all                 #
#SBATCH --mail-user=sathyae@mit.edu     # send emails


# Initialize and Load Modules
source /etc/profile # lets you run "module load"
module purge 
# nah module load cuda/11.3
ml miniforge 
ml cuda/12

source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda deactivate # messes up path if you don't do this
conda deactivate # messes up path if you don't do this
conda deactivate # messes up path if you don't do this
conda activate llm # whatever env you want

# Set some environment variables needed by torch.distributed 
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507
# Get unused port
# export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')r

# set number of threads, same as cpus-per-task
export OMP_NUM_THREADS=64
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=hsn0

# export DDP vars
export RANK=$SLURM_PROCID
export WORLD_RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

# env

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

# Full-dataset training on all 10 LLM4Mat-Bench datasets using cached OrbV3 features.
# /tmp/LLM4Mat-Bench needs the per-dataset CSVs (descriptions); DBs are unused in cached mode.
# /tmp/cached_embs must already contain the .flat.bin / .flat.idx.json pairs
# (produced once by helper_scripts/flatten_cached_embs.py).
rsync -a --include='*/' --include='*.csv' --exclude='*' \
    $DATA_HOME/LLM4Mat-Bench/ /tmp/LLM4Mat-Bench/

srun -u torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --data_parent_path /tmp/LLM4Mat-Bench \
    --cached_embs_parent_path /tmp/cached_embs \
    --model_save_path /home/sathyae/mclm/alm/checkpoint.pt




# torchrun \
#     --nnodes=1 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$(hostname):29507 \
#     train.py \
#     --train_csv_path /tmp/train.csv \
#     --val_csv_path /tmp/validation.csv \
#     --db_path /tmp/oqmd.db \
#     --model_save_path /home/sathyae/mclm/alm/checkpoint.pt

# torchrun \
#     --nnodes=1 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$(hostname):29507 \
#     train.py \
#     --train_csv_path /tmp/train.csv \
#     --val_csv_path /tmp/validation.csv \
#     --db_path /tmp/oqmd.db \
#     --model_save_path /home/sathyae/mclm/alm/checkpoint.pt