#!/bin/bash

#SBATCH --job-name=cache
#SBATCH -o cache%j.o         # Job name
#SBATCH -e cache%j.e         # Job name
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --time=48:00:00
#SBATCH --nodelist=node4301
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
ml cuda/13

source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda deactivate # messes up path if you don't do this
conda deactivate # messes up path if you don't do this
conda deactivate # messes up path if you don't do this
conda activate llm # whatever env you want

# run script
echo "Running script"
sleep 2d
wait
srun python cache_embeddings_llm.py --data /home/sathyae/orcd/pool/train.csv

