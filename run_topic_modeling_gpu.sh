#!/bin/bash -l

#SBATCH --job-name=mira_topic_training
#SBATCH --output=LOGS/mira_topic_training.log
#SBATCH --error=LOGS/mira_topic_training.err
#SBATCH -p gpu
#SBATCH -c 5
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

srun nvidia-smi

source activate mira-env

cd "/gpfs/Home/esm5360/MIRA/"

echo "Running analysis of MIRA embeddings"
python3 src/mira_topic_modeling.py

echo "DONE"