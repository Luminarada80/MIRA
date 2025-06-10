#!/bin/bash -l

#SBATCH --job-name=mira_topic_training
#SBATCH --output=LOGS/mira_topic_training.log
#SBATCH --error=LOGS/mira_topic_training.err
#SBATCH -p gpu
#SBATCH -c 5
#SBATCH --mem=128
#SBATCH --gres=gpu:1

set -euo pipefail

# srun nvidia-smi

source activate mira-env

cd "/gpfs/Home/esm5360/MIRA/"

echo "Running data preprocessing"
python3 Step010.preprocess_data.py

echo "Running mira_topic_modeling.py"
python3 Step020.mira_topic_modeling.py

echo "Running analysis of MIRA embeddings"
python3 Step030.mira_prediction_analysis.py

echo "DONE"