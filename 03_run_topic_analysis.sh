#!/bin/bash -l

#SBATCH --job-name=mira_topic_analysis
#SBATCH --output=LOGS/mira_topic_analysis.log
#SBATCH --error=LOGS/mira_topic_analysis.err
#SBATCH -p compute
#SBATCH -c 32
#SBATCH --mem=128G

set -euo pipefail

source activate mira-env

cd "/gpfs/Home/esm5360/MIRA/"

BASE_DIR="/gpfs/Home/esm5360/MIRA/"
DATASET_NAME="ds011_full"

echo "Running MIRA Topic Analysis"
python3 src/03_mira_topic_analysis.py \
    --base_dir "$BASE_DIR" \
    --dataset_name "$DATASET_NAME"

echo "DONE"