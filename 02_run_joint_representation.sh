#!/bin/bash -l

#SBATCH --job-name=mira_joint_representation
#SBATCH --output=LOGS/mira_joint_representation.log
#SBATCH --error=LOGS/mira_joint_representation.err
#SBATCH -p compute
#SBATCH -c 32
#SBATCH --mem=128G

set -euo pipefail

source activate mira-env

cd "/gpfs/Home/esm5360/MIRA/"

BASE_DIR="/gpfs/Home/esm5360/MIRA/"
DATASET_NAME="ds011_full"

echo "Running MIRA Joint Representation"
python3 src/02_mira_joint_representation.py \
    --base_dir "$BASE_DIR" \
    --dataset_name "$DATASET_NAME"

echo "DONE"