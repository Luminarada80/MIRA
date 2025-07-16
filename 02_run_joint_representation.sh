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

echo "Running MIRA Joint Representation"
python3 src/02_mira_joint_representation.py

echo "DONE"