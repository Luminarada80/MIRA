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

echo "Running MIRA Topic Analysis"
python3 src/03_mira_topic_analysis.py

echo "DONE"