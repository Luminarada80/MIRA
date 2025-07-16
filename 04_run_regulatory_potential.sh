#!/bin/bash -l

#SBATCH --job-name=mira_regulatory_potential
#SBATCH --output=LOGS/mira_regulatory_potential.log
#SBATCH --error=LOGS/mira_regulatory_potential.err
#SBATCH -p compute
#SBATCH -c 32
#SBATCH --mem=128G

set -euo pipefail

source activate mira-env

cd "/gpfs/Home/esm5360/MIRA/"

echo "Running MIRA LITE and NITE RP Modeling"
python3 src/04_mira_regulatory_potential.py

echo "DONE"