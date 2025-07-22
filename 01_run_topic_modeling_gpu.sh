#!/bin/bash -l

#SBATCH --job-name=mira_topic_training
#SBATCH --output=LOGS/mira_topic_training.log
#SBATCH --error=LOGS/mira_topic_training.err
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

srun nvidia-smi

source activate mira-env

cd "/gpfs/Home/esm5360/MIRA/"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

BASE_DIR="/gpfs/Home/esm5360/MIRA/"
DATASET_NAME="ds011_full"
INPUT_DATA_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/"
ATAC_DATA_FILENAME="DS011_mESC_ATAC.parquet"
RNA_DATA_FILENAME="DS011_mESC_RNA.parquet"


echo "Running analysis of MIRA embeddings"
python3 src/01_mira_topic_modeling_mesc.py \
    --base_dir "$BASE_DIR" \
    --dataset_name "$DATASET_NAME" \
    --input_data_dir "$INPUT_DATA_DIR" \
    --atac_data_filename "$ATAC_DATA_FILENAME" \
    --rna_data_filename "$RNA_DATA_FILENAME"

echo "DONE"