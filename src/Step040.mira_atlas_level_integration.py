import os
from typing import Union

import anndata
import matplotlib.pyplot as plt
import mira  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from utils.data_processing import (
    anndata_from_dataframe,
    atac_data_preprocessing,
    rna_data_preprocessing
    )

from utils.topic_models import (
    load_or_create_mira_accessibility_topic_model,
    set_model_learning_parameters,
    create_and_fit_bayesian_tuner_to_data
)

plt.rcParams.update({'font.size': 14})

BASE_DIR = "/gpfs/Home/esm5360/MIRA/"
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATASET_DIR = os.path.join(BASE_DIR, "mira-datasets")
DATASET_NAME = "ds011"



input_data_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/"

atac_data_path = os.path.join(input_data_dir, "DS011_mESC_ATAC.parquet")
atac_h5ad_save_path = os.path.join(DATASET_DIR, "atac_data.h5ad")

rna_data_path = os.path.join(input_data_dir, "DS011_mESC_RNA.parquet")
rna_h5ad_save_path = os.path.join(DATASET_DIR, "rna_data.h5ad")

def load_and_process_rna_data(rna_data_path, rna_h5ad_save_path):
    
    if not os.path.isfile(rna_h5ad_save_path):
        print("  - Reading RNAseq raw data parquet file")
        rna_data = pd.read_parquet(rna_data_path, engine="pyarrow")

        print("  - Converting DataFrame to AnnData object")
        rna_adata = anndata_from_dataframe(rna_data, "gene_id")

        print("  - Running RNA preprocessing")
        rna_adata = rna_data_preprocessing(
            rna_adata=rna_adata,
            min_cells_per_gene=15,
            target_read_depth=1e4,
            min_gene_disp=0.5,
            h5ad_save_path=rna_h5ad_save_path
        )
        
        return rna_adata
        
    else:
        return anndata.read_h5ad(rna_h5ad_save_path)
    
    
rna_adata = load_and_process_rna_data(rna_data_path, rna_h5ad_save_path)
barcodes = rna_adata.obs_names.to_list()

def load_and_process_atac_data(atac_data_path, atac_h5ad_save_path, barcodes):
    
    if not os.path.isfile(atac_h5ad_save_path):
        print("  - Reading ATACseq raw data parquet file")
        atac_data = pd.read_parquet(atac_data_path, engine="pyarrow")

        print("  - Converting DataFrame to AnnData object")
        atac_adata = anndata_from_dataframe(atac_data, "peak_id")

        print("  - Running ATAC preprocessing")
        atac_adata = atac_data_preprocessing(
            atac_adata,
            barcodes,
            filter_gene_min_cells=30,
            min_genes=1000,
            fig_dir=FIG_DIR,
            h5ad_save_path=os.path.join(DATASET_DIR, "atac_data.h5ad")
        )
        return atac_adata
        
    else:
        return anndata.read_h5ad(atac_h5ad_save_path)
    
atac_adata = load_and_process_atac_data(atac_data_path, atac_h5ad_save_path, barcodes)

model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model")

model = load_or_create_mira_accessibility_topic_model(atac_adata, model_save_path)

training_cache = os.path.join(DATASET_DIR, "ds011_training")
os.makedirs(training_cache, exist_ok=True)

train, test = model.train_test_split(atac_adata)

if not 'atac_train' in os.listdir(training_cache):
    model.write_ondisk_dataset(train, dirname=os.path.join(training_cache, 'atac_train'))
    
if not 'atac_test' in os.listdir(training_cache):
    model.write_ondisk_dataset(test, dirname=os.path.join(training_cache, 'atac_test'))


model, num_sig_topics = set_model_learning_parameters(
    model=model,
    adata=os.path.join(training_cache, 'atac_train'),
    fig_dir=FIG_DIR
)

tuner_save_dir = os.path.join(BASE_DIR, f"{DATASET_NAME}_atac/0")

model = create_and_fit_bayesian_tuner_to_data(
    model=model,
    adata=os.path.join(training_cache, 'atac_train'),
    num_sig_topics=num_sig_topics,
    tuner_save_name=tuner_save_dir,
    model_save_path=model_save_path,
    fig_dir=FIG_DIR,
    plot_loss=True,
    plot_pareto=True
)