import os
from typing import Union

import anndata
import matplotlib.pyplot as plt
import mira  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from utils.data_processing import (
    load_and_process_atac_data,
    load_and_process_rna_data,
    )

from utils.topic_models import (
    load_or_create_mira_accessibility_topic_model,
    load_or_create_mira_expression_topic_model,
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
    
def create_atac_topic_model(atac_data_path, atac_h5ad_save_path, barcodes):
    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model.pth")

    atac_adata = load_and_process_atac_data(atac_data_path, atac_h5ad_save_path, barcodes, FIG_DIR)
    
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

    trained_atac_model = create_and_fit_bayesian_tuner_to_data(
        model=model,
        adata=os.path.join(training_cache, 'atac_train'),
        num_sig_topics=num_sig_topics,
        tuner_save_name=tuner_save_dir,
        model_save_path=model_save_path,
        fig_dir=FIG_DIR,
        plot_loss=True,
        plot_pareto=True
    )
    
    return atac_adata, trained_atac_model
    
def create_rna_topic_model(rna_data_path, rna_h5ad_save_path):
    
    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model.pth")
    tuner_save_dir = os.path.join(BASE_DIR, f"{DATASET_NAME}_rna/0")
    
    rna_adata = load_and_process_rna_data(rna_data_path, rna_h5ad_save_path)
    
    barcodes = rna_adata.obs_names.to_list()
    
    rna_expr_model = load_or_create_mira_expression_topic_model(rna_adata, model_save_path)
    rna_expr_model, num_sig_topics = set_model_learning_parameters(rna_expr_model, rna_adata)

    trained_rna_model = create_and_fit_bayesian_tuner_to_data(
        rna_expr_model, 
        rna_adata, 
        num_sig_topics, 
        tuner_save_name=tuner_save_dir,
        model_save_path=model_save_path,
        fig_dir=FIG_DIR,
        plot_loss=True,
        plot_pareto=True
        )
    
    return rna_adata, trained_rna_model, barcodes

rna_adata, trained_rna_model, barcodes = create_rna_topic_model(rna_data_path, rna_h5ad_save_path)
atac_adata, trained_atac_model = create_atac_topic_model(atac_data_path, atac_h5ad_save_path, barcodes)