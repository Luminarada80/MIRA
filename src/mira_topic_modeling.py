import os

import matplotlib.pyplot as plt
import torch

from utils.data_processing import ( # type: ignore[import-not-found]
    load_and_process_atac_data,
    load_and_process_rna_data,
    )

from utils.topic_models import ( # type: ignore[import-not-found]
    load_or_create_mira_accessibility_topic_model,
    load_or_create_mira_expression_topic_model,
    set_model_learning_parameters,
    create_and_fit_bayesian_tuner_to_data
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

plt.rcParams.update({'font.size': 14})

BASE_DIR = "/gpfs/Home/esm5360/MIRA/"
FIG_DIR = os.path.join(BASE_DIR, "figures")
TUNER_DIR = os.path.join(BASE_DIR, "tuners")
DATASET_DIR = os.path.join(BASE_DIR, "mira-datasets")
DATASET_NAME = "ds011"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

input_data_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/"

atac_data_path = os.path.join(input_data_dir, "DS011_mESC_ATAC_processed.parquet")
rna_data_path = os.path.join(input_data_dir, "DS011_mESC_RNA_processed.parquet")

    
def create_atac_topic_model(atac_data_path, barcodes):
    
    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model.pth")
    tuner_save_dir = os.path.join(TUNER_DIR, f"{DATASET_NAME}_atac")
    atac_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data.h5ad")
    training_cache = os.path.join(DATASET_DIR, "ds011_training")

    logging.info("\nLoading and processing the scATAC-seq data")
    atac_adata = load_and_process_atac_data(atac_data_path, atac_h5ad_save_path, barcodes, FIG_DIR)
    
    logging.info("Loading or creating the MIRA ATAC expression topic model")
    model = load_or_create_mira_accessibility_topic_model(atac_adata, model_save_path)
    
    os.makedirs(training_cache, exist_ok=True)

    train, test = model.train_test_split(atac_adata)

    logging.info("Writing the train / test splits to the training data cache")
    if not 'atac_train' in os.listdir(training_cache):
        model.write_ondisk_dataset(train, dirname=os.path.join(training_cache, 'atac_train'))
        
    if not 'atac_test' in os.listdir(training_cache):
        model.write_ondisk_dataset(test, dirname=os.path.join(training_cache, 'atac_test'))

    logging.info("Setting the topic model learning parameters")
    model, num_sig_topics = set_model_learning_parameters(
        model=model,
        adata=os.path.join(training_cache, 'atac_train'),
        fig_dir=FIG_DIR
    )
    
    logging.info("Creating and fitting the Bayesian tuner to the scATAC-seq expression data")
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
    logging.info("Done!\n")
    
    return atac_adata, trained_atac_model
    
def create_rna_topic_model(rna_data_path):
    
    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model.pth")
    tuner_save_dir = os.path.join(TUNER_DIR, f"{DATASET_NAME}_rna")
    rna_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data.h5ad")
    
    logging.info("\nLoading and processing the scRNA-seq data")
    rna_adata = load_and_process_rna_data(rna_data_path, rna_h5ad_save_path)
    
    barcodes = rna_adata.obs_names.to_list()
    
    logging.info("Loading or creating the MIRA RNA expression topic model")
    rna_expr_model = load_or_create_mira_expression_topic_model(rna_adata, model_save_path)
    
    logging.info("Setting the topic model learning parameters")
    rna_expr_model, num_sig_topics = set_model_learning_parameters(rna_expr_model, rna_adata)

    logging.info("Creating and fitting the Bayesian tuner to the scRNA-seq expression data")
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
    logging.info("Done!\n")
    
    return rna_adata, trained_rna_model, barcodes

assert torch.cuda.is_available()

rna_adata, trained_rna_model, barcodes = create_rna_topic_model(rna_data_path)
atac_adata, trained_atac_model = create_atac_topic_model(atac_data_path, barcodes)