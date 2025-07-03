import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from utils.data_processing import ( # type: ignore[import-not-found]
    filter_atac_by_distance_to_tss,
    atac_data_preprocessing,
    rna_data_preprocessing,
    convert_anndata_to_pandas,
    write_processed_dataframe_to_parquet
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
DATASET_NAME = "ds011_full"
NUM_CPU = 2

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

input_data_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/"

atac_data_path = os.path.join(input_data_dir, "DS011_mESC_ATAC.parquet")
rna_data_path = os.path.join(input_data_dir, "DS011_mESC_RNA.parquet")

atac_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_full.h5ad")
rna_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_full.h5ad")

def create_atac_topic_model(atac_adata, bayesian_tuner = True):
    
    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model.pth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tuner_save_dir = os.path.join(TUNER_DIR, f"{DATASET_NAME}_atac_{timestamp}")
    
    training_cache = os.path.join(DATASET_DIR, f"{DATASET_NAME}_training")
    
    # MIRA requires raw counts, which were stored in a layer of the AnnData object.
    # We need to set the expression matrix back to the counts matrix for MIRA
    atac_adata.X = atac_adata.layers["counts"]
    
    logging.info("Loading or creating the MIRA ATAC expression topic model")
    model = load_or_create_mira_accessibility_topic_model(atac_adata, model_save_path)
    
    os.makedirs(training_cache, exist_ok=True)

    train, test = model.train_test_split(atac_adata)
    
    logging.info("Writing the train / test splits to the training data cache")
    train_dir = os.path.join(training_cache, 'atac_train')
    test_dir = os.path.join(training_cache, 'atac_test')

    if not os.path.exists(train_dir):
        model.write_ondisk_dataset(train, dirname=train_dir)
    if not os.path.exists(test_dir):
        model.write_ondisk_dataset(test, dirname=test_dir)

    logging.info("Setting the topic model learning parameters")
    model, num_topics = set_model_learning_parameters(
        model=model,
        adata=os.path.join(training_cache, 'atac_train'),
        fig_dir=FIG_DIR
    )
    
    # min_lr = 1e-5
    # max_lr = 5e-4
    # model.set_learning_rates(min_lr, max_lr)
    # num_topics = 2
    
    # Skipping the Bayesian tuner is faster, but less optimal
    if bayesian_tuner == True:
        train_path = os.path.join(training_cache, 'atac_train')
        test_path  = os.path.join(training_cache, 'atac_test')
        
        logging.info("Creating and fitting the Bayesian tuner to the scATAC-seq expression data")
        trained_atac_model = create_and_fit_bayesian_tuner_to_data(
            model,
            (train_path, test_path),
            num_topics,
            n_jobs=NUM_CPU,
            tuner_save_name=tuner_save_dir,
            model_save_path=model_save_path,
            fig_dir=FIG_DIR,
            plot_loss=True,
            plot_pareto=True
        )
    else:
        logging.info(f"Skipping Bayesian tuner, training the accessibility model using {num_topics} topics")
        model.set_params(num_topics=num_topics, batch_size=256)
        trained_atac_model = model.fit(atac_adata)
    
        trained_atac_model.save(model_save_path)
        
    logging.info("Done!\n")
    
    return atac_adata, trained_atac_model
    
def create_rna_topic_model(rna_adata, bayesian_tuner = True):
    
    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model.pth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tuner_save_dir = os.path.join(TUNER_DIR, f"{DATASET_NAME}_rna_{timestamp}")
    
    logging.info("Loading or creating the MIRA RNA expression topic model")
    rna_expr_model = load_or_create_mira_expression_topic_model(rna_adata, model_save_path)
    
    logging.info("Setting the topic model learning parameters")
    rna_expr_model, num_topics = set_model_learning_parameters(rna_expr_model, rna_adata)
    
    # min_lr = 1e-5
    # max_lr = 5e-4
    # rna_expr_model.set_learning_rates(min_lr, max_lr)
    # num_topics = 5
    
    # Skipping the Bayesian tuner is faster, but less optimal
    if bayesian_tuner == True:
        logging.info("Creating and fitting the Bayesian tuner to the scRNA-seq expression data")
        trained_rna_model = create_and_fit_bayesian_tuner_to_data(
            rna_expr_model, 
            rna_adata, 
            num_topics, 
            n_jobs=NUM_CPU,
            tuner_save_name=tuner_save_dir,
            model_save_path=model_save_path,
            fig_dir=FIG_DIR,
            plot_loss=True,
            plot_pareto=True
            )
    else:
        logging.info(f"Skipping Bayesian tuner, training the expression model using {num_topics} topics")
        rna_expr_model = rna_expr_model.set_params(num_topics=num_topics, batch_size=256)
        trained_rna_model = rna_expr_model.fit(rna_adata)
        
        trained_rna_model.save(model_save_path)
    logging.info("Done!\n")
    
    return rna_adata, trained_rna_model

assert torch.cuda.is_available()

logging.info("\nLoading and processing the scRNA-seq data")
rna_adata_processed = rna_data_preprocessing(
    rna_data_path, 
    rna_h5ad_save_path,
    min_cells_per_gene = 15,
    target_read_depth = 1e6, 
    min_gene_disp = 0.5,
    min_genes = 200,
    max_genes = 2500,
    max_pct_mt = 5.0,
    overwrite=False
)

rna_df = convert_anndata_to_pandas(rna_adata_processed, "gene_id")
write_processed_dataframe_to_parquet(rna_df, rna_data_path)

barcodes = rna_adata_processed.obs_names.to_list()
gene_names = rna_adata_processed.var_names.to_list()

logging.info("\nRunning ATAC preprocessing")
atac_adata_processed = atac_data_preprocessing(
    atac_data_path,
    barcodes,
    gene_names,
    filter_peak_min_cells=30,
    min_peaks_per_cell=1000,
    target_read_depth=1e6,
    tss_distance_cutoff=1e6,
    fig_dir=FIG_DIR,
    dataset_dir=DATASET_DIR,
    plot_peaks_by_counts=True,
    h5ad_save_path=atac_h5ad_save_path,
    overwrite=True
)
logging.info(f"  - Pre-processed ATAC shape: {atac_adata_processed.shape}")

atac_df = convert_anndata_to_pandas(atac_adata_processed, "peak_id")
write_processed_dataframe_to_parquet(atac_df, atac_data_path)

logging.info(f"  - ATAC filtered by TSS shape: {atac_adata_processed.shape}")

logging.info("\n----- Creating RNA Topic Model -----")
# rna_adata, trained_rna_model = create_rna_topic_model(rna_adata_processed, bayesian_tuner=True)

logging.info("\n----- Creating ATAC Topic Model -----")
atac_adata, trained_atac_model = create_atac_topic_model(atac_adata_processed, bayesian_tuner=True)

# trained_atac_model.get_enriched_TFs(atac_adata, topic_num=17, top_quantile=0.1)
