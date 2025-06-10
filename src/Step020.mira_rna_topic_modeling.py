import mira # type: ignore[import-untyped]

import os
import anndata # type: ignore[import-untyped]
import scanpy as sc # type: ignore[import-untyped]
import numpy as np
import matplotlib.pyplot as plt
from mira.topic_model.modality_mixins.accessibility_model import AccessibilityModel # type: ignore[import-untyped]
from mira.topic_model.modality_mixins.expression_model import ExpressionModel # type: ignore[import-untyped]
from typing import Union, Tuple

plt.rcParams.update({'font.size': 14})

from utils.topic_models import (
    load_or_create_mira_expression_topic_model,
    set_model_learning_parameters,
    create_and_fit_bayesian_tuner_to_data
)

BASE_DIR = "/gpfs/Homer/esm5360/MIRA/"
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATASET_DIR = os.path.join(BASE_DIR, "mira-datasets")
DATASET_NAME = "ds011"

def create_rna_topic_model():
    rna_adata = anndata.read_h5ad(os.path.join(DATASET_DIR, "rna_adata.h5ad"))

    model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model")

    rna_expr_model = load_or_create_mira_expression_topic_model(rna_adata, model_save_path)
    rna_expr_model, num_sig_topics = set_model_learning_parameters(rna_expr_model, rna_adata)

    tuner_save_dir = os.path.join(BASE_DIR, f"{DATASET_NAME}_rna/0")

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
    
    return trained_rna_model

