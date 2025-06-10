import mira # type: ignore[import-untyped]

import os
import pandas as pd # type: ignore[import-untyped]

import scanpy as sc # type: ignore[import-untyped]
import numpy as np
import matplotlib.pyplot as plt

from utils.data_processing import anndata_from_dataframe, rna_data_preprocessing

plt.rcParams.update({'font.size': 14})

input_data_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/""

rna_data = pd.read_parquet(os.path.join(input_data_dir, "DS011_mESC_RNA_processed.parquet"), engine="pyarrow")

rna_adata = anndata_from_dataframe(rna_data, "gene_id")

rna_adata = rna_data_preprocessing(
    rna_adata=rna_adata,
    min_cells_per_gene=15,
    target_read_depth=1e4,
    min_gene_disp=0.5,
    h5ad_save_path="mira-datasets/rna_data.h5ad"
)

