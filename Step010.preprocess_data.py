import mira

import pandas as pd
import anndata
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

plt.rcParams.update({'font.size': 14})

def create_batch_effect_umap(data):
    sc.settings.figdir = "figures/"
    sc.tl.pca(data)
    sc.pp.neighbors(data, n_pcs=6)
    sc.tl.umap(data, min_dist = 0.2, negative_sample_rate=0.2)
    sc.pl.umap(
        data, 
        color = 'batch', 
        palette= ['#8f7eadff', '#c1e1e2ff'], 
        frameon=False, 
        save='tutorial_data_batch_umamp.png'
        )

# print("Installing CODAL synthetic single-cell dataset")
# mira.datasets.CodalFrankencellTutorial()
# print("\tDone!")

# print("Reading KO h5ad")
# ko = anndata.read_h5ad('mira-datasets/CODAL_tutorial/perturbation.h5ad')
# print("\tDone!")

# print("Reading WT h5ad")
# wt = anndata.read_h5ad('mira-datasets/CODAL_tutorial/wild-type.h5ad')
# print("\tDone!")

# print("Combinding KO and WT data")
# data = anndata.concat({'ko' : ko, 'wt' : wt}, label='batch', index_unique=':') # combine into one dataframe
# print("\tDone!")

atac_data = pd.read_parquet("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/DS011_mESC_ATAC_processed.parquet", engine="pyarrow")
rna_data = pd.read_parquet("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/DS011_mESC_RNA_processed.parquet", engine="pyarrow")

def anndata_from_dataframe(df, id_col_name):
    # 1) Validate input
    if id_col_name not in df.columns:
        raise ValueError(f"Identifier column '{id_col_name}' not found in DataFrame.")

    # Separate gene IDs vs. raw count matrix
    gene_or_peak_ids = df[id_col_name].astype(str).tolist()
    counts_df = df.drop(columns=[id_col_name]).copy()

    # Ensure all other columns are numeric; coerce non‐numeric to NaN→0
    counts_df = counts_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Extract cell IDs from the DataFrame columns
    cell_ids = counts_df.columns.astype(str).tolist()

    # 2) Build AnnData with shape (cells × genes)
    #    We must transpose counts so rows=cells, columns=genes
    counts_matrix = csr_matrix(counts_df.values)           # shape: (n_genes, n_cells)
    counts_matrix = counts_matrix.T                         # now (n_cells, n_genes)

    adata = anndata.AnnData(X=counts_matrix)
    adata.obs_names = cell_ids       # each row = one cell
    adata.var_names = gene_or_peak_ids       # each column = one gene or peak
    
    return adata

atac_adata = anndata_from_dataframe(atac_data, "peak_id")
rna_adata = anndata_from_dataframe(rna_data, "gene_id")

import torch
assert torch.cuda.is_available()

print("Filtering out very rare genes")
sc.pp.filter_genes(rna_adata, min_cells=15)
rawdata = rna_adata.X.copy()

print("Normalizing the read depths of each cell")
sc.pp.normalize_total(rna_adata, target_sum=1e4)

print("Logarithmizing the data")
sc.pp.log1p(rna_adata)

print("Calculating highly variable genes for statistical model")
sc.pp.highly_variable_genes(rna_adata, min_disp = 0.5)

rna_adata.layers['counts'] = rawdata

# create_batch_effect_umap(data)

rna_adata.write_h5ad("mira-datasets/rna_adata.h5ad")