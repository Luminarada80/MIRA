import os

import anndata  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scanpy as sc  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix
from typing import Union
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

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

def atac_data_preprocessing(
    atac_adata: anndata.AnnData, 
    barcodes: list[str],
    filter_gene_min_cells: int = 30, 
    min_genes_per_cell: int = 1000,
    fig_dir: str = 'figures',
    plot_genes_by_counts: bool = True,
    h5ad_save_path: Union[None, str] = None
    ) -> anndata.AnnData:
    """
    QC filtering and preprocessing of an ATACseq AnnData object.

    Args:
        atac_adata (anndata.AnnData): 
            Unprocessed ATACseq AnnData object.
        barcodes (list[str]):
            A list of paired barcodes from the RNAseq dataset.
        filter_gene_min_cells (int, optional): 
            A gene must be be expressed in greater than this number of cells. Defaults to 30.
        min_genes_per_cell (int, optional): 
            A cell must be expressing more than this number of genes. Defaults to 1000.
        fig_dir (str, optional): 
            Figure for saving the `accessibility_genes_by_counts.png` figure. Defaults to 'figures'.
        plot_genes_by_counts (bool, optional): 
            True to plot the figure, False to skip plotting. Defaults to True.
        h5ad_save_path (None | str): 
            Path to save the processed ATAC AnnData object as an h5 file.

    Returns:
        atac_adata (anndata.AnnData): Filtered ATAC AnnData object
    """
    
    logging.info("    (1/4) Filtering out very rare peaks")
    sc.pp.filter_genes(atac_adata, min_cells = filter_gene_min_cells)

    atac_adata = atac_adata[barcodes]
    
    logging.info("    (2/4) Calculating QC metrics")
    sc.pp.calculate_qc_metrics(atac_adata, inplace=True, log1p=False)
    
    if plot_genes_by_counts:
        logging.info("      - Plotting genes by counts vs total counts")
        ax: plt.Axes = sc.pl.scatter(atac_adata,
                    x = 'n_genes_by_counts',
                    y = 'total_counts',
                    show = False,
                    size = 2,
                    )

        ax.vlines(1000, 100, 1e5)
        ax.set(xscale = 'log', yscale = 'log')
        
        fig = ax.get_figure()
        
        qc_fig_path = os.path.join(fig_dir, "QC_figs")
        os.makedirs(qc_fig_path, exist_ok=True)
        
        if isinstance(fig, plt.Figure):
            fig.savefig(
                os.path.join(qc_fig_path, "accessibility_genes_by_counts.png"),
                dpi=200,
                bbox_inches="tight"
            )

    logging.info(f"    (3/4) Filtering cells by {min_genes_per_cell} min genes per cell")
    sc.pp.filter_cells(atac_adata, min_genes=min_genes_per_cell)

    logging.info(f"    (4/4) Subsampling to 1e5 peaks per cell")
    # If needed, reduce the size of the dataset by subsampling
    np.random.seed(0)
    atac_adata.var['endogenous_peaks'] = np.random.rand(atac_adata.shape[1]) <= min(1e5/atac_adata.shape[1], 1)
    
    if h5ad_save_path:
        logging.info(f"    Writing h5ad file to {os.path.basename(h5ad_save_path)}")
        atac_adata.write_h5ad(h5ad_save_path)
    
    return atac_adata

def rna_data_preprocessing(
    rna_adata: anndata.AnnData, 
    min_cells_per_gene: int = 15,
    target_read_depth: float = 1e4, 
    min_gene_disp: float = 0.5,
    h5ad_save_path: Union[None, str] = None
    ) -> anndata.AnnData:
    """
    QC filtering and preprocessing of an RNAseq AnnData object.

    Args:
        rna_adata (anndata.AnnData): 
            Unprocessed RNAseq AnnData object.
        min_cells_per_gene (int):
            Genes must be expressed in at least this number of cells. Defaults to 15
        target_read_depth (float, optional): 
            Normalizes the read depth of each cell. Defaults to 1e4.
        min_gene_disp (float, optional): 
            Minimum gene variability by dispersion. Defaults to 0.5.
        h5ad_save_path (None | str): 
            Path to save the processed RNA AnnData object as an h5 file.

    Returns:
        rna_adata (anndata.AnnData): Filtered RNA AnnData object
    """
    
    logging.info("    (1/4) Filtering out very rare genes")
    sc.pp.filter_genes(rna_adata, min_cells=min_cells_per_gene)
    rawdata = rna_adata.X.copy()

    logging.info(f"    (2/4) Normalizing to a read depth of {target_read_depth}")
    sc.pp.normalize_total(rna_adata, target_sum=target_read_depth)

    logging.info("    (3/4) Logarithmizing the data")
    sc.pp.log1p(rna_adata)

    logging.info(f"    (4/4) Filtering for highly variable genes with dispersion > {min_gene_disp}")
    sc.pp.highly_variable_genes(rna_adata, min_disp = min_gene_disp)

    rna_adata.layers['counts'] = rawdata

    if h5ad_save_path:
        logging.info(f"    Writing h5ad file to {os.path.basename(h5ad_save_path)}")
        rna_adata.write_h5ad(h5ad_save_path)
    
    return rna_adata

def load_and_process_rna_data(rna_data_path, rna_h5ad_save_path):
    
    if not os.path.isfile(rna_h5ad_save_path):
        logging.info("  - Reading RNAseq raw data parquet file")
        rna_data = pd.read_parquet(rna_data_path, engine="pyarrow")

        logging.info("  - Converting DataFrame to AnnData object")
        rna_adata = anndata_from_dataframe(rna_data, "gene_id")

        logging.info("  - Running RNA preprocessing")
        rna_adata = rna_data_preprocessing(
            rna_adata=rna_adata,
            min_cells_per_gene=15,
            target_read_depth=1e4,
            min_gene_disp=0.5,
            h5ad_save_path=rna_h5ad_save_path
        )
        
        return rna_adata
        
    else:
        logging.info("  - RNA h5ad file found, loading")
        return anndata.read_h5ad(rna_h5ad_save_path)

def load_and_process_atac_data(atac_data_path, atac_h5ad_save_path, barcodes, fig_dir):
    
    if not os.path.isfile(atac_h5ad_save_path):
        logging.info("  - Reading ATACseq raw data parquet file")
        atac_data = pd.read_parquet(atac_data_path, engine="pyarrow")

        logging.info("  - Converting DataFrame to AnnData object")
        atac_adata = anndata_from_dataframe(atac_data, "peak_id")

        logging.info("  - Running ATAC preprocessing")
        atac_adata = atac_data_preprocessing(
            atac_adata,
            barcodes,
            filter_gene_min_cells=30,
            min_genes_per_cell=1000,
            fig_dir=fig_dir,
            plot_genes_by_counts=True,
            h5ad_save_path=atac_h5ad_save_path
        )
        return atac_adata
        
    else:
        return anndata.read_h5ad(atac_h5ad_save_path)
