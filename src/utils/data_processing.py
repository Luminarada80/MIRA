import os

import anndata  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scanpy as sc  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix
from typing import Union
import logging
import pybedtools # type: ignore[import-untyped]
from pybiomart import Server # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ------------- IMPORT THESE FOR THE FINAL PIPELINE, DONT KEEP THEM HERE ----------------
def plot_feature_score_histogram(df, score_col, fig_dir):
    logging.info("\tPlotting feature score histogram")
    
    df_series = df[score_col]
    
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))

    plt.hist(df_series.dropna(), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f"{score_col}", fontsize=14)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlim((0, 1))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{score_col}_hist.png"), dpi=300)
    plt.close()


def minmax_normalize_pandas(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """
    Applies global min-max normalization to selected columns in a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame containing the columns to normalize.
    score_cols : list of str
        List of column names to normalize.

    Returns
    -------
    pd.DataFrame
        Normalized Pandas DataFrame.
    """
    df = df.copy()
        
    for col in score_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df.loc[:, col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df.loc[:, col] = 0.0
    return df

# ----------------------------------------------------------------------------------------


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


def convert_anndata_to_pandas(adata: anndata.AnnData, id_col_name: str) -> pd.DataFrame:
    """
    Convert an AnnData object to a Pandas DataFrame.
    
    - **var_names** = gene / peak names
    - **obs_names** = cell names / barcodes

    Args:
        adata (AnnData): AnnData object containing scATAC-seq or scRNA-seq data
        id_col_name (str): Name for the peak / gene ID column ("peak_id" or "gene_id")

    Returns:
        pd.DataFrame: DataFrame of gene x cell expression data. Header contains cell names, column 0 
        contains gene / peak names
    """
    adata = adata.copy()
    
    df = pd.DataFrame(
        data=adata.X.T.toarray(),
        index=adata.var_names,    # genes
        columns=adata.obs_names    # filtered cells
    )
    
    # Add the gene / peak names as column 0 rather than the index
    df.insert(loc=0, column=id_col_name, value=df.index.astype(str))
    df = df.reset_index(drop=True)

    return df


def write_processed_dataframe_to_parquet(df: pd.DataFrame, data_file_path: str) -> None:
    """
    Writes the processed DataFrame object to a '_processed.parquet' file.
    
    The data_file_path argument should be the same as the path to the raw data file. This function
    saves the processed parquet file to the same location as the input data, but adds '_processed.parquet'
    to the end of the file.

    Args:
        df (pd.DataFrame): DataFrame of gene x cell expression data. Header contains cell names, column 0 
        contains gene / peak names
        data_file_path (str): Path to the input data file.
    """

    if not "_processed.parquet" in data_file_path:
    
        def update_name(filename):
            base, ext = os.path.splitext(filename)
            return f"{base}_processed.parquet"

        data_file_path = update_name(data_file_path)
        logging.info(f"  - Updated file: {data_file_path}")
        
        
        
    else:
        logging.info(f"  - Save file already contains '_processed.parquet' ({os.path.basename(data_file_path)}), skipping renaming")
    
    logging.info(f'  - Writing processed dataset to {data_file_path}')
    df.to_parquet(data_file_path, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")


def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    """
    Loads an scATAC-seq dataset from a CSV, TSV, or parquet file.
    
    - **Cell Names**: First row, set as header
    - **Peak Location**: First column

    Args:
        atac_data_file (str): Path to the scATAC-seq data file.

    Raises:
        ValueError: ATAC data file must be .csv, .tsv, or .parquet
        RuntimeError: DataFrame was empty after loading the input file

    Returns:
        pd.DataFrame: DataFrame with column 0 as 'peak_id' and the header as cell barcodes
    """
    
    df: pd.DataFrame = pd.DataFrame()
    if atac_data_file.lower().endswith('.parquet'):
        df = pd.read_parquet(atac_data_file)
        
    elif atac_data_file.lower().endswith('.csv'):
        df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
        
    elif atac_data_file.lower().endswith('.tsv'):
        df = pd.read_csv(atac_data_file, sep="\t", header=0, index_col=None)
        
    else:
        raise ValueError(f"ATAC data file must be .csv, .tsv or .parquet: got {atac_data_file}")
    
    if df.empty:
        raise RuntimeError(f"Failed to load ATAC file: {atac_data_file}")
    
    df = df.rename(columns={df.columns[0]: "peak_id"})
    
    return df


def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    if rna_data_file.lower().endswith('.parquet'):
        df = pd.read_parquet(rna_data_file)
        
    elif rna_data_file.lower().endswith('.csv'):
        df = pd.read_csv(rna_data_file, sep=",", header=0, index_col=None)
        
    elif rna_data_file.lower().endswith('.tsv'):
        df = pd.read_csv(rna_data_file, sep="\t", header=0, index_col=None)
        
    else:
        raise ValueError(f"RNA data file must be .csv, .tsv or .parquet: got {rna_data_file}")
    
    df = df.rename(columns={df.columns[0]: "gene_id"})
    
    if df.empty:
        raise RuntimeError(f"Failed to load RNA file: {rna_data_file}")
    
    logging.info(f'\tNumber of genes: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1]-1}')
    
    return df


def extract_atac_peaks(atac_df, tmp_dir):
    
    if not os.path.exists(f"{tmp_dir}/peak_df.parquet"):
        logging.info(f"Extracting peak information and saving as a bed file")
        def parse_peak_str(s):
            try:
                chrom, coords = s.split(":")
                start_s, end_s = coords.split("-")
                return chrom.replace("chr", ""), int(start_s), int(end_s)
            except Exception:
                raise ValueError(f"Malformed peak_id '{s}'; expected 'chrN:start-end'.")

        # List of peak strings
        peak_pos = atac_df["peak_id"].tolist()

        # Apply parsing function to all peak strings
        parsed = [parse_peak_str(s) for s in peak_pos]

        # Construct DataFrame
        peak_df = pd.DataFrame(parsed, columns=["chr", "start", "end"])
        peak_df["peak_id"] = peak_pos
        
        # Write the peak DataFrame to a file
        peak_df.to_parquet(os.path.join(tmp_dir, "peak_df.parquet"), engine="pyarrow", index=False, compression="snappy")
        
    else:
        logging.info("ATAC-seq peak_df.parquet file exists, loading...")
        peak_df = pd.read_parquet(os.path.join(tmp_dir, "peak_df.parquet"), engine="pyarrow")
        
    return peak_df


def load_ensembl_organism_tss(organism, tmp_dir):
    if not os.path.exists(os.path.join(tmp_dir, "ensembl.parquet")):
        logging.info(f"Loading Ensembl TSS locations for {organism}")
        # Connect to the Ensembl BioMart server
        server = Server(host='http://www.ensembl.org')

        gene_ensembl_name = f'{organism}_gene_ensembl'
        
        # Select the Ensembl Mart and the human dataset
        mart = server['ENSEMBL_MART_ENSEMBL']
        try:
            dataset = mart[gene_ensembl_name]
        except KeyError:
            raise RuntimeError(f"BioMart dataset {gene_ensembl_name} not found. Check if ‘{organism}’ is correct.")

        # Query for attributes: Ensembl gene ID, gene name, strand, and transcription start site (TSS)
        ensembl_df = dataset.query(attributes=[
            'external_gene_name', 
            'strand', 
            'chromosome_name',
            'transcription_start_site'
        ])

        ensembl_df.rename(columns={
            "Chromosome/scaffold name": "chr",
            "Transcription start site (TSS)": "tss",
            "Gene name": "gene_id"
        }, inplace=True)
        
        # Make sure TSS is integer (some might be floats).
        ensembl_df["tss"] = ensembl_df["tss"].astype(int)

        # In a BED file, we’ll store TSS as [start, end) = [tss, tss+1)
        ensembl_df["start"] = ensembl_df["tss"].astype(int)
        ensembl_df["end"] = ensembl_df["tss"].astype(int) + 1

        # Re-order columns for clarity: [chr, start, end, gene]
        ensembl_df = ensembl_df[["chr", "start", "end", "gene_id"]]
        
        ensembl_df["chr"] = ensembl_df["chr"].astype(str)
        ensembl_df["gene_id"] = ensembl_df["gene_id"].astype(str)
        
        # Write the peak DataFrame to a file
        ensembl_df.to_parquet(os.path.join(tmp_dir, "ensembl.parquet"), engine="pyarrow", index=False, compression="snappy")
        
    else:
        logging.info("Ensembl gene TSS BED file exists, loading...")
        ensembl_df = pd.read_parquet(os.path.join(tmp_dir, "ensembl.parquet"), engine="pyarrow")
    
    return ensembl_df


def extract_atac_peaks_near_rna_genes(
    atac_df: pd.DataFrame, 
    gene_names: set[str], 
    organism: str, 
    tss_distance_cutoff: Union[int, float], 
    output_dir: str
    ) -> pd.DataFrame:
    """
    Identify genes whose transcription start sites (TSS) are near scATAC-seq peaks.
    
    This function:
        1. Uses BedTools to find peaks that are within peak_dist_limit bp of each gene's TSS.
        2. Converts the BedTool result to a pandas DataFrame.
        3. Computes the absolute distance between the peak end and gene start (as a proxy for TSS distance).
        4. Scales these distances using an exponential drop-off function (e^-dist/250000),
           the same method used in the LINGER cis-regulatory potential calculation.
        5. Deduplicates the data to keep the minimum (i.e., best) peak-to-gene connection.
        6. Only keeps genes that are present in the RNA-seq dataset.
        
    Parameters
    ----------
    atac_df (pd.DataFrame):
        DataFrame of scATAC-seq peak x cell counts. Must contain column "peak_id" with peaks in format chr:start-end.
    gene_names (set[str]):
        The set of gene names in the scRNA-seq datset.
    organism (str):
        The Ensembl organism name for downloading the gene TSS locations ("hsapiens" or "mmusculus")
    tss_distance_cutoff : int
        The maximum distance (in bp) from a TSS to consider a peak as potentially regulatory. Default 1e6.
    output_dir: str
        Output directory for the sample, used to save the peaks_near_genes.parquet file

        
    Returns
    -------
    peaks_near_genes_df : pandas.DataFrame
        A DataFrame containing columns "peak_id", "target_id", and the scaled TSS distance "TSS_dist"
        for peak–gene pairs.
    """
    
    if not os.path.exists(os.path.join(output_dir, "peaks_near_genes.parquet")):
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        if organism not in ("hsapiens", "mmusculus"):
            if organism == "hg38":
                organism = "hsapiens"
            elif organism == "mm10":
                organism = "mmusculus"
            else:
                raise ValueError(f"Organism not recognized: {organism} (must be 'hg38' or 'mm10').")

        peak_df: pd.DataFrame = extract_atac_peaks(atac_df, tmp_dir)
        tss_df: pd.DataFrame = load_ensembl_organism_tss(organism, tmp_dir)

        pybedtools.set_tempdir(tmp_dir)
        try:
            peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
            tss_bed = pybedtools.BedTool.from_dataframe(tss_df)
            
            # 3) Find peaks that are within peak_dist_limit bp of each gene's TSS using BedTools
            logging.info(f"Locating peaks that are within {tss_distance_cutoff} bp of each gene's TSS")
            peak_tss_overlap = peak_bed.window(tss_bed, w=tss_distance_cutoff)
            
            # Define the column types for conversion to DataFrame
            dtype_dict = {
                "peak_chr": str,
                "peak_start": int,
                "peak_end": int,
                "peak_id": str,
                "gene_chr": str,
                "gene_start": int,
                "gene_end": int,
                "gene_id": str
            }
            
            # Convert the BedTool result to a DataFrame for further processing.
            peaks_near_genes_df = peak_tss_overlap.to_dataframe(
                names = [
                    "peak_chr", "peak_start", "peak_end", "peak_id",
                    "gene_chr", "gene_start", "gene_end", "gene_id"
                ],
                dtype=dtype_dict,
                low_memory=False  # ensures the entire file is read in one go
            ).rename(columns={"gene_id": "target_id"}).dropna()
            
            # Calculate the absolute distance between the peak's end and gene's start.
            # This serves as a proxy for the TSS distance for the peak-to-gene pair.
            distances = np.abs(peaks_near_genes_df["peak_end"].values - peaks_near_genes_df["gene_start"].values)
            peaks_near_genes_df["TSS_dist"] = distances
            
            # Scale the TSS distance using an exponential drop-off function
            # e^-dist/25000, same scaling function used in LINGER Cis-regulatory potential calculation
            # https://github.com/Durenlab/LINGER
            peaks_near_genes_df["TSS_dist_score"] = np.exp(-peaks_near_genes_df["TSS_dist"] / 250000)
            
            # Keep only the necessary columns.
            peaks_near_genes_df = peaks_near_genes_df[["peak_id", "target_id", "TSS_dist_score"]]
            
            # Filter out any genes not found in the RNA-seq dataset.
            gene_names_upper = set(g.upper() for g in gene_names)
            mask = peaks_near_genes_df["target_id"].str.upper().isin(gene_names_upper)
            peaks_near_genes_df = peaks_near_genes_df[mask]
            
            logging.info(f'\t- Number of peaks: {len(peaks_near_genes_df.drop_duplicates(subset="peak_id"))}')
            
            peaks_near_genes_df = minmax_normalize_pandas(
                df=peaks_near_genes_df, 
                score_cols=["TSS_dist_score"], 
            )
                
            peaks_near_genes_df.to_parquet(os.path.join(output_dir, "peaks_near_genes.parquet"), index=False, engine="pyarrow", compression="snappy")
        
        finally:
            pybedtools.helpers.cleanup(verbose=False, remove_all=True)
    
    else:
        logging.info('TSS distance file exists, loading...')
        peaks_near_genes_df = pd.read_parquet(os.path.join(output_dir, "peaks_near_genes.parquet"), engine="pyarrow")
    
    
    return peaks_near_genes_df


def atac_data_preprocessing(
    atac_data_path: str, 
    barcodes: list[str],
    h5ad_save_path: str,
    filter_peak_min_cells: int = 30, 
    min_peaks_per_cell: int = 1000,
    target_read_depth: float = 1e6,
    fig_dir: str = 'figures',
    plot_peaks_by_counts: bool = True,
    overwrite: bool = False
    ) -> anndata.AnnData:
    """
    QC filtering and preprocessing of an ATACseq AnnData object.

    Args:
        atac_data_path (str): 
            Path to the input ATAC data (.csv, .tsv. or .parquet file).
        atac_h5ad_save_path (str): 
            Path to save the processed ATAC AnnData object as an h5ad file.
        barcodes (list[str]):
            A list of paired barcodes from the RNAseq dataset.
        filter_peak_min_cells (int, optional): 
            A peak must be be expressed in greater than this number of cells. Defaults to 30.
        min_peaks_per_cell (int, optional): 
            A cell must be expressing more than this number of peaks. Defaults to 1000.
        target_read_depth (float, optional):
            Normalizes counts per cell to this value. Defaults to 1e6 (CPM normalization)/
        fig_dir (str, optional): 
            Figure for saving the `accessibility_genes_by_counts.png` figure. Defaults to 'figures'.
        plot_peaks_by_counts (bool, optional): 
            True to plot the figure, False to skip plotting. Defaults to True.
        h5ad_save_path (None | str): 
            Path to save the processed ATAC AnnData object as an h5 file.
        overwrite (bool):
            Set to True to overwrite the h5ad file if it exists. Defaults to False.

    Returns:
        atac_adata (anndata.AnnData): Filtered ATAC AnnData object
    """
    if not os.path.isfile(atac_data_path):
        raise FileNotFoundError(f"ATAC file not found: {atac_data_path}")
    
    if not barcodes:
        raise Exception("barcodes argument is None or empty, pass in the cell names / barcodes \
            from the scRNA-seq dataset")
    
    file_missing = not os.path.isfile(h5ad_save_path)
    if file_missing or overwrite:
        if "_processed" in os.path.basename(atac_data_path):
            raise Exception("Use scATAC-seq dataset with raw counts, raw counts \
                required when fitting the MIRA LITE model")
    
    
        logging.info("  - Reading ATACseq raw data")
        raw_atac_df = load_atac_dataset(atac_data_path)
        
        atac_adata = anndata_from_dataframe(raw_atac_df, "peak_id")
        
        logging.info(f"    - Number of Cells (unfiltered): {atac_adata.shape[0]}")
        logging.info(f"    - Number of Peaks (unfiltered): {atac_adata.shape[1]-1}")
        
        logging.info("    (1/5) Filtering out very rare peaks")
        sc.pp.filter_genes(atac_adata, min_cells = filter_peak_min_cells)

        valid_barcodes = [bc for bc in barcodes if bc in atac_adata.obs_names]
        
        if len(valid_barcodes) == 0:
            raise Exception("No matches between barcodes and atac_adata.obs_names")
        
        atac_adata = atac_adata[valid_barcodes]
        
        logging.info("    (2/5) Calculating QC metrics")
        sc.pp.calculate_qc_metrics(atac_adata, inplace=True, log1p=False)
        
        if plot_peaks_by_counts:
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
                    os.path.join(qc_fig_path, "accessibility_peaks_by_counts.png"),
                    dpi=200,
                    bbox_inches="tight"
                )

        logging.info(f"    (3/5) Filtering cells by {min_peaks_per_cell} min peaks per cell")
        sc.pp.filter_cells(atac_adata, min_genes=min_peaks_per_cell)
        
        logging.info(f"    (4/5) Normalizing to a read depth of {target_read_depth}")
        sc.pp.normalize_total(atac_adata, target_sum=target_read_depth)

        logging.info("    (5/5) Logarithmizing the data")
        sc.pp.log1p(atac_adata)

        logging.info(f"    (5/5) Subsampling to 1e5 peaks per cell")
        # # If needed, reduce the size of the dataset by subsampling
        np.random.seed(0)
        atac_adata.var['endogenous_peaks'] = np.random.rand(atac_adata.shape[1]) <= min(1e5/atac_adata.shape[1], 1)
        
        logging.info(f"    - Number of Cells (filtered): {atac_adata.shape[0]}")
        logging.info(f"    - Number of Peaks (filtered): {atac_adata.shape[1]-1}")
        
        if h5ad_save_path:
            logging.info(f"    Writing h5ad file to {os.path.basename(h5ad_save_path)}")
            atac_adata.write_h5ad(h5ad_save_path)
    
        return atac_adata
    
    else:
        logging.info(f"  - Loading existing h5ad file found at {h5ad_save_path}")
        return anndata.read_h5ad(h5ad_save_path)


def rna_data_preprocessing(
    rna_data_path: str, 
    rna_h5ad_save_path: str,
    min_cells_per_gene: int = 15,
    target_read_depth: float = 1e6, 
    min_gene_disp: float = 0.5,
    min_genes: int = 200,
    max_genes: int = 2500,
    max_pct_mt: float = 5.0,
    overwrite: bool = False
    ) -> anndata.AnnData:
    """
    Runs QC filtering and preprocessing for scRNA-seq data.
    
    Args:
        rna_data_path (str): 
            Path to the raw, unprocessed scRNA-seq gene x cell count matrix.
        rna_h5ad_save_path (str):
            Path to save the processed and filtered scRNA-seq AnnData object.
        min_cells_per_gene (int):
            Genes must be expressed in at least this number of cells. Defaults to 15.
        target_read_depth (float, optional): 
            Normalizes the read depth of each cell. Defaults to 1e6.
        min_gene_disp (float, optional): 
            Minimum gene variability by dispersion. Defaults to 0.5.
        min_genes (int, optional):
            Cells must be expressing at least this number of genes. Defaults to 200.
        max_genes (int, optional):
            Cells cannot be expressing over this number of genes. Defaults to 2500.
        max_pct_mt (float, optional):
            Cells cannot be expressing over this percentage of mitochondrial genes. Defaults to 5%.
        h5ad_save_path (None | str): 
            Path to save the processed RNA AnnData object as an h5 file.
        overwrite (bool):
            Set to True to overwrite the h5ad file if it exists. Defaults to False.

    Returns:
        rna_adata (anndata.AnnData): Filtered RNA AnnData object
    """
    file_missing = not os.path.isfile(rna_h5ad_save_path)
    if file_missing or overwrite:
        logging.info("  - Reading RNAseq raw data parquet file")
        rna_data = load_rna_dataset(rna_data_path)
        logging.info(f"    - Number of Cells (unfiltered): {rna_data.shape[0]}")
        logging.info(f"    - Number of Genes (unfiltered): {rna_data.shape[1]-1}")

        logging.info("  - Converting DataFrame to AnnData object")
        rna_adata = anndata_from_dataframe(rna_data, "gene_id")
        
        logging.info("    (1/6) Filtering out mitochondrial, ribosomal, and hemoglobin genes")
        #    Mitochondrial: genes whose name starts with "MT-" (case-insensitive).
        #    Ribosomal:     genes whose name starts with "RPS" or "RPL".
        #    Hemoglobin:    genes whose name matches "^HB(?!P)" (so HB* but not HBP).
        var_names_lower = rna_adata.var_names.str.lower()

        rna_adata.var["mt"] = var_names_lower.str.startswith("mt-")
        rna_adata.var["ribo"] = var_names_lower.str.startswith(("rps", "rpl"))
        rna_adata.var["hb"] = var_names_lower.str.contains(r"^hb(?!p)")
        
        sc.pp.calculate_qc_metrics(
            rna_adata,
            qc_vars=["mt", "ribo", "hb"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        
        cell_mask = (
            (rna_adata.obs["n_genes_by_counts"] >= min_genes)
            & (rna_adata.obs["n_genes_by_counts"] <= max_genes)
            & (rna_adata.obs["pct_counts_mt"] < max_pct_mt)
        )

        n_before = rna_adata.n_obs
        n_after = cell_mask.sum()
        logging.info(f"    (2/6) Filtering genes expressed in fewer than {min_cells_per_gene} cells (after cell filtering)")
        logging.info(f"\t - Cells before filtering: {n_before}")
        logging.info(f"\t - Cells after filtering : {n_after}  (kept those with {min_genes} ≤ n_genes ≤ {max_genes} and pct_counts_mt < {max_pct_mt}%)")

        if n_after == 0:
            raise RuntimeError("No cells passed the filtering criteria. "
                            "Check `min_genes`, `max_genes`, `max_pct_mt` settings.")

        filtered_adata = rna_adata[cell_mask].copy()
        
        logging.info("    (3/6) Filtering out very rare genes")
        sc.pp.filter_genes(filtered_adata, min_cells=min_cells_per_gene)
        rawdata = filtered_adata.X.copy()
        
        logging.info(f"    (4/6) Normalizing to a read depth of {target_read_depth}")
        sc.pp.normalize_total(filtered_adata, target_sum=target_read_depth)

        logging.info("    (5/6) Log1p normalizing the data")
        sc.pp.log1p(filtered_adata)

        logging.info(f"    (6/6) Filtering for highly variable genes with dispersion > {min_gene_disp}")
        sc.pp.highly_variable_genes(filtered_adata, min_disp = min_gene_disp)
        
        logging.info(f"    - Number of Cells (filtered): {filtered_adata.shape[0]}")
        logging.info(f"    - Number of Genes (filtered): {filtered_adata.shape[1]-1}")

        filtered_adata.layers['counts'] = rawdata

        if rna_h5ad_save_path:
            logging.info(f"    Writing h5ad file to {os.path.basename(rna_h5ad_save_path)}")
            filtered_adata.write_h5ad(rna_h5ad_save_path)
        
        return filtered_adata
                
    else:
        logging.info("  - RNA h5ad file found, loading")
        return anndata.read_h5ad(rna_h5ad_save_path)


def filter_atac_by_distance_to_tss(
    atac_df: pd.DataFrame, 
    gene_names: Union[list[str], set[str]],
    species: str,
    tss_distance_cutoff: Union[int, float],
    output_dir: str,
    fig_dir: str,
    ) -> anndata.AnnData:
    """
    Measures the distance between each peaks and gene transcription start sites within tss_distance_cutoff base pairs of each peak.
    
    Filters the scATAC-seq data to only include peaks within `tss_distance_cutoff` base pairs of a gene
    in the scRNA-seq datset. Saves the TSS distances and TSS distance scores to the `output_dir` directory
    as `peaks_near_genes.parquet`.
    
    If the h5ad file exists at `atac_h5ad_save_path`, the function returns the ATAC AnnData object. 
    Writes the procesed data to the same location as `atac_data_path` but with the file ending in
    `_processed.parquet` if it does not exist.

    Args:
        atac_df (pd.DataFrame): 
            Processed scATAC-seq gene x cell DataFrame. Must contain the column "peak_id" with peak locations
            in the format of "chr:start-end".
        gene_names (list[str] | set[str]):
            List of gene names from the scRNA-seq dataset.
        species (str):
            Ensembl species name for loading gene TSS locations ('hsapiens' or 'mmusculus').
        tss_distance_cutoff (int | float):
            Maximum distance in base pairs to filter when associating peaks to potential target genes by distance.
        output_dir (str):
            Path to the output directory for the sample.
        fig_dir (str): 
            Path to the figure directory for the sample.

    Raises:
        Exception: barcodes argument must contain cell names / barcodes.
        FileNotFoundError: ATAC data file path must exist.

    Returns:
        anndata.AnnData: AnnData object of the processed scATAC-seq dataset.
    """
    
    if not os.path.isdir(output_dir):
        raise Exception(f"Output directory {output_dir} does not exist")
        
    logging.info("  - Extracting ATAC peaks within 1 MB of a gene from the RNA dataset")
    gene_names_set = set(gene_names)
    
    peaks_near_genes_df = extract_atac_peaks_near_rna_genes(
        atac_df, 
        gene_names_set, 
        species, 
        tss_distance_cutoff, 
        output_dir
        )
    
    plot_feature_score_histogram(peaks_near_genes_df, "TSS_dist_score", fig_dir)
    
    logging.info("  - Filtering for peaks with 1MB of a gene's TSS")
    peak_subset = set(peaks_near_genes_df["peak_id"])
    atac_df_filtered = atac_df[atac_df["peak_id"].isin(peak_subset)]
    logging.info(f'\tNumber of peaks after filtering: {len(atac_df_filtered)} / {len(atac_df)}')

    logging.info("  - Converting DataFrame to AnnData object")
    atac_adata = anndata_from_dataframe(atac_df_filtered, "peak_id")
    
    return atac_adata
    

