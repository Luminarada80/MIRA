import os
import mira
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import warnings
import argparse

# Style and config
plt.rcParams['font.size'] = 12
warnings.simplefilter("ignore")
mira.utils.pretty_sderr()

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the MIRA directory"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Descriptive name for the dataset being analyzed"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    BASE_DIR = args.base_dir
    DATASET_NAME = args.dataset_name

    FIG_DIR = os.path.join(BASE_DIR, "figures")
    TUNER_DIR = os.path.join(BASE_DIR, "tuners")
    DATASET_DIR = os.path.join(BASE_DIR, f"mira-datasets/{DATASET_NAME}")

    NUM_CPU = 2

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TUNER_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # File paths
    atac_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_topic_analysis.h5ad")
    rna_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_topic_analysis.h5ad")
    atac_model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model.pth")
    rna_model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model.pth")

    # ========== LOAD DATA ==========
    rna_adata = anndata.read_h5ad(rna_h5ad_save_path)
    atac_adata = anndata.read_h5ad(atac_h5ad_save_path)

    rna_model = mira.topics.load_model(rna_model_save_path)
    atac_model = mira.topics.load_model(atac_model_save_path)
    
    os.makedirs(os.path.join(BASE_DIR, f"data/{DATASET_NAME}_rpmodels"), exist_ok=True)

    # TSS Annotations
    mira.datasets.mm10_chrom_sizes()
    mira.datasets.mm10_tss_data()
    atac_adata.var["chr"] = atac_adata.var["chr"].astype(str)

    mira.tl.get_distance_to_TSS(
        atac_adata,
        tss_data='mira-datasets/mm10_tss_data.bed12',
        genome_file='mira-datasets/mm10.chrom.sizes'
    )

    atac_adata.write_h5ad(
        os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_tss_dist.h5ad"),
        convert_strings_to_categoricals=False
    )

    # ========== RP MODEL TRAINING ==========
    rp_args = dict(expr_adata=rna_adata, atac_adata=atac_adata)

    tss_bed_file = os.path.join(BASE_DIR, "mira-datasets/mm10_tss_data.bed12")

    # Load as a DataFrame
    tss_df = pd.read_csv(
        tss_bed_file,
        sep="\t",
        header=None,
        names=[
            "chrom", "start", "end", "name", "score", "strand",
            "thickStart", "thickEnd", "itemRgb", "blockCount",
            "blockSizes", "blockStarts"
        ]
    )

    # Extract gene names from the 'name' column
    annotated_genes = set(tss_df["name"].unique())

    # Filter your gene list to only those with valid TSS annotation
    rp_genes = [g for g in rna_adata.var_names if g in annotated_genes]

    # LITE Model
    print("Creating LITE model")
    litemodel = mira.rp.LITE_Model(expr_model=rna_model, accessibility_model=atac_model, genes=rp_genes)
    rna_adata.X = rna_adata.layers["counts"]
    litemodel.fit(**rp_args, n_workers=32, callback=mira.rp.SaveCallback(os.path.join(BASE_DIR, f'data/{DATASET_NAME}_rpmodels/')))
    litemodel.predict(**rp_args)
    
    # NITE Model
    print("Creating NITE model")
    nitemodel = litemodel.spawn_NITE_model()
    nitemodel.fit(**rp_args, n_workers=32)
    nitemodel.predict(**rp_args)
    nitemodel.save(os.path.join(BASE_DIR, f"data/{DATASET_NAME}_rpmodels/"))

    # ========== BUILD RP MATRICES MANUALLY ==========
    n_peaks = atac_adata.shape[1]
    n_genes = rna_adata.shape[1]
    peak_names = list(atac_adata.var_names)
    gene_names = list(rna_adata.var_names)
    peak_index = {p: i for i, p in enumerate(peak_names)}
    gene_index = {g: i for i, g in enumerate(gene_names)}

    # --- LITE ---
    print("Extracting LITE model results")
    rp_matrix_lite = lil_matrix((n_peaks, n_genes))
    for gene in gene_names:
        try:
            params = litemodel[gene].parameters_
            distance_upstream = params["distance_upstream"]
            distance_downstream = params["distance_downstream"]
            decay_periods = 1_000 / max(distance_upstream, distance_downstream)
            peaks_df = litemodel[gene].get_influential_local_peaks(atac_adata, decay_periods=decay_periods)
            for _, row in peaks_df.iterrows():
                peak = row.name
                dist_kb = row["distance_to_TSS"] / 1000
                decay = params["distance_upstream"] if row["is_upstream"] else params["distance_downstream"]
                weight = params["a_upstream"] if row["is_upstream"] else params["a_downstream"]
                score = weight * np.exp(-dist_kb / decay)
                if peak in peak_index and gene in gene_index:
                    rp_matrix_lite[peak_index[peak], gene_index[gene]] = score
        except Exception:
            continue

    lite_df = pd.DataFrame.sparse.from_spmatrix(rp_matrix_lite.tocsr(), index=peak_names, columns=gene_names)
    lite_melt = lite_df.stack().reset_index()
    lite_melt.columns = ["peak_id", "target_id", "LITE_score"]

    # --- NITE ---
    print("Extracting NITE model results")
    rp_matrix_nite = lil_matrix((n_peaks, n_genes))
    for gene in gene_names:
        try:
            params = nitemodel[gene].parameters_
            distance_upstream = params["distance_upstream"]  # in kb
            distance_downstream = params["distance_downstream"]
            decay_periods = 1_000 / max(distance_upstream, distance_downstream)

            peaks_df = nitemodel[gene].get_influential_local_peaks(atac_adata, decay_periods=decay_periods)
            for _, row in peaks_df.iterrows():
                peak = row.name
                dist_kb = row["distance_to_TSS"] / 1000
                decay = params["distance_upstream"] if row["is_upstream"] else params["distance_downstream"]
                weight = params["a_upstream"] if row["is_upstream"] else params["a_downstream"]
                score = weight * np.exp(-dist_kb / decay)
                if peak in peak_index and gene in gene_index:
                    rp_matrix_nite[peak_index[peak], gene_index[gene]] = score
        except Exception:
            continue

    nite_df = pd.DataFrame.sparse.from_spmatrix(rp_matrix_nite.tocsr(), index=peak_names, columns=gene_names)
    nite_melt = nite_df.stack().reset_index()
    nite_melt.columns = ["peak_id", "target_id", "NITE_score"]

    # Merge RP scores
    print("Merging LITE and NITE scores")
    regulatory_potential_df = pd.merge(lite_melt, nite_melt, on=["peak_id", "target_id"])

    # Filter to only keep scores where the LITE and/or NITE scores are > 0
    regulatory_potential_df = regulatory_potential_df[
        (regulatory_potential_df["LITE_score"] > 0) |
        (regulatory_potential_df["NITE_score"] > 0)
    ]


    # Save RP scores
    # regulatory_potential_df.to_parquet(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mira_peak_to_tg_scores.parquet"), engine="pyarrow", compression="snappy")

    # Save decay parameters
    pd.DataFrame(litemodel.parameters_).T.to_csv(
        os.path.join(DATASET_DIR, f"{DATASET_NAME}_gene_tss_decay_parameters.tsv"),
        sep="\t", header=True, index=True
    )

    # Save AnnData
    rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_lite_pred.h5ad"))
    atac_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_lite_pred.h5ad"))
    rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_nite_pred.h5ad"))
    atac_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_nite_pred.h5ad"))

    # # ISD + TF-TG scores
    # print("Running probabilistic ISD")
    # litemodel.probabilistic_isd(**rp_args, n_workers=32)
    # isd_matrix = mira.utils.fetch_ISD_matrix(rna_adata)
    # isd_matrix.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mira_tf_tg.tsv"), sep="\t")
    # rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_tf_tg_scores.h5ad"))

    # Chromatin differential
    print("Running chromatin differential")
    mira.tl.get_chromatin_differential(rna_adata)
    gene_cell_chromatin_diff = pd.DataFrame(
        rna_adata.layers["chromatin_differential"].toarray().T,
        columns=rna_adata.obs_names,
        index=rna_adata.var_names
    )

    print("Saving output files")
    gene_cell_chromatin_diff.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_chromatin_differential.csv"))
    avg_chrom_diff = gene_cell_chromatin_diff.mean(axis=1)
    regulatory_potential_df["avg_chromatin_differential"] = regulatory_potential_df["target_id"].map(avg_chrom_diff)

    regulatory_potential_df["LITE_score"] = regulatory_potential_df["LITE_score"].astype(float)
    regulatory_potential_df["NITE_score"] = regulatory_potential_df["NITE_score"].astype(float)

    regulatory_potential_df.to_parquet(
        os.path.join(DATASET_DIR, f"{DATASET_NAME}_mira_peak_to_tg_scores.parquet"),
        engine="pyarrow",
        compression="snappy"
    )
    print("DONE!")
