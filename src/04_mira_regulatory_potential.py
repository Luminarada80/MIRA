import os
import mira
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
matplotlib.rc('font',size=12)
import logging
import warnings
warnings.simplefilter("ignore")
mira.utils.pretty_sderr()

# ======== SET VARIABLES AND FILE PATHS ==========
BASE_DIR = "/gpfs/Home/esm5360/MIRA/"
FIG_DIR = os.path.join(BASE_DIR, "figures/joint_representation")
TUNER_DIR = os.path.join(BASE_DIR, "tuners")
DATASET_DIR = os.path.join(BASE_DIR, "mira-datasets/mESC_filtered_L2_E7.5_rep1")
DATASET_NAME = "mESC_E7.5_rep1"
NUM_CPU = 2

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, f'data/{DATASET_NAME}_rpmodels/'), exist_ok=True)

atac_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_topic_analysis.h5ad")
rna_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_topic_analysis.h5ad")

atac_model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model.pth")
rna_model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model.pth")
# ================================================

# TSS Annotations
rna_adata = anndata.read_h5ad(rna_h5ad_save_path)
atac_adata = anndata.read_h5ad(atac_h5ad_save_path)

rna_model = mira.topics.load_model(rna_model_save_path)
atac_model = mira.topics.load_model(atac_model_save_path)

mira.datasets.mm10_chrom_sizes()
mira.datasets.mm10_tss_data()

atac_adata.var["chr"] = atac_adata.var["chr"].astype(str)

mira.tl.get_distance_to_TSS(atac_adata,
                            tss_data='mira-datasets/mm10_tss_data.bed12',
                            genome_file='mira-datasets/mm10.chrom.sizes')

atac_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_tss_dist.h5ad"), convert_strings_to_categoricals = False)

# RP Model Training
rp_args = dict(expr_adata = rna_adata, atac_adata= atac_adata)

# Set the list of gene names (all highly variable genes + top 200 genes from each topic)
rp_genes = list(rna_model.features[rna_model.highly_variable])
for topic in range(rna_model.num_topics):
    rp_genes.extend(rna_model.get_top_genes(topic, 200))
rp_genes = list(set(g.capitalize() for g in rp_genes if g.capitalize() in rna_adata.var_names))

# ----- LITE MODEL -----
litemodel = mira.rp.LITE_Model(expr_model = rna_model,
                              accessibility_model= atac_model,
                              genes = rp_genes)

# Fit the LITE model
rna_adata.X = rna_adata.layers["counts"]
litemodel.fit(
    **rp_args,
    n_workers=32,
    callback=mira.rp.SaveCallback(os.path.join(BASE_DIR, f'data/{DATASET_NAME}_rpmodels/'))
)

litemodel.predict(**rp_args)

def compute_rp_score(row, params):
    dist_kb = row['distance_to_TSS'] / 1000  # convert bp → kb
    if row['is_upstream']:
        decay = params['distance_upstream']
        weight = params['a_upstream']
    else:
        decay = params['distance_downstream']
        weight = params['a_downstream']
    return weight * np.exp(-dist_kb / decay)

all_rp_records = []

missing_genes = 0

for gene in rna_adata.var_names:
    try:
        params = litemodel[gene].parameters_
        peaks_df = litemodel[gene].get_influential_local_peaks(atac_adata, decay_periods=5.)

        peaks_df['MIRA_LITE_RP_score'] = peaks_df.apply(lambda row: compute_rp_score(row, params), axis=1)

        df = peaks_df[["distance_to_TSS", "MIRA_LITE_RP_score"]].rename_axis("peak_id").reset_index()
        df["target_id"] = gene
        df = df[["peak_id", "target_id", "distance_to_TSS", "MIRA_LITE_RP_score"]]

        all_rp_records.append(df)

    except IndexError:
        missing_genes += 1
    except KeyError:
        print(f"Gene {gene} not found in litemodel — skipping.")
    except Exception as e:
        print(f"Error processing {gene}: {e}")

print(f"Missing models for {missing_genes}")

regulatory_potential_df = pd.concat(all_rp_records, ignore_index=True)

regulatory_potential_df.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_peak_to_gene_lite_rp_score.csv"))

rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_lite_pred.h5ad"))
atac_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_lite_pred.h5ad"))

# Saving the TSS decay parameters for each TF
TSS_dist_decay_df = pd.DataFrame(
    litemodel.parameters_
).T
TSS_dist_decay_df.to_csv(os.path.join(DATASET_DIR, F"{DATASET_NAME}_gene_tss_decay_parameters.tsv"), sep="\t", header=True, index=True)

# Probabilistic in-silico deletion (TF-TG association scores)
litemodel.probabilistic_isd(**rp_args, n_workers = 32)
isd_matrix = mira.utils.fetch_ISD_matrix(rna_adata) # ISD results stored in RNA AnnData

rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_tf_tg_scores.h5ad"))
isd_matrix.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mira_tf_tg.tsv"), sep="\t", header=True, index=True)

# ---- NITE MODEL -----
nitemodel = litemodel.spawn_NITE_model()
nitemodel.fit(**rp_args, n_workers=32)
nitemodel.predict(**rp_args)
nitemodel.save(os.path.join(DATASET_DIR, f"data/{DATASET_NAME}_rpmodels/"))

rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_nite_pred.h5ad"))
atac_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_nite_pred.h5ad"))

mira.tl.get_chromatin_differential(rna_adata)

gene_cell_chromatin_diff = pd.DataFrame(rna_adata.layers["chromatin_differential"].toarray().T, columns=rna_adata.obs_names, index=rna_adata.var_names)
gene_cell_chromatin_diff.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_chromatin_differential.csv"))

avg_chrom_diff = gene_cell_chromatin_diff.mean(axis=1)
regulatory_potential_df["avg_chromatin_differential"] = regulatory_potential_df["target_id"].map(avg_chrom_diff)

regulatory_potential_df.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mira_peak_to_tg_scores.csv"))
