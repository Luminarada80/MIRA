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

atac_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_joint_representation.h5ad")
rna_h5ad_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_joint_representation.h5ad")

atac_model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_model.pth")
rna_model_save_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_model.pth")
# ================================================

rna_adata = anndata.read_h5ad(rna_h5ad_save_path)
atac_adata = anndata.read_h5ad(atac_h5ad_save_path)

rna_model = mira.topics.load_model(rna_model_save_path)
atac_model = mira.topics.load_model(atac_model_save_path)

num_genes = rna_adata.X.shape[0]
top_n_genes = math.ceil(num_genes * 0.05)

rna_model.post_topics(top_n=top_n_genes)

rna_model.fetch_enrichments(ontologies=['WikiPathways_2019_Mouse'])

mm10_fasta_file = os.path.join(BASE_DIR, "data/mm10.fa")

peak_locations = atac_adata.var.index

if not any(["chr", "start", "end"]) in peak_locations:
    peak_data: dict[str, list] = {
        "peak_id": [],
        "chr": [],
        "start": [],
        "end": []
    }
    for i, peak in enumerate(peak_locations):
        peak_id = i
        chr_num = peak.split(":")[0]
        peak_start = int(peak.split(":")[1].split("-")[0])
        peak_end = int(peak.split(":")[1].split("-")[1])
        
        peak_data["peak_id"].append(peak_id)
        peak_data["chr"].append(chr_num)
        peak_data["start"].append(peak_start)
        peak_data["end"].append(peak_end)
        
    peak_df = pd.DataFrame(peak_data, index=peak_locations)
    atac_adata.var = pd.concat([atac_adata.var, peak_df], axis=1)
    
os.environ["PATH"] = os.pathsep.join([
    os.path.expanduser("~/miniconda3/envs/mira-env/bin"),
    os.environ["PATH"]
])

mira.tools.motif_scan.logger.setLevel(logging.INFO) # make sure progress messages are displayed
mira.tl.get_motif_hits_in_peaks(atac_adata,
                    genome_fasta=os.path.join(BASE_DIR, 'data/mm10.fa'),
                    chrom = 'chr', start = 'start', end = 'end',
                    pvalue_threshold=1e-4
                    ) # indicate chrom, start, end of peaks

mira.utils.subset_factors(atac_adata,
                          use_factors=[factor.upper() for factor in rna_adata.var_names
                                       if not ('FOS' in factor or 'JUN' in factor)])

topics = [int(i.replace("topic_", "")) for i in atac_adata.obs if "topic" in i]
for topic in topics:
    atac_model.get_enriched_TFs(atac_adata, topic_num=topic, top_quantile=0.1)
    
motif_scores = atac_model.get_motif_scores(atac_adata)

# Reformat to make it convenient for plotting
motif_scores.var = motif_scores.var.set_index('parsed_name')
motif_scores.var_names_make_unique()
motif_scores.obsm['X_umap'] = atac_adata.obsm['X_umap']

rna_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_rna_data_topic_analysis.h5ad"))
atac_adata.write_h5ad(os.path.join(DATASET_DIR, f"{DATASET_NAME}_atac_data_topic_analysis.h5ad"))