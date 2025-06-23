import mira
import anndata
import scanpy as sc
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',size=12)

import logging
mira.logging.getLogger().setLevel(logging.INFO)
import warnings
warnings.simplefilter("ignore")
umap_kwargs = dict(
    add_outline=True, outline_width=(0.1,0), outline_color=('grey', 'white'),
    legend_fontweight=350, frameon = False, legend_fontsize=12
)
print(mira.__version__)
mira.utils.pretty_sderr()

rna_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_rna_data.h5ad")
atac_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_atac_data.h5ad")

rna_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_rna_model.pth")
atac_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_atac_model.pth")

atac_model.predict(atac_adata)
rna_model.predict(rna_adata)

rna_model.get_umap_features(rna_adata, box_cox=0.25)
atac_model.get_umap_features(atac_adata, box_cox=0.25)

# Run K-NN and UMAP for RNA data
sc.pp.neighbors(rna_adata, use_rep = 'X_umap_features', metric = 'manhattan', n_neighbors = 21)
sc.tl.umap(rna_adata, min_dist = 0.1)
rna_adata.obsm['X_umap'] = rna_adata.obsm['X_umap']*np.array([-1,-1]) # flip for consistency

# Run K-NN and UMAP for ATAC data
sc.pp.neighbors(atac_adata, use_rep = 'X_umap_features', metric = 'manhattan', n_neighbors = 21)
sc.tl.umap(atac_adata, min_dist = 0.1)
atac_adata.obsm['X_umap'] = atac_adata.obsm['X_umap']*np.array([1,-1]) # flip for consistency

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
umap_kwargs = dict(color='topic_0', na_color="lightgrey")
sc.pl.umap(
    rna_adata,
    ax=ax[0],
    size=20,
    title="Expression Only",
    show=False,
    **umap_kwargs
)

sc.pl.umap(
    atac_adata,
    ax=ax[1],
    size=20,
    title="Accessibility Only",
    show=False,
    **umap_kwargs
)

plt.tight_layout()
plt.savefig("/gpfs/Home/esm5360/MIRA/figures/ds011_joint_representations/cell_topic_knn_umap.png", dpi=200)

rna_adata, atac_adata = mira.utils.make_joint_representation(rna_adata, atac_adata)

sc.pp.neighbors(rna_adata, use_rep = 'X_joint_umap_features', metric = 'manhattan',
               n_neighbors = 20)
sc.tl.umap(rna_adata, min_dist = 0.1)
fig, ax = plt.subplots(1,1,figsize=(8,5))
sc.pl.umap(rna_adata, legend_loc = 'on data', ax = ax, size = 20,
          **umap_kwargs, title = '')
plt.savefig("/gpfs/Home/esm5360/MIRA/figures/ds011_joint_representations/joint_representation_knn_umap.png", dpi=200)

rna_adata.obs = rna_adata.obs.join(
    atac_adata.obs.add_prefix('ATAC_') # add a prefix so we know which AnnData the column came from
)

atac_adata.obsm['X_umap'] = rna_adata.obsm['X_umap']

mira.tl.get_cell_pointwise_mutual_information(rna_adata, atac_adata)

fig, ax = plt.subplots(1,1,figsize=(8,5))
sc.pl.umap(rna_adata, color = 'pointwise_mutual_information', ax = ax, vmin = 0,
          color_map='magma', frameon=False, add_outline=True, vmax = 3, size = 25)
plt.savefig("/gpfs/Home/esm5360/MIRA/figures/ds011_joint_representations/mutual_info_knn_umap.png", dpi=200)

mutual_info_score = mira.tl.summarize_mutual_information(rna_adata, atac_adata)
print("Mutual information score (0 - low concordence, 0.5 - high concordance)")
print(mutual_info_score)

cross_correlation = mira.tl.get_topic_cross_correlation(rna_adata, atac_adata)
clustermap = sns.clustermap(cross_correlation, vmin = 0,
               cmap = 'magma', method='ward',
               dendrogram_ratio=0.05, cbar_pos=None, figsize=(7,7))
clustermap.savefig("/gpfs/Home/esm5360/MIRA/figures/ds011_joint_representations/topic_cross_correlation_clustermap.png", dpi=200)

atac_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_atac_data_joint_representation.h5ad")
rna_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_rna_data_joint_representation.h5ad")
