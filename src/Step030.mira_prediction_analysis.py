import mira # type: ignore[import-untyped]

import os
import anndata # type: ignore[import-untyped]
import scanpy as sc # type: ignore[import-untyped]
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

plt.rcParams.update({'font.size': 14})

fig_dir = "/gpfs/Home/esm5360/MIRA/figures"

data = anndata.read_h5ad("mira-datasets/data.h5ad")

model = mira.topic_model.load_model('mira-datasets/tutorial_model.pth')

model.predict(data)

# scanpy workflow #

sc.settings.figdir = fig_dir
sc.pp.neighbors(data, use_rep = 'X_umap_features', metric = 'manhattan')
sc.tl.umap(data, min_dist=0.1, negative_sample_rate=0.05, save="nearest_neighbor_by_topic_composition.png")

ax: plt.Axes = sc.pl.umap(data[np.random.choice(len(data), len(data))], frameon=False, color = 'batch',
               title = '', palette= ['#8f7eadff', '#c1e1e2ff'], show = False)
ax.set_title('UMAP projection')
fig = ax.get_figure()
fig.savefig(os.path.join(fig_dir, "merged_batches_umap.png"), dpi=200)

sc.pl.umap(data, color = model.topic_cols, cmap='BuPu', ncols=3,
           add_outline=True, outline_width=(0.1,0), frameon=False,
           save="distribution_of_topics_on_umap.png")

fig, ax = plt.subplots(1,2,figsize=(10,4.5), sharey=True)

gene = '112'
mira.pl.plot_disentanglement(data, gene = gene, hue = 'batch', palette=['#8f7eadff', '#c1e1e2ff'], ax = ax[0])
mira.pl.plot_disentanglement(data, gene = gene, palette='Greys', vmin = -1, ax = ax[1])

ax[0].set(title = 'Colored by batch', xlim = (-0.5,0.5))
ax[1].set(title = 'Colored by counts', xlim = (-0.5,0.5))

fig.savefig(os.path.join(fig_dir, "disentanglement_of_technical_variation.png"), dpi=200)


