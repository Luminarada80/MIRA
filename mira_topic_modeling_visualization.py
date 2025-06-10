import scanpy as sc
import mira
import anndata
import matplotlib.pyplot as plt

data = anndata.read_h5ad("mira-datasets/data.h5ad")

model = mira.topic_model.load_model('mira-datasets/tutorial_model.pth')

model.predict(data)

# scanpy workflow #
sc.pp.neighbors(data, use_rep = 'X_umap_features', metric = 'manhattan')
sc.tl.umap(data, min_dist=0.1, negative_sample_rate=0.05)

ax = sc.pl.umap(data[np.random.choice(len(data), len(data))], frameon=False, color = 'batch',
               title = '', palette= ['#8f7eadff', '#c1e1e2ff'], show = False)
ax.set_title('UMAP projection')

sc.pl.umap(data, color = model.topic_cols, cmap='BuPu', ncols=3,
           add_outline=True, outline_width=(0.1,0), frameon=False)

model.impute(data)
model.get_batch_effect(data)

fig, ax = plt.subplots(1,2,figsize=(10,4.5), sharey=True)

gene = '112'
mira.pl.plot_disentanglement(data, gene = gene, hue = 'batch', palette=['#8f7eadff', '#c1e1e2ff'], ax = ax[0])
mira.pl.plot_disentanglement(data, gene = gene, palette='Greys', vmin = -1, ax = ax[1])

ax[0].set(title = 'Colored by batch', xlim = (-0.5,0.5))
ax[1].set(title = 'Colored by counts', xlim = (-0.5,0.5))