import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

din='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
dout='/projects/b1131/saya/amppd_v2/data_summary'

## Load data
X = pd.read_csv(f'{din}/vc_mx.csv', index_col=0)
y = pd.read_csv(f'{din}/label.csv', index_col=0)

studies = {
    'BF' : 0, 
    'PP' : 1, 
    'HB' : 2, 
    'LC' : 3, 
    'PD' : 4, 
    'SU' : 5, 
    'SY' : 6, 
    'LB' : 7
}

y['study'] = None
for i in y.index:
    y.loc[i,'study'] = studies[i[:2]]

pca = PCA()
Xtf = pca.fit_transform(X)

# plot cumulative sum of this 
# pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.axhline(.9, color='r', linestyle='--')
ax.set_title(f'Cumulative Explained Variance of PCA')
plt.tight_layout()
fig.savefig(f'{dout}/amppd_wgs_pca_explained_variance.png')
plt.close()

# plot the first two PCs to see if they correspond to cohort studies 
fig, ax = plt.subplots(figsize=(9,5))
ax.scatter(Xtf[:,0], Xtf[:,1], c=y['study'].values, cmap='Set1', label=y['study'], s=4)
ax.set_title(f'First two PCs of AMP-PD variant counts')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.tight_layout()
fig.savefig(f'{dout}/amppd_wgs_pca_first_two_PCs.png')
plt.close()


dim = X.shape[1]

## tSNE with principal components removed 
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(9,5))
for i in range(8):
    row = i//4
    col = i%4
    X_tSNE = TSNE(n_components=2, init='random').fit_transform(X=Xtf[:,i*5:])
    scatter = ax[row,col].scatter(X_tSNE[:,0], X_tSNE[:,1], c=y['study'].values, cmap='Set1', label=y['study'], s=4)
    ax[row,col].set_title(f'{i*5} PCs removed')
    # if i==0:
    #     ax.legend(handles=scatter.legend_elements()[0], labels=list(studies.keys()), title='study', bbox_to_anchor=(1.05, 1.0), loc='upper left')
fig.suptitle('tSNE plot of AMP-PD variant counts with top PCs removed')
plt.tight_layout()
fig.savefig(f'{dout}/amppd_wgs_tSNE.png')
plt.close()

## UMAP
X_umap = umap.UMAP().fit_transform(X)

fig, ax = plt.subplots(figsize=(9,5))
scatter = ax.scatter(X_umap[:,0], X_umap[:,1], c=y['study'].values, cmap='Set1', label=y['study'], s=4)
ax.set_title(f'UMAP plot of AMP-PD variant counts ({dim} dimensions)')
ax.legend(handles=scatter.legend_elements()[0], labels=list(studies.keys()), title='study', bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
fig.savefig(f'{dout}/amppd_wgs_UMAP.png')
plt.close()
