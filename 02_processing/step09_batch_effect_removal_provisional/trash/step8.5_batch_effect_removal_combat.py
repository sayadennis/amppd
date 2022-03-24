import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append('pd_project/codes/02_processing/step8.5_batch_effect_removal_provisional')
from combat import combat

ddata = '/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
dsum = '/projects/b1131/saya/amppd_v2/data_summary'

## Load/generate variant count data and study labels 
X = pd.read_csv(f'{ddata}/vc_mx.csv', index_col=0).T
y = pd.read_csv(f'{ddata}/label.csv', index_col=0)

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

study = y['study'] # pd.Series([x[:2] for x in y.index], index=y.index)

## Remove batch effect using combat: https://github.com/brentp/combat.py/blob/master/combat.py
Xcrt = combat(X, study).T # batch-corrected X

#########################################################
#### Visualize batch-corrected data with a tSNE plot ####
#########################################################

## Enumerate study labels 
# ## Reduce dimensionality 
# pca = PCA()
# Xtf = pca.fit_transform(X_cor)

# ## Plot cumulative sum of explained variance ratio of PCA 
# # pca.explained_variance_ratio_
# fig, ax = plt.subplots(figsize=(9,5))
# ax.plot(np.cumsum(pca.explained_variance_ratio_))
# ax.axhline(.9, color='r', linestyle='--')
# ax.set_title(f'Cumulative Explained Variance of PCA')
# plt.tight_layout()
# fig.savefig(f'{dsum}/amppd_wgs_pca_explained_variance_batchcorr.png')
# plt.close()

dim = Xcrt.shape[1]

## tSNE 
X_tSNE = TSNE(n_components=2, init='random').fit_transform(X=Xcrt)
fig, ax = plt.subplots(figsize=(9,5))
scatter = ax.scatter(X_tSNE[:,0], X_tSNE[:,1], c=y['study'].values, cmap='Set1', label=y['study'], s=4)
ax.set_title(f'tSNE of AMP-PD variant counts post-batch correction ({dim} dimensions)')
ax.legend(handles=scatter.legend_elements()[0], labels=list(studies.keys()), title='study', bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
fig.savefig(f'{dsum}/amppd_wgs_tSNE_batchcorr.png')
plt.close()

# ## UMAP
# X_umap = umap.UMAP().fit_transform(Xtf)

# fig, ax = plt.subplots(figsize=(9,5))
# ax.scatter(X_umap[:,0], X_umap[:,1], c=y['study'].values, cmap='Set1', label=y['study'], s=4)
# ax.set_title(f'UMAP of AMP-PD variant counts post-batch correction ({dim} dimensions)')
# ax.legend(handles=scatter.legend_elements()[0], labels=list(studies.keys()), title='study', bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# fig.savefig(f'{dsum}/amppd_wgs_UMAP_batchcorr.png')
# plt.close()
