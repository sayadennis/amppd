import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

din = '/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
dsum = '/projects/b1131/saya/amppd_v2/data_summary'
dout = '/projects/b1131/saya/amppd_v2/wgs/07_batch_rm'

## Load/generate variant count data and study labels 
X = pd.read_csv(f'{din}/vc_mx.csv', index_col=0)
y = pd.read_csv(f'{din}/label.csv', index_col=0)

## Run PCA on the input data 
pca = PCA()
Xtf = pca.fit_transform(X)

dim = Xtf.shape[1]

################################
#### Save PC-removed matrix ####
################################

n_rm = 20 # number of PCs to be removed --this was decided based on the visualization "amppd_wgs_tSNE.png" 

## First save the PCA-transformed matrix without the top PCs 
pd.DataFrame(Xtf[:,n_rm:], columns=np.arange(Xtf.shape[1]-n_rm), index=X.index).to_csv(f'{dout}/vc_mx_bcor_pca.csv', header=True, index=True)

## Next transform the data back to the original features with the top PCs removed 
cmp = pca.components_ # PCA components matrix: shape = m_features x m_features 
X_batch_rm = np.dot(Xtf[:,n_rm:], cmp[n_rm:,:]) # this will be back to shape = n_samples x m_features 
pd.DataFrame(X_batch_rm, columns=X.columns, index=X.index).to_csv(f'{dout}/vc_mx_bcor_pca_reverse_tf.csv', header=True, index=True)

# Save the data distribution of this reverse-transformed matrix 
plt.hist(X_batch_rm.ravel(), bins=50)
plt.vlines(np.array([np.min(X_batch_rm), np.max(X_batch_rm)]), ymin=0, ymax=4*1e+6, linestyles='dashed', colors=['red', 'red'], label='Min/Max')
plt.xlabel('Values of reverse-transformed matrix')
plt.ylabel('Counts')
plt.ylim(0,3.5*1e+6)
plt.title('Histogram of reverse-transformed \nvariant count mtx values w/ top 20 PCs removed')
plt.legend()
plt.savefig(f'{dsum}/reverse_pca_transformed_rm20_data_histogram.png')
plt.close()
