import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

din='/projects/b1131/saya/amppd_v2/rnaseq/rnaseq/salmon/quantification'
dout = '/projects/b1131/saya/amppd_v2/data_summary'

ixs=pd.read_csv(f'{din}/matrix.genes.tsv', sep='\t').columns
# data = pd.read_csv(f'{din}/aggregated.genes.tsv')

ids=[]
for ix in ixs[1:]:
    sample_id=ix.split('-')[0]+'-'+ix.split('-')[1]
    if sample_id not in ids:
        ids.append(sample_id)

print(f'Total number of patients with RNA data: {len(ids)}')

num_obs = [np.sum([x.startswith(sample_id) for x in ixs]) for sample_id in ids]
print(f'IQR of number of observations per patient for RNA data: {np.quantile(num_obs, [0, .25, .5, .75, 1.])}')

#### evaluate case/control composition of RNA data ####

didx='/projects/b1131/saya/amppd_v2/wgs/indices'
dlab='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'

# Only load the target because we only need the patient IDs 
y = pd.read_csv(f'{dlab}/label.csv', index_col=0)

# # Load train and test indices 
# train_ix = pd.read_csv('%s/train_ix.csv' % (didx), header=None)
# test_ix = pd.read_csv('%s/test_ix.csv' % (didx), header=None)

# # Split to train and test 
# y_train, y_test = y.iloc[train_ix[0]], y.iloc[test_ix[0]]

case_ids = y.iloc[y.values==1,:].index
cont_ids = y.iloc[y.values==0,:].index

print(f'Number of cases that have RNA data: {np.sum([x in ids for x in case_ids])}')
print(f'Number of cases that have RNA data: {np.sum([x in ids for x in cont_ids])}')

## Plot
plt.hist(num_obs, bins=30)
plt.ylim(0,250)
plt.xlabel('Number of observations')
plt.ylabel('Patient counts')
plt.title('Number of RNA Observations Per Patient')

plt.tight_layout()
plt.savefig(f'{dout}/longitudinal_data_availability/expression_num_obs_per_patient.png')
plt.close()
