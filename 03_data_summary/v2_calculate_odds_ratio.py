import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

din='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
dout='/projects/b1131/saya/amppd_v2/data_summary'

## Load data
X = pd.read_csv(f'{din}/vc_mx.csv', index_col=0)
y = pd.read_csv(f'{din}/label.csv', index_col=0)

## Create study labels 
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

X = np.minimum(1, X)

## One-hot encode the study labels 
enc = OneHotEncoder()
y_onehot = pd.DataFrame(enc.fit_transform(y['study'].values.reshape(-1,1)).toarray(), index=X.index, columns=np.arange(len(studies)))

# Include the study labels into the data to correct for this effect
Xstudy = pd.concat([X, y_onehot], axis=1)

## Train logistic regression to calculate odds ratio 
lrm = LogisticRegression(max_iter=2000)
lrm.fit(Xstudy, y['label'])

ORs = np.array([np.exp(x) for x in lrm.coef_])

## Create clean gene list of the genes that made up 50% of feature importance 
def clean_genelist(gene_ls):
    cleaned_list = []
    for item in gene_ls:
        if ';' in item:
            cleaned_list = cleaned_list + item.split(';')
        else:
            cleaned_list.append(item)
    cleaned_list = list(dict.fromkeys(cleaned_list))
    return cleaned_list

## Find columns with high odds ratio
for thres in [1.1, 1.15, 1.2]:
    orfeatures = list(Xstudy.columns[np.where(ORs>=thres)[1]])
    # separate genes and study cohorts 
    or_studies = list(compress(orfeatures, [type(x)==int for x in orfeatures]))
    or_genes = list(compress(orfeatures, [type(x)==str for x in orfeatures]))
    or_genes = clean_genelist(or_genes)
    # Save studies that appeared to have an effect 
    new_studies = dict([(value, key) for key, value in studies.items()])
    print(f'\n\n\n#### Studies that had OR>= {thres} ####')
    print([new_studies[x] for x in or_studies])
    # Save gene names with large OR
    with open(f'{dout}/genes_high_odds_ratio_thres{thres}.txt', 'w') as f:
        for item in or_genes:
            f.write(f'{item}\n')
