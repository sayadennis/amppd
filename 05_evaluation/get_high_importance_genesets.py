# from distutils.command.clean import clean
# from sre_parse import _OpGroupRefExistsType
import numpy as np
import pandas as pd
import pickle
import xgboost
import matplotlib.pyplot as plt

#### Define directory names ####
inputdir='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
labeldir='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
pfdir='/home/srd6051/pd_project/model_performance/v2'
itpdir='/home/srd6051/pd_project/model_interpretation/v2'
ixdir='/projects/b1131/saya/amppd_v2/wgs/indices'
dout='pd_project/model_interpretation/v2'

#### Load performance records ####
pf = pd.read_csv(f'{pfdir}/amppd_v2_classicalml.csv', index_col=0)
pf_nmf = pd.read_csv(f'{pfdir}/amppd_v2_classicalml_nmf.csv', index_col=0)

print('Best performing model for classical ML: ', pf.index[np.argmax(pf['crossval_roc_auc'].values)])
print('Best performing model for classical ML + NMF: ', pf_nmf['model'][np.argmax(pf_nmf['crossval_roc_auc'].values)])
# both were XGB (3/11/2022)

#### Load models and factorized matrices ####
m_fn = '20220118_saved_best_XGB_vc_mx.p' # model filename
m_nmf_fn = '20220309_saved_best_nmf_XGB_vc_mx.p' # model filename for NMF 
mx_nmf_fn = '20220309_saved_factorized_mx_for_XGB_vc_mx.p' # factorized matrices for NMF

with open(m_fn, 'rb') as f:
    m = pickle.load(f)

with open(m_nmf_fn, 'rb') as f:
    m_nmf = pickle.load(f)

with open(mx_nmf_fn, 'rb') as f:
    mx_nmf = pickle.load(f)

#### Load input, target, and train/test indices ####
X = pd.read_csv(f'{inputdir}/vc_mx.csv', index_col=0)
y = pd.read_csv(f'{labeldir}/label.csv', index_col=0)

train_ix = pd.read_csv('%s/train_ix.csv' % (ixdir), header=None)
test_ix = pd.read_csv('%s/test_ix.csv' % (ixdir), header=None)

X_train, X_test = X.iloc[train_ix[0]], X.iloc[test_ix[0]]
y_train, y_test = y.iloc[train_ix[0]], y.iloc[test_ix[0]]

##############################################
#### Interpretation of classical ML model ####
##############################################

#### How many of the top features combined make up 50% of the feature importance? #### 
n_features_keep = np.where(np.cumsum(m.feature_importances_[np.argsort(-1*m.feature_importances_)])<=.5)[0][-1] # 240

gene_list = list(X.columns[np.argsort(-1*m.feature_importances_)[:n_features_keep]])

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

cleaned_gene_list = clean_genelist(gene_list)

## Save gene list into a text file 
with open(f'{itpdir}/high_importance_genes_20220118_saved_best_XGB_vc_mx.txt', 'w') as f:
    for item in cleaned_gene_list:
        f.write('%s\n' % item)

####################################################
#### Interpretation of NMF + classical ML model ####
####################################################

## F is n_samples x k_components
## W is m_features x k_components

## How many components to consider to include 50% feature importance? 
n_important_components = np.where(np.cumsum(m_nmf.feature_importances_[np.argsort(-1*m_nmf.feature_importances_)])>0.5)[0][0] # 34

## Which components are those? 
ix_important_components = np.argsort(-1*m_nmf.feature_importances_)[:n_important_components]

## Which genes are included in those components? 
W = mx_nmf[2] # m_features x k_components
cmp_dict = {}
for i in range(len(ix_important_components)): # loop through components 
    cmp = W[:,ix_important_components[i]] # pick out weight vector of i'th most important component 
    genes = X.columns[cmp>1e-02] # get genes with non-zero coefficient in this component 
    genes = clean_genelist(genes)
    cmp_dict[i] = genes

for key in cmp_dict.keys():
    with open(f'{dout}/nmf_high_importance_genes_comp{key}.txt', 'w') as f:
        for gene in cmp_dict[key]:
            f.write(f'{gene}\n')

with open(f'{dout}/nmf_high_importance_genes.pkl', 'wb') as f:
    pickle.dump(cmp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

## Plot number of genes that belong to each component 
plt.bar(np.array(list(cmp_dict.keys()))+1, [len(cmp_dict[key]) for key in cmp_dict.keys()])
plt.title('Number of genes in top NMF components')
plt.ylabel('Number of genes in component')
plt.xlabel('Component with i\'th highest feature importance')
plt.tight_layout()
plt.savefig(f'{dout}/nmf_number_genes_important_components.png')
plt.close()

## Plot the feature importance of these components 
plt.bar(np.arange(len(cmp_dict.keys()))+1, m_nmf.feature_importances_[np.argsort(-1*m_nmf.feature_importances_)[:n_important_components]])
plt.title('Feature importance of top NMF components')
plt.ylabel('Feature importance score (XGB)')
plt.xlabel('Component with i\'th highest feature importance')
plt.tight_layout()
plt.savefig(f'{dout}/nmf_feature_importance_important_components.png')
plt.close()
