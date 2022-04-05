import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dwgs = '/projects/b1131/saya/amppd_v2/wgs/05_vc_matrix'
dcln = '/projects/b1131/saya/amppd_v2/clinical'
ddata = '/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
dindex = '/projects/b1131/saya/amppd_v2/wgs/indices'

wgs = pd.read_csv(f'{dwgs}/vc_matrix.csv', index_col=0)
cln = pd.read_csv(f'{dcln}/clinical_int_label.csv', index_col=0)

########################################
#### Remove participants to exclude ####
########################################

## Will remove: 
# Participants who do not simply fall into case/control category (label changes over time)
# Participants who do not have both WGS and clinical data

## Remove participants with missing label
cln = cln.iloc[~pd.isnull(cln['label']).values,:]

## Select participants with both WGS and clinical data 
ol_participants = []
for participant_id in cln.index:
    if participant_id in wgs.index:
        ol_participants.append(participant_id)

wgs = wgs.loc[ol_participants,:]
cln = cln.loc[ol_participants,:]

##########################################
#### Create train and test index sets ####
##########################################

train_ix, test_ix = train_test_split(np.arange(len(cln.index)), random_state=11, shuffle=True, test_size=0.20, stratify=cln.values)

pd.DataFrame(list(train_ix)).to_csv(f'{dindex}/train_ix.csv', index=False, header=False)
pd.DataFrame(list(test_ix)).to_csv(f'{dindex}/test_ix.csv', index=False, header=False)

######################################################################
#### Perform frequency-based feature reduction using training set ####
######################################################################

train_wgs = wgs.iloc[train_ix,:]
print('Original feature size: ', train_wgs.shape[1])
keep_features = train_wgs.columns[(train_wgs==0).sum(axis=0)/train_wgs.shape[0]<0.95]
reduced_train_wgs = train_wgs[keep_features]
print('Reduced feature size: ', reduced_train_wgs.shape[1])

wgs[keep_features].to_csv(f'{ddata}/vc_mx.csv')
cln.to_csv(f'{ddata}/label.csv')
