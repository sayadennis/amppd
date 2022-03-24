import numpy as np
import pandas as pd

din='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
ddup='/projects/b1131/saya/amppd_v2/clinical'
dout='/projects/b1131/saya/amppd_v2/data_summary'

## Load data
# X = pd.read_csv(f'{din}/vc_mx.csv', index_col=0)
y = pd.read_csv(f'{din}/label.csv', index_col=0)
dup = pd.read_csv(f'{ddup}/amp_pd_participant_wgs_duplicates.csv')

dup_records = pd.DataFrame()

for i in dup.index:
    if ((dup.loc[i,'participant_id'] in list(y.index)) & (dup.loc[i,'duplicate_sample_id'] in list(y.index))):
        dup_records = dup_records.append(dup.loc[i,:])

print('Number of records that need to be removed from dataset: ', dup_records.shape[0])


# #### Tests #### 

# for i in dup.index:
#     if (dup.loc[i,'participant_id'] in list(y.index)):
#         dup_records = dup_records.append(dup.loc[i,:])


# for i in dup.index:
#     if (dup.loc[i,'duplicate_sample_id'] in list(y.index)):
#         dup_records = dup_records.append(dup.loc[i,:])

