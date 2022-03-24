import numpy as np
import pandas as pd

din = '/projects/b1131/saya/amppd_v2/clinical'
dout = '/projects/b1131/saya/amppd_v2/clinical'

clin = pd.read_csv(f'{din}/amp_pd_case_control.csv')
clin['label'] = None
for i in clin.index:
    if (clin.loc[i,'case_control_other_at_baseline']=='Case') & (clin.loc[i,'case_control_other_latest']=='Case'):
        clin.loc[i,'label'] = 1
    elif (clin.loc[i,'case_control_other_at_baseline']=='Control') & (clin.loc[i,'case_control_other_latest']=='Control'):
        clin.loc[i,'label'] = 0
    else:
        continue

clin[['participant_id', 'label']].to_csv(f'{dout}/clinical_int_label.csv', index=False, header=True)
