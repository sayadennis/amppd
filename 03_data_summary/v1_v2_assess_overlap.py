import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dv1='/projects/b1131/saya/amppd/clinical_amppd'
dv2='/projects/b1131/saya/amppd_v2/clinical'

v1=list(pd.read_csv(f'{dv1}/amp_pd_case_control.csv')['participant_id'].unique())
v2=list(pd.read_csv(f'{dv2}/amp_pd_case_control.csv')['participant_id'].unique())

ol=0

for item in v1:
    if item in v2:
        ol+=1

print(f'{ol} out of {len(v1)} V1 participants are found in V2 AMP-PD data.') # all found!
