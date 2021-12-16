import glob
import numpy as np
import pandas as pd

din = '/projects/b1131/saya/amppd_v2/clinical'
dout = '/projects/b1131/saya/amppd_v2/data_summary'

cc = pd.read_csv(f'{din}/amp_pd_case_control.csv')
pt = pd.read_csv(f'{din}/amp_pd_participants.csv')
dup = pd.read_csv(f'{din}/amp_pd_participant_wgs_duplicates.csv')

