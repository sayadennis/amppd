import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dcln = '/projects/b1131/saya/amppd_v2/clinical'
didx='/projects/b1131/saya/amppd_v2/wgs/indices'
dlab='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
dout = '/projects/b1131/saya/amppd_v2/data_summary'

####################################################
#### Load train/test split and demographic data ####
####################################################

# Only load the target because we only need the patient IDs 
y = pd.read_csv(f'{dlab}/label.csv', index_col=0)

# Load train and test indices 
train_ix = pd.read_csv('%s/train_ix.csv' % (didx), header=None)
test_ix = pd.read_csv('%s/test_ix.csv' % (didx), header=None)

# Split to train and test 
y_train, y_test = y.iloc[train_ix[0]], y.iloc[test_ix[0]]

# Load demographic data 
demo = pd.read_csv(f'{dcln}/clinical/Demographics.csv')

#############
#### Age ####
#############

age_train = demo['age_at_baseline'].iloc[[x in y_train.index for x in demo['participant_id']]]
age_test = demo['age_at_baseline'].iloc[[x in y_test.index for x in demo['participant_id']]]

print(f'Age -- Train: {age_train.mean():.2f} (+/-{age_train.std():.2f}) // Test: {age_test.mean():.2f} (+/-{age_test.std():.2f})')

################
#### Gender ####
################

sex_train = demo['sex'].iloc[[x in y_train.index for x in demo['participant_id']]]
sex_test = demo['sex'].iloc[[x in y_test.index for x in demo['participant_id']]]

train_male = 100*(sex_train=='Male').sum()/sex_train.shape[0]
test_male = 100*(sex_test=='Male').sum()/sex_test.shape[0]

print(f'Sex -- Train: {train_male:.1f}% male // Test: {test_male:.1f}% male')

