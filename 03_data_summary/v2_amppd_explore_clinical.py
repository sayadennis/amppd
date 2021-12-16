import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

din = '/projects/b1131/saya/amppd_v2/clinical'
dout = '/projects/b1131/saya/amppd_v2/data_summary'

cc = pd.read_csv(f'{din}/amp_pd_case_control.csv')
pt = pd.read_csv(f'{din}/amp_pd_participants.csv')
dup = pd.read_csv(f'{din}/amp_pd_participant_wgs_duplicates.csv')

rnainv = pd.read_csv(f'{din}/rna_sample_inventory.csv')
# jointgeno = pd.read_csv(f'{din}/wgs_gatk_joint_genotyping_samples.csv') # this table is useless - all values in column 1 and 2 are equal
# wgsinv = pd.read_csv(f'{din}/wgs_sample_inventory.csv') # this table is also useless - all values in column 1 and 2 are equal

#################################
#### Get case/control counts ####
#################################

casecontrolsummary = pd.DataFrame(0, index=["Case", "Control", "Other"], columns=cc.columns[3:])
for i in range(cc.shape[0]):
    cc_base = cc.iloc[i]["case_control_other_at_baseline"]
    cc_late = cc.iloc[i]["case_control_other_latest"]
    casecontrolsummary.loc[cc_base]["case_control_other_at_baseline"] += 1
    casecontrolsummary.loc[cc_late]["case_control_other_latest"] += 1

casecontrolsummary.to_csv(f'{dout}/amppd_v2_casecontrol_summary.csv', header=True, index=True)

## Diagnosis summary
all_diag = list(set(cc["diagnosis_at_baseline"].unique()) | set(cc["diagnosis_latest"].unique()))
diagsummary = pd.DataFrame(0, index=all_diag, columns=cc.columns[1:3])
for i in range(cc.shape[0]):
    diag_base = cc.iloc[i]["diagnosis_at_baseline"]
    diag_late = cc.iloc[i]["diagnosis_latest"]
    diagsummary.loc[diag_base]["diagnosis_at_baseline"] += 1
    diagsummary.loc[diag_late]["diagnosis_latest"] += 1

diagsummary.to_csv(f'{dout}/amppd_v2_diagnosis_summary.csv', header=True, index=True)

##################################
#### Assess data availability ####
##################################

avail = pd.DataFrame(columns=['category', 'data', '#patients with data', 'longitudinal'])

fdict = {
    'Demographic' : [
        'Demographics.csv', # 'Demographics_dictionary.csv'
        'Caffeine_history.csv', # 'Caffeine_history_dictionary.csv'
        'Family_History_PD.csv', # 'Family_History_PD_dictionary.csv'
        'PD_Medical_History.csv', # 'PD_Medical_History_dictionary.csv'
        'Smoking_and_alcohol_history.csv', # 'Smoking_and_alcohol_history_dictionary.csv'
    ],
    'Biospecimen' : [
        'Biospecimen_analyses_CSF_abeta_tau_ptau.csv', # 'Biospecimen_analyses_CSF_abeta_tau_ptau_dictionary.csv'
        'Biospecimen_analyses_CSF_beta_glucocerebrosidase.csv', # 'Biospecimen_analyses_CSF_beta_glucocerebrosidase_dictionary.csv'
        'Biospecimen_analyses_SomaLogic_plasma.csv', # 'Biospecimen_analyses_SomaLogic_plasma_dictionary.csv'
        'Biospecimen_analyses_other.csv', # 'Biospecimen_analyses_other_dictionary.csv'
    ],
    'Imaging' : [
        'DaTSCAN_SBR.csv', # DaTSCAN_SBR_dictionary.csv
        'DaTSCAN_visual_interpretation.csv', # DaTSCAN_visual_interpretation_dictionary.csv
        'MRI.csv', # MRI_dictionary.csv
        'DTI.csv' # DTI_dictionary.csv
    ],
    'Clinical Score' : [
    'Epworth_Sleepiness_Scale.csv', # Epworth_Sleepiness_Scale_dictionary.csv
    'MDS_UPDRS_Part_I.csv', # MDS_UPDRS_Part_I_dictionary.csv
    'MDS_UPDRS_Part_II.csv', # MDS_UPDRS_Part_II_dictionary.csv
    'MDS_UPDRS_Part_III.csv', # MDS_UPDRS_Part_III_dictionary.csv
    'MDS_UPDRS_Part_IV.csv', # MDS_UPDRS_Part_IV_dictionary.csv
    'MOCA.csv', # MOCA_dictionary.csv
    'REM_Sleep_Behavior_Disorder_Questionnaire_Mayo.csv', # REM_Sleep_Behavior_Disorder_Questionnaire_Mayo_dictionary.csv
    'REM_Sleep_Behavior_Disorder_Questionnaire_Stiasny_Kolster.csv', # REM_Sleep_Behavior_Disorder_Questionnaire_Stiasny_Kolster_dictionary.csv
    'UPDRS.csv', # UPDRS_dictionary.csv
    'UPSIT.csv', # UPSIT_dictionary.csv
    'LBD_Cohort_Clinical_Data.csv', # LBD_Cohort_Clinical_Data_dictionary.csv # Lewy Body Dementia - there's overlap w other scores
    'LBD_Cohort_Path_Data.csv', # LBD_Cohort_Path_Data_dictionary.csv
    'MMSE.csv', # MMSE_dictionary.csv # "Mini-Mental State Examination" 
    'Modified_Schwab___England_ADL.csv', # Modified_Schwab___England_ADL_dictionary.csv # ADL = "Activities of Daily Living" Scale
    'PDQ_39.csv' # PDQ_39_dictionary.csv # Parkinson's Disease Questionnaire 
    ]
}

for category in fdict.keys():
    for fn in fdict[category]:
        data = pd.read_csv(f'{din}/clinical/{fn}')
        patient_list = list(data['participant_id'].unique())
        avail = avail.append({
            'category' : category,
            'data' : (fn.split('.')[0] if 'Biospecimen' not in fn else fn.split('.')[0].split('Biospecimen_analyses_')[1]),
            '#patients with data' : len(patient_list),
            'longitudinal' : None # might change how to record this data
        }, ignore_index=True)

avail.to_csv(f'{dout}/data_availability_summary.csv', header=True, index=False)

##############################################
#### Assess presence of longitudinal data ####
##############################################

for datatype in fdict.keys():
    if datatype=='Clinical Score':
        for fn in fdict[datatype]:
            dataname = fn.split('.')[0]
            data = pd.read_csv(f'{din}/clinical/{fn}')
            plt.hist(data['visit_month'])
            plt.title(f'Longitudinal data availability for {dataname}')
            plt.xlabel('Months')
            plt.ylabel('#Patients w/ observations')
            plt.savefig(f'{dout}/longitudinal_data_availability/{dataname}_longavail.png')
            plt.close()
    elif datatype=='Imaging':
        fn = 'Biospecimen_analyses_CSF_abeta_tau_ptau.csv'
        dataname = fn.split('.')[0]
        data = pd.read_csv(f'{din}/clinical/{fn}')
        plt.hist(data['visit_month'])
        plt.title(f'Longitudinal data availability for {dataname}')
        plt.xlabel('Months')
        plt.ylabel('#Patients w/ observations')
        plt.savefig(f'{dout}/longitudinal_data_availability/{dataname}_longavail.png')
        plt.close()
    elif datatype=='Demographic':
        fn = 'PD_Medical_History.csv'
        dataname = fn.split('.')[0]
        data = pd.read_csv(f'{din}/clinical/{fn}')
        plt.hist(data['visit_month'])
        plt.title(f'Longitudinal data availability for {dataname}')
        plt.xlabel('Months')
        plt.ylabel('#Patients w/ observations')
        plt.savefig(f'{dout}/longitudinal_data_availability/{dataname}_longavail.png')
        plt.close()
    else:
        continue
