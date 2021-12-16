import os
import datetime
import numpy as np
import pandas as pd

labelfn="/projects/b1131/saya/clinical_amppd/amp_pd_case_control.csv"
samplefn="/home/srd6051/sampleset_amppd/sample_list_amppd_wgs.txt"

data = pd.read_csv(labelfn)
wgs_samples = list(pd.read_csv(samplefn).values.T[0])
label_samples = list(data["participant_id"].values)

print("Summary information of AMP-PD data: logged {}".format(datetime.datetime.now()))
print("")

print("Number of samples that have WGS data: {}".format(len(wgs_samples)))
print("Number of samples with AMP-PD labels: {}".format(len(label_samples)))
print("")

print("Number of WGS samples that have label information available: {}".format(
    np.sum([wgs_samples[x] in label_samples for x in range(len(wgs_samples))])
))
print("Number of labeled samples that are present in the WGS set: {}".format(
    np.sum([label_samples[x] in wgs_samples for x in range(len(label_samples))])
))
print("")

data = data.iloc[[label_samples[x] in wgs_samples for x in range(len(label_samples))],:]

print("Number of samples that change from case to control in baseline to latest: {}".format(
    np.sum([
        (data.loc[i]["case_control_other_at_baseline"]=="Case") & (data.loc[i]["case_control_other_latest"]=="Control") for i in data.index
    ])
))
print("Number of samples that change from control to case in baseline to latest: {}".format(
    np.sum([
        (data.loc[i]["case_control_other_at_baseline"]=="Control") & (data.loc[i]["case_control_other_latest"]=="Case") for i in data.index
    ])
))
print("Number of samples that change from control to other in baseline to latest: {}".format(
    np.sum([
        (data.loc[i]["case_control_other_at_baseline"]=="Control") & (data.loc[i]["case_control_other_latest"]=="Other") for i in data.index
    ])
))
print("Number of samples that change from case to other in baseline to latest: {}".format(
    np.sum([
        (data.loc[i]["case_control_other_at_baseline"]=="Case") & (data.loc[i]["case_control_other_latest"]=="Other") for i in data.index
    ])
))
print("")
print("Number of samples that change from other to case in baseline to latest: {}".format(
    np.sum([
        (data.loc[i]["case_control_other_at_baseline"]=="Other") & (data.loc[i]["case_control_other_latest"]=="Case") for i in data.index
    ])
))
print("Number of samples that change from other to control in baseline to latest: {}".format(
    np.sum([
        (data.loc[i]["case_control_other_at_baseline"]=="Other") & (data.loc[i]["case_control_other_latest"]=="Control") for i in data.index
    ])
))
print("")


## Case vs. control summary 
casecontrolsummary = pd.DataFrame(0, index=["Case", "Control", "Other"], columns=data.columns[3:])
for i in range(data.shape[0]):
    cc_base = data.iloc[i]["case_control_other_at_baseline"]
    cc_late = data.iloc[i]["case_control_other_latest"]
    casecontrolsummary.loc[cc_base]["case_control_other_at_baseline"] += 1
    casecontrolsummary.loc[cc_late]["case_control_other_latest"] += 1

casecontrolsummary.to_csv("/projects/b1131/saya/amppd_wgssamples_casecontrol_summary.csv", header=True, index=True)

## Diagnosis summary
all_diag = list(set(data["diagnosis_at_baseline"].unique()) | set(data["diagnosis_latest"].unique()))
diagsummary = pd.DataFrame(0, index=all_diag, columns=data.columns[1:3])
for i in range(data.shape[0]):
    diag_base = data.iloc[i]["diagnosis_at_baseline"]
    diag_late = data.iloc[i]["diagnosis_latest"]
    diagsummary.loc[diag_base]["diagnosis_at_baseline"] += 1
    diagsummary.loc[diag_late]["diagnosis_latest"] += 1

diagsummary.to_csv("/projects/b1131/saya/amppd_wgssamples_diagnosis_summary.csv", header=True, index=True)

## Relationship between case/control and diagnosis 
# for any timepoint, what are the diagnosis of subjects being categorized as cases/controls? 
cc_diag_summary = pd.DataFrame(0, index=all_diag, columns=["Case", "Control", "Other"])
for i in range(data.shape[0]):
    diag_base = data.iloc[i]["diagnosis_at_baseline"]
    diag_late = data.iloc[i]["diagnosis_latest"]
    cc_base = data.iloc[i]["case_control_other_at_baseline"]
    cc_late = data.iloc[i]["case_control_other_latest"]
    cc_diag_summary.loc[diag_base][cc_base] += 1
    cc_diag_summary.loc[diag_late][cc_late] += 1

cc_diag_summary.to_csv("/projects/b1131/saya/amppd_wgssamples_casecontrol_diagnosis_summary.csv", header=True, index=True)
