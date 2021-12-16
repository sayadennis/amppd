import os
import datetime
import numpy as np
import pandas as pd

clinicalfn = "/projects/b1131/saya/amppd/clinical_amppd/amp_pd_case_control.csv"
samplefn = "sampleset_amppd/sample_list_amppd_wgs.txt"

clinical = pd.read_csv(clinicalfn)

# all wgs_samples are present in label_samples
wgs_samples = list(pd.read_csv(samplefn).values.T[0])
label_samples = list(clinical["participant_id"].values)

target = pd.DataFrame(None, index=wgs_samples, columns=["label"])

for sample in wgs_samples:
    cc_baseline = clinical.iloc[[x == sample for x in clinical["participant_id"]],:]["case_control_other_at_baseline"].values[0]
    cc_latest = clinical.iloc[[x == sample for x in clinical["participant_id"]],:]["case_control_other_latest"].values[0]
    if ((cc_baseline == "Case") & (cc_latest == "Case")):
        target.loc[sample]["label"] = 1
    elif ((cc_baseline == "Control") & (cc_latest == "Control")):
        target.loc[sample]["label"] = 0
    else:
        continue

target.to_csv("/projects/b1131/saya/amppd/amppd_wgs_target.csv", header=True, index=True)
