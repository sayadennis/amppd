import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("pd_project/src")
# import AMPPDModelTraining
import AlignInputTarget

seed = 0

#### AMP-PD ####
data = pd.read_csv("/projects/b1131/saya/amppd/vc_matrix/vc_matrix_del_005_twotail.csv", header=0, index_col=0)
target = pd.read_csv("/projects/b1131/saya/amppd/amppd_wgs_target.csv", header=0, index_col=0)

aligner = AlignInputTarget.AlignInputTarget(data, target)
aligned_input = aligner.align_input()
aligned_target = aligner.align_target()

X_train, X_test, y_train, y_test = train_test_split(aligned_input, aligned_target, test_size=0.2, random_state=seed)

## Cohort composition
print("#### AMP-PD Cohort Composition ####\n")
print("## Overall counts")
print("Total: {}".format(aligned_input.shape[0]))
print("Train: {}".format(X_train.shape[0]))
print("Test: {}".format(X_test.shape[0]))
print("\n\n")

print("## Cases counts")
print("Total: {}".format(np.sum(aligned_target.values==1)))
print("Train: {}".format(np.sum(y_train.values==1)))
print("Test: {}".format(np.sum(y_test.values==1)))
print("\n\n")

print("## Controls counts")
print("Total: {}".format(np.sum(aligned_target.values==0)))
print("Train: {}".format(np.sum(y_train.values==0)))
print("Test: {}".format(np.sum(y_test.values==0)))
print("\n\n")

## Demographics
case_id = list(aligned_target[aligned_target.values==1].index)
control_id = list(aligned_target[aligned_target.values==0].index)
demo = pd.read_csv("/projects/b1131/saya/amppd/clinical_amppd/clinical/Demographics.csv")
demo = demo.iloc[[x in aligned_target.index for x in demo["participant_id"].values],:]
demo_cases = demo.iloc[[x in case_id for x in demo["participant_id"].values],:]
demo_control = demo.iloc[[x in control_id for x in demo["participant_id"].values],:]

print("#### AMP-PD demographic summary ####\n")
print("## Age")
print("Overall mean: {:.2f} // SD: {:.2f}".format(np.mean(demo["age_at_baseline"]), np.std(demo["age_at_baseline"])))
print("Cases mean: {:.2f} // SD: {:.2f}".format(np.mean(demo_cases["age_at_baseline"]), np.std(demo_cases["age_at_baseline"])))
print("Controls mean: {:.2f} // SD: {:.2f}".format(np.mean(demo_control["age_at_baseline"]), np.std(demo_control["age_at_baseline"])))
print("\n\n")

print("## Gender")
print("Overall counts:")
print("Male: {} ({:.2f}%)".format(np.sum(demo["sex"].values=="Male"), 100*np.sum(demo["sex"].values=="Male")/demo.shape[0]))
print("Female: {} ({:.2f}%)".format(np.sum(demo["sex"].values=="Female"), 100*np.sum(demo["sex"].values=="Female")/demo.shape[0]))
print("")

print("Cases counts:")
print("Male: {} ({:.2f}%)".format(np.sum(demo_cases["sex"].values=="Male"), 100*np.sum(demo_cases["sex"].values=="Male")/demo_cases.shape[0]))
print("Female: {} ({:.2f}%)".format(np.sum(demo_cases["sex"].values=="Female"), 100*np.sum(demo_cases["sex"].values=="Female")/demo_cases.shape[0]))
print("")

print("Controls counts:")
print("Male: {} ({:.2f}%)".format(np.sum(demo_control["sex"].values=="Male"), 100*np.sum(demo_control["sex"].values=="Male")/demo_control.shape[0]))
print("Female: {} ({:.2f}%)".format(np.sum(demo_control["sex"].values=="Female"), 100*np.sum(demo_control["sex"].values=="Female")/demo_control.shape[0]))
print("")

