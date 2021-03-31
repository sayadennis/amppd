import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("pd_project/src")
import AMPPDModelTraining
import AlignInputTarget

seed = 0

data = pd.read_csv("/projects/b1131/saya/amppd/vc_matrix/vc_matrix_del_005_twotail.csv", header=0, index_col=0)
target = pd.read_csv("/projects/b1131/saya/amppd/amppd_wgs_target.csv", header=0, index_col=0)

aligner = AlignInputTarget.AlignInputTarget(data, target)
aligned_input = aligner.align_input()
aligned_target = aligner.align_target()

X_train, X_test, y_train, y_test = train_test_split(aligned_input, aligned_target, test_size=0.2, random_state=seed)
AMPPDModelTraining.record_tuning(
    X_train, y_train, X_test, y_test, 
    "pd_project/model_performance/test_crossval_module.csv"
)
