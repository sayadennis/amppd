import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("amppd_project/src")
import AMPPDModelTraining
# import AlignInputTarget

seed = 0

din = "/path/to/ppmi/wes/variant_count_and_label"
dout = "amppd_project/model_performance"

data = pd.read_csv(os.path.join(din, "200825_aligned_applythres_vc.csv"), header=0, index_col=0)
target = pd.read_csv(os.path.join(din, "aligned_subtype.csv"), header=0, index_col=0)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=seed)
AMPPDModelTraining.record_tuning(
    X_train, y_train, X_test, y_test, 
    os.path.join(dout, "ppmi_crossval_record_tune.csv"),
    multiclass=True
)
