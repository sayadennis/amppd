import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.append("pd_project/src")
import applyThres

dn = "/projects/b1131/saya/amppd/vc_matrix"
vc_fn = "vc_matrix_del.csv"

threslist = [0, 0.01, 0.02, 0.03, 0.05]
vc = pd.read_csv(os.path.join(dn, vc_fn), header=0, index_col=0)

for thres in threslist[1:]:
    # Two-tail thresholding (eliminate feature if rare or frequent)
    f = applyThres.applyThres(vc, thres=thres, twotail=True)
    fout = vc_fn.split(".")[0] + "_" + str(thres).split(".")[0] + str(thres).split(".")[1] + "_twotail.csv"
    f.to_csv(os.path.join(dn, fout), header=True, index=True)
    # One-tail thresholding (eliminate feature only if it's rare)
    f = applyThres.applyThres(vc, thres=thres, twotail=False)
    fout = vc_fn.split(".")[0] + "_" + str(thres).split(".")[0] + str(thres).split(".")[1] + "_onetail.csv"
    f.to_csv(os.path.join(dn, fout), header=True, index=True)
