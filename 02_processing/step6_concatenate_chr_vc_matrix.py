import os
import sys
import glob
import numpy as np
import pandas as pd

fpattern = "/projects/b1131/saya/amppd/vc_matrix/*"

mx_dict = {}

for fn in glob.glob(fpattern):
    vcf = pd.read_csv(fn, header=0, index_col=0) # patient IDs as index so that join is easier later
    chrn = fn.split("_")[-1].split(".")[0]
    mx_dict[chrn] = vcf

all_mx=pd.DataFrame()
col_num=0

for i in np.arange(1,23):
    chrn = "chr" + str(i)
    if chrn == "chr1":
        col_num += mx_dict[chrn].shape[1]
        all_mx = mx_dict[chrn]
    else:
        col_num += mx_dict[chrn].shape[1]
        all_mx = pd.concat([all_mx, mx_dict[chrn]], axis=1)

if col_num == all_mx.shape[1]:
    all_mx.to_csv("/projects/b1131/saya/amppd/vc_matrix/vc_matrix_del.csv",header=True, index=True)
else:
    print("Column numbers don't add up: {} and {}".format(col_num, all_mx.shape[1]))
