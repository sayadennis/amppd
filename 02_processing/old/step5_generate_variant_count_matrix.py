import sys
import numpy as np

sys.path.append("pd_project/src")
import vc_matrix

chrlist =[]
for n in np.arange(1,23):
    chrlist.append("chr" + str(n))

for chrn in chrlist:
    vc_matrix.write_matrix(
        fin_pattern = "/projects/b1131/saya/amppd/delMuts_filtered/{}_*".format(chrn), 
        out_fn = "/projects/b1131/saya/amppd/vc_matrix/vc_matrix_{}.csv".format(chrn)
    )
