import sys
import glob
sys.path.append("../src")
import delMuts_amppd

inputdir = "/projects/b1131/saya/annovar_amppd"
outdir = "/projects/b1131/saya/delMuts_filtered_himemout"

for fn in glob.glob("/projects/b1131/saya/annovar_amppd/*.txt"):
    # fn = "<data_dir>/annovar_amppd/chr<num>_<subset_name>_av.hg38_multianno.txt"
    delMuts_amppd.delMuts(fn, outdir)
