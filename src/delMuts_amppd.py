import os
import datetime
import numpy as np
import pandas as pd

# a dictionary that will select the text file containing the corresponding sample names 
subset_dict = {
    "bf" : "sampleset_amppd/AMPPD_BF_samplenames.txt",
    "pd1-1" : "sampleset_amppd/AMPPD_PD_samplenames_1.1.txt",
    "pd1-2" : "sampleset_amppd/AMPPD_PD_samplenames_1.2.txt",
    "pd2-1" : "sampleset_amppd/AMPPD_PD_samplenames_2.1.txt",
    "pd2-2" : "sampleset_amppd/AMPPD_PD_samplenames_2.2.txt",
    "pd3-1" : "sampleset_amppd/AMPPD_PD_samplenames_3.1.txt",
    "pd3-2" : "sampleset_amppd/AMPPD_PD_samplenames_3.2.txt",
    "pp1-1" : "sampleset_amppd/AMPPD_PP_samplenames_1.1.txt",
    "pp1-2" : "sampleset_amppd/AMPPD_PP_samplenames_1.2.txt",
    "pp2-1" : "sampleset_amppd/AMPPD_PP_samplenames_2.1.txt",
    "pp2-2" : "sampleset_amppd/AMPPD_PP_samplenames_2.2.txt",
    "pp3-1" : "sampleset_amppd/AMPPD_PP_samplenames_3.1.txt",
    "pp3-2" : "sampleset_amppd/AMPPD_PP_samplenames_3.2.txt"
}

delExonicFuncMuts = ["frameshift insertion", "frameshift deletion", "stopgain", "stoploss"]
delFuncMuts = ["exonic;splicing", "splicing", "ncRNA_splicing"]

# define the filtering function 
def delMuts(fn, outdir): # fn = "chr<num>_<subset>_av.hg38_multianno.txt" where <subset> is "bf" or "p[pd][1-3]-[1-2]"
    # ## First define column datatypes 
    dtype_dict = {}
    # # get subset patient IDs
    subset_name = fn.split("/")[-1].split("_")[1]
    with open(subset_dict[subset_name], "r") as f:
        rawlines = f.readlines()
    sampleids = []
    for rawline in rawlines:
        sampleids.append(rawline.rstrip())
    # write dtypes to dictionary
    with open("pd_project/colnames.txt", "r") as f:
        lines = f.readlines()
    for i in range(len(lines)): # for default column names (non-sample-ID column names)
        dtype_dict[i] = lines[i].split("\t")[1].rstrip()
    for j in range(len(sampleids)):
        dtype_dict[j+len(lines)] = "object" # all these are dtype strings
    # read in the ANNOVAR output file
    print("\n\nStarting to read file {} ... {}".format(fn, datetime.datetime.now()))
    va = pd.read_csv(fn, sep="\t", skiprows=1, header=None, na_values=".", dtype=dtype_dict) # skip first row because header's column number doesn't match actual columns
    # set the column names
    default_colnames = []
    for line in lines:
        default_colnames.append(line.split("\t")[0])
    va.columns = default_colnames + sampleids
    # create filter
    print("Finished reading file. Creating filters ... {}".format(datetime.datetime.now()))
    funcfilter = []
    exonicfunc = va["ExonicFunc.refGene"]
    func = va["Func.refGene"]
    for i in range(va.shape[0]):
        if ((exonicfunc.iloc[i] in delExonicFuncMuts) | (func.iloc[i] in delFuncMuts)):
            funcfilter.append(True)
        else:
            funcfilter.append(False)
    # filter the VCF using filter
    print("Finished creating filters. Applying filters ... {}".format(datetime.datetime.now()))
    new_va = va.iloc[funcfilter,:]
    print("Finished applying filters. Saving file ... {}".format(datetime.datetime.now()))
    # save the filtered VCF to a new file
    outfn = fn.split("/")[-1][:-3] + "delMuts.csv"
    new_va.to_csv(os.path.join(outdir, outfn))
    print("...Done saving file: {} ... {}".format(os.path.join(outdir, outfn), datetime.datetime.now()))
    print("")
    return
