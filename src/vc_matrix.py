import os
import pandas as pd
import glob
from datetime import datetime

# This module allows you to parse through the delMuts-filtered pseudo-VCF files of AMP-PD,
# Find the genotype count (0/1/2), and 
# Write a line with variant counts in the appropriate gene column into the final matrix.

def get_format_vals(string):
    gt = string.split(":")[0]
    dp = int(string.split(":")[2])
    gq = int(string.split(":")[3])
    return gt, dp, gq

def get_count(gt_string):
    ct = 0
    gt_values = [int(gt_string.split("/")[0]), int(gt_string.split("/")[1])]
    for k in gt_values:
        if k >= 1:
            ct += 1
        else:
            continue
    return ct


def write_matrix(fin_pattern, out_fn):
    mx = pd.DataFrame(0, index=[], columns=[]) # Initialize empty matrix 
    for fn in glob.glob(fin_pattern): # Loop through the matched pseudo-VCFs
        vcf = pd.read_csv(fn, header=0, index_col=0) # Read pseudo-VCF
        subids = list(vcf.columns)[146:] # Define list of patient IDs 
        for subid in subids: # Check whether these subjects already have rows in mx, add row if not
            if subid not in mx.columns:
                mx = mx.append(pd.DataFrame(None, index=[subid], columns=mx.columns))
            else:
                continue
        # Below: loop through rows of VCF, count variants for each patient
        for i in range(vcf.shape[0]): # loop through lines of VCF
            ct_dict = {} # dictionary that sill store patient's mutation count
            gene_name = vcf["Gene.refGene"].iloc[i] # refGene gene name
            for subid in subids: # Loop through subjects and fill in ct_dict for this position
                gt, dp, gq = get_format_vals(vcf.iloc[i,[x==subid for x in list(vcf.columns)]].values[0]) # gt = "0/0", "1/0", "2/2" etc.
                if ((dp >= 10) & (gq >= 10)):
                    ct_dict[subid] = get_count(gt)
            # if there is already a column in mx named gene_name, put count value in mx.iloc[i, <column location>]
            # if not, create a new column named gene_names and put count calue in mx.iloc[i, <new column location>]
            if gene_name in list(mx.columns):
                for subid in subids:
                    mx.loc[subid][gene_name] = ct_dict[subid]
            else:
                mx[gene_name] = 0 # add new column with default value zero 
                for subid in subids:
                    mx.loc[subid][gene_name] = ct_dict[subid]
        # update on progress 
        print("Done counting for patient {}... {}".format(subid, datetime.now().strftime("%m-%d-%Y, %H:%M:%S")))
    mx.to_csv(out_fn, header=True, index=True)
