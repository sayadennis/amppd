import os
import sys
import numpy as np
import pandas as pd
import getopt
import glob
from datetime import datetime

opts, extraparams = getopt.getopt(sys.argv[1:], 'c:', 
                                  ['chrn='])

for o,p in opts:
    if o in ['-c', '--chrn']:
        chrn = p

din='/projects/b1131/saya/amppd_v2/wgs/04_deleterious'
dout='/projects/b1131/saya/amppd_v2/wgs/05_vc_matrix'

# This script parses through the delMuts-filtered pseudo-VCF files of AMP-PD,
# Find the genotype count (0/1/2), and 
# Write a line with variant counts in the appropriate gene column into the final matrix.

##########################
#### Define functions ####
##########################

def get_format_vals(string):
    dp_missct = 0
    gq_missct = 0
    gt = string.split(':')[0]
    dp = string.split(':')[2]
    if dp == '.':
        dp = 0
        dp_missct += 1
    else:
        dp = int(dp)
    gq = string.split(':')[3]
    if gq == '.':
        gq = 0
        gq_missct += 1
    else:
        gq = int(gq)
    return gt, dp, gq, dp_missct, gq_missct

def get_count(gt_string):
    ct = 0
    gt_values = [int(gt_string.split('/')[0]), int(gt_string.split('/')[1])]
    for k in gt_values:
        if k >= 1:
            ct += 1
        else:
            continue
    return ct

def write_matrix(fin_pattern, out_fn):
    mx = pd.DataFrame(0, index=[], columns=[]) # Initialize empty matrix 
    for fn in glob.glob(fin_pattern): # Loop through the matched pseudo-VCFs
        vcf = pd.read_csv(fn, header=0, index_col=0, sep='\t') # Read pseudo-VCF
        subids = list(vcf.columns)[146:] # Define list of patient IDs 
        for subid in subids: # Check whether these subjects already have rows in mx, add row if not
            if subid not in mx.columns:
                mx = mx.append(pd.DataFrame(0, index=[subid], columns=mx.columns))
            else:
                continue
        # Below: loop through rows of VCF, count variants for each patient
        dp_miss = 0
        gq_miss = 0
        for i in range(vcf.shape[0]): # loop through lines of VCF
            ct_dict = {} # dictionary that sill store patient's mutation count
            gene_name = vcf['Gene.refGene'].iloc[i] # refGene gene name
            for subid in subids: # Loop through subjects and fill in ct_dict for this position
                gt, dp, gq, dp_missct, gq_missct = get_format_vals(
                    vcf.iloc[i,[x==subid for x in list(vcf.columns)]].values[0]
                ) # gt = '0/0', '1/0', '2/2' etc.
                dp_miss += dp_missct
                gq_miss += gq_missct
                if ((dp >= 10) & (gq >= 10)):
                    ct_dict[subid] = get_count(gt)
                else:
                    ct_dict[subid] = 0
            # if there is already a column in mx named gene_name, put count value in mx.iloc[i, <column location>]
            # if not, create a new column named gene_names and put count calue in mx.iloc[i, <new column location>]
            if gene_name in list(mx.columns):
                for subid in subids:
                    mx.loc[subid][gene_name] += ct_dict[subid]
            else:
                mx[gene_name] = 0 # add new column with default value zero 
                for subid in subids:
                    mx.loc[subid][gene_name] = ct_dict[subid]
        # update on progress 
        print('Done counting for file {}... {}'.format(fn, datetime.now().strftime('%m-%d-%Y, %H:%M:%S')))
        print('Missing rate of DP: {}'.format(dp_miss/(len(subids)*vcf.shape[0])))
        print('Missing rate of GQ: {}\n'.format(gq_miss/(len(subids)*vcf.shape[0])))
    mx.to_csv(out_fn, header=True, index=True)

##########################
#### Perform counting ####
##########################

write_matrix(
    fin_pattern = f'{din}/chr{chrn}_subset*_av.del.tsv', 
    out_fn = f'{dout}/vc_matrix_chr{chrn}.csv'
)

# chrlist =['chr' + str(n) for n in np.arange(1,23)]

# for chrn in chrlist:
#     write_matrix(
#         fin_pattern = f'{din}/{chrn}_subset*_av.del.tsv', 
#         out_fn = f'{dout}/vc_matrix_{chrn}.csv'
#     )
