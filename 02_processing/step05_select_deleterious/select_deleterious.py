import numpy as np
import pandas as pd
import getopt
import sys

opts, extraparams = getopt.getopt(sys.argv[1:], 'i:o:', 
                                  ['fin=', 'fout='])

for o,p in opts:
    if o in ['-i', '--fin']:
        fin = p
    if o in ['-l', '--fout']:
        fout = p

#########################################
#### Pre-define necessary parameters ####
#########################################

## Define variants to retain 
delExonicFuncMuts = ['frameshift insertion', 'frameshift deletion', 'stopgain', 'stoploss']
delFuncMuts = ['exonic;splicing', 'splicing', 'ncRNA_splicing']

## Get sample IDs for the given subset number 
subn = fin.split('/')[-1].split('_')[1].split('t')[1] # e.g. subn='3' --splitting with 't' because it looks like 'subset3' 
samplefn = f'/projects/b1131/saya/amppd_v2/wgs/samplesets/amppd_v2_sampleset_{subn}.txt'
with open(samplefn, 'r') as f:
    lines = f.readlines()

sampleids = []
for line in lines:
    sampleids.append(line.strip())

## Define ANNOVAR output column names 
with open('pd_project/annovar_colnames.txt', 'r') as f:
    lines = f.readlines()

default_colnames = []
for line in lines:
    default_colnames.append(line.split('\t')[0])

all_colnames = default_colnames + sampleids

dtype_dict = {}
for i in range(len(lines)): # for default column names (non-sample-ID column names)
    dtype_dict[i] = lines[i].split('\t')[1].rstrip()

for j in range(len(sampleids)):
    dtype_dict[j+len(lines)] = 'object' # all these are dtype strings

#########################
#### Read input file ####
#########################

va = pd.read_csv(fin, sep='\t', skiprows=1, header=None, na_values='.', dtype=dtype_dict) # skiprows=1 b/c header's columns don't match actual columns

# fill in column names of input file 
va.columns = all_colnames

###############################################
#### Select only the deleterious mutations ####
###############################################

funcfilter = []
exonicfunc = va['ExonicFunc.refGene']
func = va['Func.refGene']
for i in range(va.shape[0]):
    if ((exonicfunc.iloc[i] in delExonicFuncMuts) | (func.iloc[i] in delFuncMuts)):
        funcfilter.append(True)
    else:
        funcfilter.append(False)

filtered = va.iloc[funcfilter,:]

###########################
#### Write output file ####
###########################

filtered.to_csv(fout, header=True, index=False, sep='\t') # double check removing index is fine 
