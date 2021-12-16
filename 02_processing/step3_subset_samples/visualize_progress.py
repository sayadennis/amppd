import os
import glob
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

din='/projects/b1131/saya/amppd_v2/wgs/02_subsets'
dout='/projects/b1131/saya/amppd_v2/data_summary/subset_progress'

flist=glob.glob(f'{din}/chr*_subset*.vcf.gz')

progress = {}

for i in range(1,23):
    flist=glob.glob(f'{din}/chr{i}_subset*.vcf.gz')
    progress[i] = len(flist) # save the number of subsets that are complete 

## plot and save progress 
labels=list(np.arange(1,23))
width = 0.35 
timestamp = datetime.datetime.now().strftime('%Y_%m_%d')

fig, ax = plt.subplots()
ax.set_ylim(0, 42)
ax.bar(labels, [progress[i] for i in range(1,23)], width, label='complete')
ax.bar(labels, [42-progress[i] for i in range(1,23)], width, bottom=[progress[i] for i in range(1,23)], label='pending')
ax.set_xticks(np.arange(1,23))
ax.set_xticklabels(np.arange(1,23), fontsize=10)
ax.set_ylabel('Progress')
ax.set_xlabel('Chromosome')
ax.set_title('Subsetting Progress')
ax.legend()
fig.savefig(f'{dout}/{timestamp}_subset_progress.png')
