"""
Test:
python generate_av_scripts.py --chrsub chr10_subset0 --fout Documents/test.sh
"""

import os
import sys
import getopt

opts, extraparams = getopt.getopt(sys.argv[1:], 'i:o:', ['chrsub=', 'fout='])

for o,p in opts:
    if o in ['-i', '--chrsub']:
        chrsub = p # e.g. chr10_subset0
    if o in ['-o', '--fout']:
        fout = p # e.g. /home/srd6051/pd_project/codes/02_processing/step4_run_annovar/run_annovar_chr10_subset0.sh

chrname=chrsub.split('_')[0] # e.g. 'chr10'
chrn=chrname[3:] # '10'
subname='sub'+chrsub.split('_')[1][6:] # e.g. 'sub0'
subn=chrsub.split('_')[1][6:]

if int(chrn)<=10:
    mem='--mem=0'
else:
    mem='--mem=90G'

with open(fout, 'w') as f:
    # write script
    f.write('#!/bin/bash\n')
    f.write('#SBATCH -A b1042\n')
    f.write('#SBATCH -p genomics\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -n 4\n')
    f.write('#SBATCH -t 48:00:00\n')
    f.write(f'#SBATCH {mem}\n')
    f.write('#SBATCH --mail-user=sayarenedennis@northwestern.edu\n')
    f.write('#SBATCH --mail-type=FAIL\n')
    f.write(f'#SBATCH --job-name="av{chrn}_{subn}"\n')
    f.write(f'#SBATCH --output=pd_project/out/run_annovar_amppd_{chrsub}.out\n')
    f.write('\n')
    f.write('module load perl\n')
    f.write('\n')
    f.write('din=/projects/b1131/saya/amppd_v2/wgs/02_subsets\n')
    f.write('dout=/projects/b1131/saya/amppd_v2/wgs/03_annovar\n')
    f.write('dir_annovar=/projects/p30791/annovar\n')
    f.write('\n')
    f.write(f'fin={chrsub}.vcf.gz\n')
    f.write(f'fout={chrsub}_av\n')
    f.write('\n')
    f.write('perl ${dir_annovar}/table_annovar.pl \\\n')
    f.write('    ${din}/${fin} \\\n')
    f.write('    ${dir_annovar}/humandb/ \\\n')
    f.write('    -buildver hg38 \\\n')
    f.write('    -out $dout/$fout \\\n')
    f.write('    -remove \\\n')
    f.write('    -protocol refGene,knownGene,ensGene,avsnp150,dbnsfp35a,dbnsfp31a_interpro,exac03,gnomad211_exome,gnomad211_genome \\\n')
    f.write('    -operation g,g,g,f,f,f,f,f,f \\\n')
    f.write('    -nastring . -vcfinput\n')
