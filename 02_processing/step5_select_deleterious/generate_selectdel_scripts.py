import numpy as np

for chrn in np.arange(1,23):
    for subn in np.arange(42):
        chrname=f'chr{chrn}'
        subname=f'subset{subn}'
        fout=f'/home/srd6051/pd_project/codes/02_processing/step5_select_deleterious/select_deleterious_{chrname}_{subname}.sh'
        # set memory 
        if int(chrn)<=10:
            mem='--mem=0'
        else:
            mem='--mem=90G'
        #
        with open(fout, 'w') as f:
            # write script
            f.write('#!/bin/bash\n')
            f.write('#SBATCH -A b1042\n')
            f.write('#SBATCH -p genomics\n')
            f.write('#SBATCH -n 1\n')
            f.write('#SBATCH -N 1\n')
            f.write('#SBATCH -t 48:00:00\n')
            f.write(f'#SBATCH {mem}\n')
            f.write('#SBATCH --mail-user=sayarenedennis@northwestern.edu\n')
            f.write('#SBATCH --mail-type=FAIL\n')
            f.write(f'#SBATCH --job-name="selectdel{chrn}_{subn}"\n')
            f.write(f'#SBATCH --output=pd_project/out/select_deleterious_{chrname}_{subname}.out\n')
            f.write('\n')
            f.write('. ~/anaconda3/etc/profile.d/conda.sh\n')
            f.write('conda activate pdenv\n')
            f.write('\n')
            f.write(f'fin=/projects/b1131/saya/amppd_v2/wgs/03_annovar/{chrname}_{subname}_av.hg38_multianno.txt\n')
            f.write(f'fout=/projects/b1131/saya/amppd_v2/wgs/04_deleterious/{chrname}_{subname}_av.deleterious.tsv\n')
            f.write('\n')
            f.write('python pd_project/codes/02_processing/step5_select_deleterious/select_deleterious.py --fin $fin --fout $fout\n')
