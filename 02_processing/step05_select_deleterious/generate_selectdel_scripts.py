import numpy as np

dout='/home/srd6051/pd_project/codes/02_processing/step5_select_deleterious/array_scripts'

for chrn in np.arange(1,23):
    chrname=f'chr{chrn}'
    fout=f'{dout}/select_deleterious_{chrname}_array.sh'
    # set memory 
    if int(chrn)<=10:
        mem='--mem=500G'
    else:
        mem='--mem=250G'
    #
    with open(fout, 'w') as f:
        # write script
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -A b1042\n')
        f.write(f'#SBATCH -p genomics-himem\n')
        f.write('#SBATCH -n 1\n')
        f.write('#SBATCH -N 1\n')
        f.write('#SBATCH -t 48:00:00\n')
        f.write('#SBATCH --array=0-41\n')
        f.write(f'#SBATCH {mem}\n')
        f.write('#SBATCH --mail-user=sayarenedennis@northwestern.edu\n')
        f.write('#SBATCH --mail-type=START,END,FAIL\n')
        f.write(f'#SBATCH --job-name="del{chrn}"\n')
        f.write(f'#SBATCH --output=pd_project/out/select_del_{chrname}.%A_%a.out\n')
        f.write('\n')
        f.write('. ~/anaconda3/etc/profile.d/conda.sh\n')
        f.write('conda activate pdenv\n')
        f.write('\n')
        f.write(f'IFS=$\'\\n\' read -d \'\' -r -a chrsub < /projects/b1131/saya/amppd_v2/wgs/batchjob_chrsub_lists/{chrname}_subnames.txt\n')
        f.write('\n')
        f.write('fin=/projects/b1131/saya/amppd_v2/wgs/03_annovar/${chrsub[$SLURM_ARRAY_TASK_ID]}_av.hg38_multianno.txt\n')
        f.write('fout=/projects/b1131/saya/amppd_v2/wgs/04_deleterious/${chrsub[$SLURM_ARRAY_TASK_ID]}_av.del.tsv\n')
        f.write('\n')
        f.write('python pd_project/codes/02_processing/step5_select_deleterious/select_deleterious.py --fin $fin --fout $fout\n')
