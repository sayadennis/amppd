#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=0-9
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="arch_av"
#SBATCH --output=pd_project/out/step4.5_subset_samples_%a.out

cd /projects/b1131/saya/amppd_v2/wgs/03_annovar/

IFS=$'\n' read -d '\' -r -a chrsub < $(ls /projects/b1131/saya/amppd_v2/wgs/03_annovar | head)

echo ${chrsub[$SLURM_ARRAY_TASK_ID]}

# # --mem=10G
