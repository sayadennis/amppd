#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomicslong
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=0-21
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="vc_mx"
#SBATCH --output=pd_project/out/step6_generate_variant_count_matrix_%a.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/codes/02_processing/step6_generate_variant_count_matrix/step6_generate_variant_count_matrix.py --chrn $((${SLURM_ARRAY_TASK_ID}+1))

# echo "python ${dn}/step6_generate_variant_count_matrix.py --chrn $((${SLURM_ARRAY_TASK_ID}+1))"
