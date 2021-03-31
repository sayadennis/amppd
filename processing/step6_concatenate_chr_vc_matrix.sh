#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 8:00:00
#SBATCH -n 4
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="cat_vcmx"
#SBATCH --output=pd_project/amppd_outputs/concatenate_chr_vc_matrix.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/scripts/step6_concatenate_chr_vc_matrix.py
