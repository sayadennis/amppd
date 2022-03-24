#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="cat_vcmx"
#SBATCH --output=pd_project/out/step7_concatenate_chr_vc_matrix.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/codes/02_processing/step7_concatenate_chr_vc_matrix/step7_concatenate_chr_vc_matrix.py
