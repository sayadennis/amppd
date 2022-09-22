#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=0G
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="amp_exp"
#SBATCH --output=pd_project/out/v2_amppd_explore_expression.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/codes/03_data_summary/v2_amppd_explore_expression.py
