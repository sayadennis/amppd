#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 4:00:00
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="cohort_sum"
#SBATCH --output=pd_project/amppd_outputs/210328_amppd_and_ppmi_cohort_summary.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/scripts/210328_amppd_and_ppmi_cohort_summary.py
