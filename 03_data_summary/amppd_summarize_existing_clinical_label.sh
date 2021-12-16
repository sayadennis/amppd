#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 4:00:00
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="amppd_summary"
#SBATCH --output=pd_project/amppd_outputs/amppd_summarize_existing_clinical_label.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/scripts/amppd_summarize_existing_clinical_label.py
