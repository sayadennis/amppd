#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 48:00:00
#SBATCH -n 8
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="bibm_onset"
#SBATCH --output=pd_project/amppd_outputs/bibm_human_genomics_amppd_onset_modeling.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/scripts/bibm_human_genomics_amppd_onset_modeling.py
