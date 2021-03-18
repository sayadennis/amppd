#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 48:00:00
#SBATCH -n 8
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="nmftwo005cv"
#SBATCH --output=~/pd_project/amppd_outputs/amppd_baseline_record_crossval.out

module load python/anaconda3

python pd_project/scripts/amppd_baseline_record_crossval.py
