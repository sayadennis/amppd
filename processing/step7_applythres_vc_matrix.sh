#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 8:00:00
#SBATCH -n 4
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="applythres"
#SBATCH --output=pd_project/amppd_outputs/applythres_vc_matrix.out

module load python/anaconda3.6

python pd_project/scripts/step7_applythres_vc_matrix.py
