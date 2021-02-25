#!/bin/bash
#SBATCH -A b1042
#SBATCH -t 48:00:00
#SBATCH -p genomics-himem
#SBATCH -N 1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=250G
#SBATCH --mail-user=<email>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="himem_delmuts"
#SBATCH --output=~/pd_project/amppd_outputs/step4_filter_deleterious_himem.out

module load python/anaconda3.6

python pd_project/scripts/step4_filter_deleterious.py
