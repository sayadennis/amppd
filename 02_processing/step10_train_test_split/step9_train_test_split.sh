#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --mem=20G
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="tt_split"
#SBATCH --output=pd_project/out/step9_train_test_split.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/codes/02_processing/step9_train_test_split/step9_train_test_split.py
