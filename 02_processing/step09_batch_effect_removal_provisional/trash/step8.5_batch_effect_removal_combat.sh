#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="batchcorr"
#SBATCH --output=pd_project/out/step8.5_batch_effect_removal_combat.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/codes/02_processing/step8.5_batch_effect_removal_provisional/step8.5_batch_effect_removal_combat.py
