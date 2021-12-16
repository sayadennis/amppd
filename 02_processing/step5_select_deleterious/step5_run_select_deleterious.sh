#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="submit_del"
#SBATCH --output=pd_project/out/step5_run_select_deleterious.out

# for fn in $(ls pd_project/codes/02_processing/step5_select_deleterious/select_deleterious_*.sh); do
#     sbatch $fn
# done

for fn in $(ls pd_project/codes/02_processing/step5_select_deleterious/select_deleterious_chr1_subset*.sh); do
    sbatch $fn
done

# for fn in $(ls pd_project/codes/02_processing/step4_run_annovar/run_annovar_chr2[0-2]_*.sh); do
#     sbatch $fn
# done

# for fn in $(ls pd_project/codes/02_processing/step4_run_annovar/run_annovar_chr1[0-9]_*.sh); do
#     sbatch $fn
# done

# for fn in $(ls pd_project/codes/02_processing/step4_run_annovar/run_annovar_chr[3-9]_*.sh); do
#     sbatch $fn
# done
