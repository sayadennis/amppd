#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 24:00:00
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="amppd_v2_create_samplesets"
#SBATCH --output=pd_project/out/amppd_v2_create_samplesets.out

module load bcftools/1.4-6

bcftools query --list-samples /projects/b1131/saya/amppd_v2/wgs/01_split_normalized/chr22.splitnorm.vcf.gz > /projects/b1131/saya/amppd_v2/wgs/samplesets/amppd_v2_samples_all.txt

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

python pd_project/processing/step2_create_samplesets.py
