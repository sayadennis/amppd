#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="generate_av_scripts"
#SBATCH --output=pd_project/out/generate_av_scripts.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

din="/projects/b1131/saya/amppd_v2/wgs/02_subsets"
dout="/home/srd6051/pd_project/codes/02_processing/step4_run_annovar"

for fn in $(ls ${din}/*.gz); do # e.g. fn=/projects/b1131/saya/amppd_v2/wgs/02_subsets/chr10_subset0.vcf.gz
    shortfn="$(cut -d'/' -f8 <<< "$fn")" # e.g. shortfn=chr10_subset0.vcf.gz
    chrsubname="$(cut -d'.' -f1 <<< "$shortfn")" # e.g. chrsubname=chr10_subset0
    fout=${dout}/run_annovar_${chrsubname}.sh # e.g. fout=/home/srd6051/pd_project/codes/02_processing/step4_run_annovar/run_annovar_chr10_subset0.sh
    python pd_project/codes/02_processing/step4_run_annovar/generate_scripts/generate_av_scripts.py --chrsub $chrsubname --fout $fout;
done
