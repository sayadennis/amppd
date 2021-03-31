#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --mem=0
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="splitnorm"
#SBATCH --output=pd_project/amppd_outputs/splitnorm.out

module load bcftools/1.4-6

din="../data/wgs_amppd"
dout="../data/splitnorm_amppd"
ref="../reference/hg38_ref/hg38.fa"

splitnormalize () {
    local fn=$1
    shortn=${fn:30:12} # this retrieves the name "chr<num>.vcf.gz"
    tmplist=$(echo $shortn | tr "." "\n") # splits by "."
    tmparr=($tmplist) # converts to array (this allows for indexing)
    chrname=${tmparr[0]} # takes only "chr<num>"
    bcftools norm -m-both -o ${dout}/${chrname}.step1.vcf ${din}/${chrname}.vcf.gz
    bcftools norm -O z -f $ref -o ${dout}/${chrname}.splitnorm.vcf.gz ${dout}/${chrname}.step1.vcf
    tabix -p vcf ${dout}/${chrname}.splitnorm.vcf.gz
}

flist=$(ls ${din}/*.vcf.gz) # can change this list to smaller subset if prefer to not run for all files

for fn in $flist; do
    splitnormalize "$fn"
done
