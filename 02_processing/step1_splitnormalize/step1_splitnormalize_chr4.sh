#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomicslong
#SBATCH -n 4
#SBATCH -t 96:00:00
#SBATCH --mem=0
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="splitnorm_v2_chr4"
#SBATCH --output=pd_project/out/splitnorm_v2_chr4.out

module load bcftools/1.4-6
module load htslib/1.10.1 # for tabix

din="/projects/b1131/saya/amppd_v2/wgs/gatk/vcf"
dout="/projects/b1131/saya/amppd_v2/wgs/01_split_normalized"
ref="/projects/p30791/hg38_ref/hg38.fa"

splitnormalize () {
    local fn=$1
    shortn=${fn:43:12} # this retrieves the name "chr<num>.vcf.gz"
    tmplist=$(echo $shortn | tr "." "\n") # splits by "."
    tmparr=($tmplist) # converts to array (this allows for indexing)
    chrname=${tmparr[0]} # takes only "chr<num>"
    bcftools norm -m-both -o ${dout}/${chrname}.step1.vcf ${din}/${chrname}.vcf.gz
    bcftools norm -O z -f $ref -o ${dout}/${chrname}.splitnorm.vcf.gz ${dout}/${chrname}.step1.vcf
    tabix -p vcf ${dout}/${chrname}.splitnorm.vcf.gz
}

splitnormalize ${din}/chr4.vcf.gz
