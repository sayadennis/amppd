#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomicslong
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 240:00:00
#SBATCH --mem=0
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="chr18_subset"
#SBATCH --output=pd_project/out/step3_subset_samples_chr18.out

module load bcftools/1.4-6
module load htslib/1.10.1 # for tabix

din="/projects/b1131/saya/amppd_v2/wgs/01_split_normalized"
dout="/projects/b1131/saya/amppd_v2/wgs/02_subsets"
dsampleset="/projects/b1131/saya/amppd_v2/wgs/samplesets"

subsetsample () {
    local fin=$1 #fin=/path/to/data/norm_wgs_amppd/chr<num>.splitnorm.vcf.gz
    shortn="$(cut -d'/' -f8 <<< "$fin")"
    chrname="$(cut -d'.' -f1 <<< "$shortn")"
    # Loop through files that has subset patient IDs and create subset VCFs 
    for subfn in $(ls ${dsampleset}/*[0-9].txt); do # subfn="/whole/path/to/amppd_v2_sampleset_<num>.txt"
        subshortfn="$(cut -d'/' -f8 <<< "$subfn")" # subshortfn="amppd_v2_sampleset_<num>.txt"
        subn="$(cut -d'.' -f1 <<< "$subshortfn")" # subn="amppd_v2_sampleset_<num>"
        subn="$(cut -d'_' -f4 <<< "$subn")" # subn="<num>"
        fout=$chrname"_subset"$subn".vcf.gz"
        bcftools view \
            -O z \
            -o ${dout}/${fout} \
            --trim-alt-alleles \
            --samples-file $subfn \
            ${fin}
        tabix -p vcf ${dout}/${fout}
    done
}

subsetsample "${din}/chr18.splitnorm.vcf.gz"
