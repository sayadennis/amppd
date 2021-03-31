#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --mem=0
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="subsam"
#SBATCH --output=pd_project/amppd_outputs/subsetsamples.out

module load bcftools/1.4-6
module load htslib/1.10.1 # for tabix

din="/path/to/data/norm_wgs_amppd"
dout="/path/to/data/subsets_wgs_amppd"
dsampleset="/path/to/sampleset_amppd"

subsetsample () {
    local fin=$1 #fin=/path/to/data/norm_wgs_amppd/chr<num>.splitnorm.vcf.gz
    shortn=${fin:36:24} #shortn="chr<num>.splitnorm.vcf.gz"
    tmplist=$(echo $shortn | tr "." "\n") # split by "."
    tmparr=($tmplist) # convert to array
    chrname=${tmparr[0]} # take the first one i.e. just the "chr<num>"
    # First take care of the BF subset since filename cannot be matched with regex 
    fout=$chrname"_bf.vcf.gz"
    bcftools view \
        -O z \
        -o ${dout}/${fout} \
        --trim-alt-alleles \
        --samples-file "${dsampleset}/AMPPD_BF_samplenames.txt" \
        ${fin}
    tabix -p vcf ${dout}/${fout}
    # Next take care of the PD and PP subsets
    for samplefn in $(ls ${dsampleset}/AMPPD_*_samplenames_[1-3].[1-2].txt); do
        # Check if the below if...else statement works to obtain list of output file name
        # fout=<output_file.vcf.gz>
        if [[ $samplefn == ${dsampleset}"/AMPPD_PD_samplenames_1.1.txt" ]]; then
            subn="pd1-1"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PD_samplenames_1.2.txt" ]]; then
            subn="pd1-2"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PD_samplenames_2.1.txt" ]]; then
            subn="pd2-1"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PD_samplenames_2.2.txt" ]]; then
            subn="pd2-2"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PD_samplenames_3.1.txt" ]]; then
            subn="pd3-1"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PD_samplenames_3.2.txt" ]]; then
            subn="pd3-2"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PP_samplenames_1.1.txt" ]]; then
            subn="pp1-1"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PP_samplenames_1.2.txt" ]]; then
            subn="pp1-2"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PP_samplenames_2.1.txt" ]]; then
            subn="pp2-1"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PP_samplenames_2.2.txt" ]]; then
            subn="pp2-2"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PP_samplenames_3.1.txt" ]]; then
            subn="pp3-1"
        elif [[ $samplefn == ${dsampleset}"/AMPPD_PP_samplenames_3.2.txt" ]]; then
            subn="pp3-2"
        else
            echo "Something went wrong."
        fi
        fout=$chrname"_"$subn".vcf.gz"
        bcftools view \
            -O z \
            -o ${dout}/${fout} \
            --trim-alt-alleles \
            --samples-file $samplefn \
            ${fin}
        tabix -p vcf ${dout}/${fout}
    done
}

flist=$(ls ${din}/chr*.splitnorm.vcf.gz) # can change this list to smaller subset if prefer to not run for all files

for fn in $flist; do 
    subsetsample "$fn"
done
