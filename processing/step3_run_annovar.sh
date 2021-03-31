#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomicslong
#SBATCH -n 4
#SBATCH -t 120:00:00
#SBATCH --mem=0
#SBATCH --mail-user=<email>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="av"
#SBATCH --output=pd_project/amppd_outputs/210115_annovar_amppd_chr1.out

module load perl

din="/path/to/data/subsets_wgs_amppd"
dout="/path/to/data/annovar_amppd"
dir_annovar="/path/to/annovar"

runANNOVAR () {
    local fin=$1 #fin=/path/to/data/subsets_wgs_amppd/chr<num>_["bf"|"pd"|"pp"][*1-3][*"-[1-3]"].vcf.gz
    pathlist=$(echo $fin | tr "/" "\n") # split by "/"
    patharr=($pathlist)
    shortfn=${patharr[4]} # get just the filename and not the whole path
    # shortfn=${fin:39:24} # just the filename without the path 
    # get fout 
    tmplist=$(echo $shortfn | tr "." "\n") # split by "."
    tmparr=($tmplist) # convert to array
    chrsubn=${tmparr[0]} # chromosome and subset name i.e. "chr<num>_["bf"|"pd"|"pp"][*1-3][*"-[1-3]"]"
    fout=${chrsubn}_av
    # run annovar
    perl ${dir_annovar}/table_annovar.pl \
        $fin \
        ${dir_annovar}/humandb/ \
        -buildver hg38 \
        -out $dout/$fout \
        -remove \
        -protocol refGene,knownGene,ensGene,avsnp150,dbnsfp35a,dbnsfp31a_interpro,exac03,gnomad211_exome,gnomad211_genome \
        -operation g,g,g,f,f,f,f,f,f \
        -nastring . \
        -vcfinput
}

flist=$(ls ${din}/chr*_*.vcf.gz) # can change this list to smaller subset if prefer to not run for all files

for fn in $flist; do 
    runANNOVAR $fn
done
