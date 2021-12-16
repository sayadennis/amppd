#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="submit_av_loop"
#SBATCH --output=pd_project/out/step4_run_annovar.out

# for fn in $(ls pd_project/codes/02_processing/step4_run_annovar/run_annovar_*.sh); do
#     sbatch $fn
# done

for fn in $(ls pd_project/codes/02_processing/step4_run_annovar/run_annovar_chr2_subset[3-9]*.sh); do
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

# module load perl

# din="/projects/b1131/saya/amppd_v2/wgs/02_subsets"
# dout="/projects/b1131/saya/amppd_v2/wgs/03_annovar"
# dir_annovar="/projects/p30791/annovar"

# runANNOVAR () {
#     local fin=$1 #fin=/path/to/data/subsets_wgs_amppd/chr<num>_["bf"|"pd"|"pp"][*1-3][*"-[1-3]"].vcf.gz
#     pathlist=$(echo $fin | tr "/" "\n") # split by "/"
#     patharr=($pathlist)
#     shortfn=${patharr[4]} # get just the filename and not the whole path
#     # shortfn=${fin:39:24} # just the filename without the path 
#     # get fout 
#     tmplist=$(echo $shortfn | tr "." "\n") # split by "."
#     tmparr=($tmplist) # convert to array
#     chrsubn=${tmparr[0]} # chromosome and subset name i.e. "chr<num>_["bf"|"pd"|"pp"][*1-3][*"-[1-3]"]"
#     fout=${chrsubn}_av
#     # run annovar
#     perl ${dir_annovar}/table_annovar.pl \
#         $fin \
#         ${dir_annovar}/humandb/ \
#         -buildver hg38 \
#         -out $dout/$fout \
#         -remove \
#         -protocol refGene,knownGene,ensGene,avsnp150,dbnsfp35a,dbnsfp31a_interpro,exac03,gnomad211_exome,gnomad211_genome \
#         -operation g,g,g,f,f,f,f,f,f \
#         -nastring . \
#         -vcfinput
# }

# flist=$(ls ${din}/chr*_*.vcf.gz) # can change this list to smaller subset if prefer to not run for all files

# for fn in $flist; do 
#     runANNOVAR $fn
# done
