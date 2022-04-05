# Run via interactive job on Quest HPC 
# srun --account=b1042 --time=1:00:00 --partition=genomics --mem=10G --pty bash -l
# module load R/4.1.0

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
#
BiocManager::install("biomaRt")

library(biomaRt)

genes <- scan("pd_project/model_interpretation/v2/high_importance_genes_20220324_saved_best_XGB_vc_mx.txt", what="", sep="\n")

ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")
data <- getBM(attributes=c('external_gene_name', 'ensembl_gene_id'),
             filters = 'hgnc_symbol',
             mart = ensembl,
             values = genes)
#

write.csv(data,"/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target/hugo_to_ensembl_gene_map.csv", row.names = FALSE, quote=FALSE)
