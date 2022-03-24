#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="pd_ml"
#SBATCH --output=pd_project/out/amppd_wgs_classicalml.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

############################
#### Main modeling code ####
############################

inputdir='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
labeldir='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
outdir='/home/srd6051/pd_project/model_performance/v2'
ixdir='/projects/b1131/saya/amppd_v2/wgs/indices'

# python classical_ml/ClassicalML/run_classical_ml.py \
#     --input $inputdir/vc_mx.csv \
#     --label $labeldir/label.csv \
#     --outfn $outdir/amppd_v2_classicalml.csv \
#     --indexdir $ixdir \
#     --scoring roc_auc \
#     --savemodel "true"
# #

#############################################
#### modeling with batch-corrected input ####
#############################################

inputdir='/projects/b1131/saya/amppd_v2/wgs/07_batch_rm'

python classical_ml/ClassicalML/run_classical_ml.py \
    --input $inputdir/vc_mx_bcor_pca.csv \
    --label $labeldir/label.csv \
    --outfn $outdir/amppd_v2_classicalml_bcor_pca.csv \
    --indexdir $ixdir \
    --scoring roc_auc \
    --savemodel "true"
#

